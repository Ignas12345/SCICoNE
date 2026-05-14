import scicone.utils as utils
import h5py
import numpy as np
import pandas as pd
import re

DEFAULT_BIN_SIZE_KB=20 # the 10x Genomics setting

def _sort_chromosomes_safe(chromosome_list):
    chromosome_list = np.array(chromosome_list).astype(str)
    try:
        return utils.sort_chromosomes(chromosome_list)
    except ValueError:
        return np.sort(chromosome_list)

def _extract_chromosome_stops_from_sizes(chromosome_order, bins_per_chromosome, bins_to_exclude=None):
    chr_ends = np.cumsum([bins_per_chromosome[ch] for ch in chromosome_order])
    chr_stops = dict()
    if bins_to_exclude is not None:
        bins_to_exclude = np.sort(np.array(bins_to_exclude).astype(int))
        for idx, pos in enumerate(chr_ends):
            excluded_count = np.searchsorted(bins_to_exclude, pos, side="left")
            chr_stops[chromosome_order[idx]] = pos - 1 - excluded_count
    else:
        for idx, pos in enumerate(chr_ends):
            chr_stops[chromosome_order[idx]] = pos - 1
    return chr_stops

def read_coverage_csv(csv_path, cell_column="CB", chromosome_column="chrom", bin_column="bin", start_column="start",
                      end_column="end", count_column="count", current_chromosome_prefix="",
                      cells_to_keep=None, bins_to_exclude=None):
    df = pd.read_csv(csv_path)
    required_columns = [cell_column, chromosome_column, bin_column, start_column, end_column, count_column]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    df = df[required_columns].copy()
    df[cell_column] = df[cell_column].astype(str)
    df[chromosome_column] = df[chromosome_column].astype(str)
    df[bin_column] = df[bin_column].astype(int)
    df[count_column] = df[count_column].astype(float)
    df[start_column] = df[start_column].astype(int)
    df[end_column] = df[end_column].astype(int)

    if current_chromosome_prefix:
        df[chromosome_column] = df[chromosome_column].str.replace(
            "^" + re.escape(current_chromosome_prefix), "", regex=True
        )

    if cells_to_keep is not None:
        cells_to_keep = [str(c) for c in cells_to_keep]
        df = df[df[cell_column].isin(cells_to_keep)].copy()
        if df.shape[0] == 0:
            raise ValueError("No rows remain after filtering by cells_to_keep.")
        cell_order = cells_to_keep
    else:
        cell_order = sorted(df[cell_column].unique().tolist())
        if df.shape[0] == 0:
            raise ValueError("Input CSV contains no rows.")

    bin_sizes = (df[end_column] - df[start_column]).unique()
    if len(bin_sizes) == 0:
        raise ValueError("No bins found in input CSV.")
    inferred_bin_size = bin_sizes[0]
    if not np.all(bin_sizes == inferred_bin_size):
        raise ValueError("All bins must have the same bin_size.")
    bin_size = int(inferred_bin_size)

    chromosome_order = _sort_chromosomes_safe(df[chromosome_column].unique())
    grouped_bins = {
        str(ch): np.sort(group[bin_column].unique().astype(int))
        for ch, group in df.groupby(chromosome_column)
    }
    chromosome_bin_map = {str(ch): grouped_bins[str(ch)] for ch in chromosome_order}
    all_columns = [(str(ch), int(b)) for ch in chromosome_order for b in chromosome_bin_map[ch]]
    full_column_index = pd.MultiIndex.from_tuples(all_columns, names=[chromosome_column, bin_column])

    matrix_df = df.pivot_table(
        index=cell_column,
        columns=[chromosome_column, bin_column],
        values=count_column,
        aggfunc="sum",
        fill_value=0
    )
    matrix_df = matrix_df.reindex(columns=full_column_index, fill_value=0)
    matrix_df = matrix_df.reindex(index=cell_order, fill_value=0)
    unfiltered_counts = matrix_df.to_numpy()

    bins_per_chromosome = {str(ch): len(chromosome_bin_map[ch]) for ch in chromosome_order}
    unfiltered_chromosome_stops = _extract_chromosome_stops_from_sizes(chromosome_order, bins_per_chromosome)

    if bins_to_exclude is None:
        excluded_bins = np.array([], dtype=int)
    else:
        all_excluded_bins = np.unique(np.array(bins_to_exclude).astype(int))
        invalid_excluded_bins = all_excluded_bins[
            (all_excluded_bins < 0) | (all_excluded_bins >= unfiltered_counts.shape[1])
        ]
        if len(invalid_excluded_bins) > 0:
            raise ValueError(
                "bins_to_exclude contains invalid indices: "
                f"{invalid_excluded_bins.tolist()} for {unfiltered_counts.shape[1]} total bins."
            )
        excluded_bins = all_excluded_bins

    is_excluded = np.zeros(unfiltered_counts.shape[1], dtype=bool)
    is_excluded[excluded_bins] = True
    filtered_counts = unfiltered_counts[:, ~is_excluded]
    filtered_chromosome_stops = _extract_chromosome_stops_from_sizes(
        chromosome_order,
        bins_per_chromosome,
        bins_to_exclude=excluded_bins
    )

    extracted_data = dict()
    extracted_data["unfiltered_counts"] = unfiltered_counts
    extracted_data["unfiltered_chromosome_stops"] = unfiltered_chromosome_stops
    extracted_data["filtered_counts"] = filtered_counts
    extracted_data["excluded_bins"] = excluded_bins
    extracted_data["filtered_cnvs"] = None
    extracted_data["filtered_chromosome_stops"] = filtered_chromosome_stops
    extracted_data["bin_size"] = bin_size
    return extracted_data

def read_hdf5(h5f_path, bins_to_exclude=None, downsampling_factor=1, remove_noisy_cells=True):
    extracted_data = dict()
    with h5py.File(h5f_path, 'r') as h5f:
        res = extract_corrected_counts_matrix(h5f, downsampling_factor=downsampling_factor, filter=False, remove_noisy_cells=remove_noisy_cells)
        extracted_data['unfiltered_counts'] = res['unfiltered_counts']
        extracted_data['unfiltered_chromosome_stops'] = extract_chromosome_stops(h5f, downsampling_factor=downsampling_factor)
        filtered_res = extract_corrected_counts_matrix(h5f, bins_to_exclude=bins_to_exclude, downsampling_factor=downsampling_factor, filter=True, remove_noisy_cells=remove_noisy_cells)
        extracted_data['filtered_counts'] = filtered_res['filtered_counts']
        extracted_data['excluded_bins'] = filtered_res['excluded_bins']
        res = extract_cnvs(h5f, bins_to_exclude=bins_to_exclude, downsampling_factor=downsampling_factor, filter=True, remove_noisy_cells=remove_noisy_cells)
        extracted_data['filtered_cnvs'] = res['filtered_cnvs']
        extracted_data['filtered_chromosome_stops'] = extract_chromosome_stops(h5f, bins_to_exclude=extracted_data['excluded_bins'], downsampling_factor=downsampling_factor)
        extracted_data['bin_size'] = DEFAULT_BIN_SIZE_KB*downsampling_factor*10**3

    return extracted_data

def merge_data_by_chromosome(h5f, key="normalized_counts", downsampling_factor=1, method='sum'):
    if method not in ['sum', 'median']:
        raise Exception('Method must be sum or median.')

    downsampling_factor = np.max([1, downsampling_factor])
    n_cells = h5f["cell_barcodes"][:].shape[0]
    sorted_chromosome_list = utils.sort_chromosomes(h5f["constants"]["chroms"][()].astype(str))

    matrix_list = []
    for ch in sorted_chromosome_list:
        chr_matrix = []
        mat = h5f[key][ch][:][0:n_cells]
        n_bins = mat.shape[1]
        if downsampling_factor > 1:
            for j in range(0, n_bins, downsampling_factor):
                start = j
                end = j+downsampling_factor
                if end > n_bins:
                    end = n_bins
                    start = end-downsampling_factor
                if method=='sum':
                    chr_matrix.append(np.nansum(mat[:, start:end], axis=1).reshape(-1,1)) # ignore NaNs
                elif method=='median':
                    median = np.nanmedian(mat[:, start:end], axis=1).reshape(-1,1) # ignore NaNs
                    if key == 'cnvs':
                        idx = median > 2
                        median[idx] = np.floor(median[idx]).astype(int)
                        idx = median < 2
                        median[idx] = np.ceil(median[idx]).astype(int)
                    chr_matrix.append(median)

            chr_matrix = np.concatenate(chr_matrix, axis=1)
        else:
            chr_matrix = mat # select only the cells, not cell groups

        matrix_list.append(chr_matrix)

    merged_matrix = np.concatenate(matrix_list, axis=1)
    return merged_matrix


def extract_corrected_counts_matrix(h5f, bins_to_exclude=None, downsampling_factor=1, filter=True, remove_noisy_cells=True):
    downsampling_factor = np.max([1, downsampling_factor])
    unfiltered_counts = merge_data_by_chromosome(h5f, key='normalized_counts', downsampling_factor=downsampling_factor, method='sum')
    sorted_chromosomes = utils.sort_chromosomes(h5f["constants"]["chroms"][()].astype(str))

    # Keep only single cells
    n_cells = h5f["cell_barcodes"].shape[0]
    filtered_counts = unfiltered_counts[:n_cells,:]
    if remove_noisy_cells:
        is_high_dimapd = np.array(h5f["per_cell_summary_metrics"]["is_high_dimapd"][()]).astype(bool)
        filtered_counts = filtered_counts[~is_high_dimapd,:]

    if filter:
        # Exclude unmappable bins
        is_mappable = []
        for ch in sorted_chromosomes:
            chr_is_mappable = []
            vec = h5f["genome_tracks"]["is_mappable"][ch][()]
            n_bins = vec.size
            if downsampling_factor > 1:
                for j in range(0, n_bins, downsampling_factor):
                    start = j
                    end = j+downsampling_factor
                    if end > n_bins:
                        end = n_bins
                        start = end-downsampling_factor
                    bin_is_mappable = np.any(vec[start:end])
                    chr_is_mappable.append(bin_is_mappable)
            else:
                chr_is_mappable = vec

            is_mappable = np.concatenate([is_mappable, chr_is_mappable])

        is_excluded = ~np.array(is_mappable, dtype=bool)
        excluded_bins = np.where(is_excluded)[0]
        if bins_to_exclude is not None:
            bins_to_exclude = np.array(bins_to_exclude)
            excluded_bins = np.unique(np.concatenate((excluded_bins, bins_to_exclude),0))
            is_excluded[excluded_bins] = True

        filtered_counts = filtered_counts[:, ~is_excluded]

        # Exclude noisy bins
        filtered_counts, noisy_bins = utils.filter_bins(filtered_counts, thres=3)
        excluded_bins = np.concatenate([excluded_bins, noisy_bins])
        excluded_bins = np.unique(excluded_bins)

        return dict(filtered_counts=filtered_counts, excluded_bins=excluded_bins)
    else:
        return dict(unfiltered_counts=filtered_counts)

def extract_chromosome_stops(h5f, bins_to_exclude=None, downsampling_factor=1):
    downsampling_factor = np.max([1, downsampling_factor])
    sorted_chromosomes = utils.sort_chromosomes(h5f["constants"]["chroms"][()].astype(str))

    if downsampling_factor > 1:
        chr_ends = np.cumsum([len(np.arange(0, n_bins, downsampling_factor)) for n_bins in h5f["constants"]["num_bins_per_chrom"]])
    else:
        chr_ends = np.cumsum(h5f["constants"]["num_bins_per_chrom"][()])

    chr_stops = dict()
    if bins_to_exclude is not None:
        bins_to_exclude = np.array(bins_to_exclude)
        for idx, pos in enumerate(chr_ends):
            chr_stops[sorted_chromosomes[idx]] = pos-1 - len(bins_to_exclude[np.where(bins_to_exclude < pos)[0]])
    else:
        for idx, pos in enumerate(chr_ends):
            chr_stops[sorted_chromosomes[idx]] = pos-1

    return chr_stops

def extract_cnvs(h5f, bins_to_exclude=None, downsampling_factor=1, filter=True, remove_noisy_cells=True):
    downsampling_factor = np.max([1, downsampling_factor])
    unfiltered_cnvs = merge_data_by_chromosome(h5f, key='cnvs', downsampling_factor=downsampling_factor, method='median')
    sorted_chromosomes = utils.sort_chromosomes(h5f["constants"]["chroms"][()].astype(str))

    # Keep only single cells
    n_cells = h5f["cell_barcodes"].shape[0]
    filtered_cnvs = unfiltered_cnvs[:n_cells,:]
    if remove_noisy_cells:
        is_high_dimapd = np.array(h5f["per_cell_summary_metrics"]["is_high_dimapd"][()]).astype(bool)
        filtered_cnvs = filtered_cnvs[~is_high_dimapd,:]

    if filter:
        # Exclude unmappable bins
        is_mappable = []
        for ch in sorted_chromosomes:
            chr_is_mappable = []
            vec = h5f["genome_tracks"]["is_mappable"][ch][()]
            n_bins = vec.size
            if downsampling_factor > 1:
                for j in range(0, n_bins, downsampling_factor):
                    start = j
                    end = j+downsampling_factor
                    if end > n_bins:
                        end = n_bins
                        start = end-downsampling_factor
                    bin_is_mappable = np.any(vec[start:end])
                    chr_is_mappable.append(bin_is_mappable)
            else:
                chr_is_mappable = vec

            is_mappable = np.concatenate([is_mappable, chr_is_mappable])

        is_excluded = ~np.array(is_mappable, dtype=bool)
        excluded_bins = np.where(is_excluded)[0]
        if bins_to_exclude is not None:
            bins_to_exclude = np.array(bins_to_exclude)
            excluded_bins = np.unique(np.concatenate((excluded_bins, bins_to_exclude),0))
            is_excluded[excluded_bins] = True

        filtered_cnvs = filtered_cnvs[:, ~is_excluded]

        return dict(filtered_cnvs=filtered_cnvs, excluded_bins=excluded_bins)
    else:
        return dict(unfiltered_cnvs=filtered_cnvs)
