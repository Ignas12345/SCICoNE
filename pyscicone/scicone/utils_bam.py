import scicone.utils as utils
import numpy as np
import pysam


SUPPORTED_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"x", "y"}
DEFAULT_BIN_SIZE_BP = 20000  # 20 kb


def read_bam(
    bam_path,
    bins_to_exclude=None,
    downsampling_factor=1,
    bin_size=DEFAULT_BIN_SIZE_BP,
    cell_tag="CB",
    fallback_barcode="bulk",
    allow_missing_cell_tag=False,
    min_mapping_quality=0,
    remove_duplicates=True,
    remove_secondary=True,
    remove_supplementary=True,
    remove_unmapped=True,
    min_reads_per_cell=1,
    remove_noisy_bins=True,
):
    downsampling_factor = max(1, int(downsampling_factor))
    effective_bin_size = int(bin_size) * downsampling_factor
    if effective_bin_size <= 0:
        raise ValueError("bin_size must be > 0")

    with pysam.AlignmentFile(bam_path, "rb") as bamf:
        selected_chromosomes = _get_selected_chromosomes(bamf)
        if len(selected_chromosomes) == 0:
            raise ValueError("No supported chromosomes found in BAM header.")

        chromosome_info = []
        for chrom in selected_chromosomes:
            ref_name = _resolve_reference_name(chrom, bamf.references)
            ref_length = bamf.get_reference_length(ref_name)
            n_bins = int(np.ceil(ref_length / effective_bin_size))
            chromosome_info.append(
                dict(chrom=chrom, ref_name=ref_name, n_bins=n_bins)
            )

        offsets = np.cumsum([0] + [info["n_bins"] for info in chromosome_info[:-1]])
        total_bins = int(np.sum([info["n_bins"] for info in chromosome_info]))

        cell_counts = {}
        for info, offset in zip(chromosome_info, offsets):
            for read in bamf.fetch(info["ref_name"]):
                if _skip_read(
                    read,
                    min_mapping_quality=min_mapping_quality,
                    remove_duplicates=remove_duplicates,
                    remove_secondary=remove_secondary,
                    remove_supplementary=remove_supplementary,
                    remove_unmapped=remove_unmapped,
                ):
                    continue

                cell_barcode = _extract_cell_barcode(
                    read,
                    cell_tag=cell_tag,
                    fallback_barcode=fallback_barcode,
                    allow_missing_cell_tag=allow_missing_cell_tag,
                )
                if cell_barcode is None:
                    continue

                bin_idx = min(read.reference_start // effective_bin_size, info["n_bins"] - 1)
                global_bin_idx = int(offset + bin_idx)
                cell_bin_counts = cell_counts.setdefault(cell_barcode, {})
                cell_bin_counts[global_bin_idx] = cell_bin_counts.get(global_bin_idx, 0) + 1

    if len(cell_counts) == 0:
        raise ValueError("No usable reads with cell barcodes were found in BAM.")

    cell_barcodes = sorted(cell_counts.keys())
    unfiltered_counts = np.zeros((len(cell_barcodes), total_bins), dtype=float)
    for row_idx, barcode in enumerate(cell_barcodes):
        for bin_idx, count in cell_counts[barcode].items():
            unfiltered_counts[row_idx, bin_idx] = count

    if min_reads_per_cell > 1:
        read_depth = np.sum(unfiltered_counts, axis=1)
        keep_cells = read_depth >= min_reads_per_cell
        unfiltered_counts = unfiltered_counts[keep_cells]
        cell_barcodes = [bc for i, bc in enumerate(cell_barcodes) if keep_cells[i]]
        if unfiltered_counts.shape[0] == 0:
            raise ValueError("No cells remain after min_reads_per_cell filtering.")

    is_excluded = np.zeros(total_bins, dtype=bool)
    if bins_to_exclude is not None:
        bins_to_exclude = np.array(bins_to_exclude, dtype=int).ravel()
        bins_to_exclude = bins_to_exclude[(bins_to_exclude >= 0) & (bins_to_exclude < total_bins)]
        is_excluded[bins_to_exclude] = True

    if remove_noisy_bins:
        kept_bins = np.where(~is_excluded)[0]
        prefiltered_counts = unfiltered_counts[:, kept_bins]
        filtered_counts_tmp, noisy_bins_tmp = utils.filter_bins(prefiltered_counts, thres=3)
        noisy_bins = kept_bins[noisy_bins_tmp]
        is_excluded[noisy_bins] = True
        filtered_counts = filtered_counts_tmp
    else:
        filtered_counts = unfiltered_counts[:, ~is_excluded]

    excluded_bins = np.where(is_excluded)[0]

    n_bins_per_chrom = [info["n_bins"] for info in chromosome_info]
    unfiltered_chromosome_stops = _extract_chromosome_stops(selected_chromosomes, n_bins_per_chrom)
    filtered_chromosome_stops = _extract_chromosome_stops(
        selected_chromosomes,
        n_bins_per_chrom,
        bins_to_exclude=excluded_bins,
    )

    return dict(
        unfiltered_counts=unfiltered_counts,
        unfiltered_chromosome_stops=unfiltered_chromosome_stops,
        filtered_counts=filtered_counts,
        excluded_bins=excluded_bins,
        filtered_chromosome_stops=filtered_chromosome_stops,
        bin_size=effective_bin_size,
        cell_barcodes=cell_barcodes,
    )


def _skip_read(
    read,
    min_mapping_quality=0,
    remove_duplicates=True,
    remove_secondary=True,
    remove_supplementary=True,
    remove_unmapped=True,
):
    if remove_unmapped and read.is_unmapped:
        return True
    if remove_secondary and read.is_secondary:
        return True
    if remove_supplementary and read.is_supplementary:
        return True
    if remove_duplicates and read.is_duplicate:
        return True
    if read.mapping_quality < min_mapping_quality:
        return True
    return False


def _extract_cell_barcode(
    read,
    cell_tag="CB",
    fallback_barcode="bulk",
    allow_missing_cell_tag=False,
):
    if read.has_tag(cell_tag):
        return read.get_tag(cell_tag)
    if allow_missing_cell_tag:
        return fallback_barcode
    return None


def _get_selected_chromosomes(bamf):
    normalized = []
    for ref_name in bamf.references:
        ref_name_lower = ref_name.lower()
        if ref_name_lower.startswith("chr"):
            ref_name_lower = ref_name_lower[3:]
        if ref_name_lower in SUPPORTED_CHROMOSOMES:
            normalized.append(ref_name_lower.upper())

    return list(utils.sort_chromosomes(np.array(normalized)))


def _resolve_reference_name(chromosome, references):
    candidate_names = {
        chromosome,
        chromosome.lower(),
        f"chr{chromosome}",
        f"chr{chromosome.lower()}",
    }
    for name in candidate_names:
        if name in references:
            return name
    raise ValueError(f"Could not find reference for chromosome {chromosome}")


def _extract_chromosome_stops(sorted_chromosomes, n_bins_per_chrom, bins_to_exclude=None):
    chr_ends = np.cumsum(n_bins_per_chrom)
    chr_stops = dict()
    if bins_to_exclude is None:
        for idx, pos in enumerate(chr_ends):
            chr_stops[sorted_chromosomes[idx]] = int(pos - 1)
        return chr_stops

    bins_to_exclude = np.array(bins_to_exclude, dtype=int).ravel()
    for idx, pos in enumerate(chr_ends):
        excluded_before = len(bins_to_exclude[np.where(bins_to_exclude < pos)[0]])
        chr_stops[sorted_chromosomes[idx]] = int(pos - 1 - excluded_before)

    return chr_stops
