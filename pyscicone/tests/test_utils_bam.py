from pathlib import Path

import pysam

from scicone import SCICoNE
from scicone import utils_bam


def _create_test_bam(path: Path, chromosome_prefix="chr"):
    header = {
        "HD": {"VN": "1.0", "SO": "coordinate"},
        "SQ": [{"SN": f"{chromosome_prefix}1", "LN": 500}, {"SN": f"{chromosome_prefix}2", "LN": 400}],
    }

    with pysam.AlignmentFile(path, "wb", header=header) as bamf:
        read_idx = 0
        for ref_id, n_bins in [(0, 5), (1, 4)]:
            for bin_idx in range(n_bins):
                for cell_barcode in ["cellA", "cellB"]:
                    read = pysam.AlignedSegment()
                    read.query_name = f"read_{read_idx}"
                    read.query_sequence = "A" * 20
                    read.flag = 0
                    read.reference_id = ref_id
                    read.reference_start = bin_idx * 100 + 5
                    read.mapping_quality = 60
                    read.cigar = ((0, 20),)
                    read.query_qualities = pysam.qualitystring_to_array("I" * 20)
                    read.set_tag("CB", cell_barcode)
                    bamf.write(read)
                    read_idx += 1

    pysam.index(str(path))


def test_read_bam_and_read_bam_api(tmp_path):
    bam_path = tmp_path / "toy.bam"
    _create_test_bam(bam_path)

    data = utils_bam.read_bam(str(bam_path), bin_size=100, remove_noisy_bins=True)
    assert data["unfiltered_counts"].shape == (2, 9)
    assert data["filtered_counts"].shape == (2, 9)
    assert data["unfiltered_chromosome_stops"] == {"1": 4, "2": 8}
    assert data["filtered_chromosome_stops"] == {"1": 4, "2": 8}
    assert data["excluded_bins"].size == 0

    sci = SCICoNE()
    sci.read_bam(str(bam_path), bin_size=100, remove_noisy_bins=True)
    assert sci.data["filtered_counts"].shape == (2, 9)


def test_read_bam_respects_excluded_bins(tmp_path):
    bam_path = tmp_path / "toy.bam"
    _create_test_bam(bam_path)

    data = utils_bam.read_bam(
        str(bam_path),
        bin_size=100,
        bins_to_exclude=[1, 7],
        remove_noisy_bins=False,
    )
    assert data["filtered_counts"].shape == (2, 7)
    assert set(data["excluded_bins"].tolist()) == {1, 7}
    assert data["filtered_chromosome_stops"] == {"1": 3, "2": 6}


def test_read_bam_chromosome_prefix_mapping(tmp_path):
    bam_path = tmp_path / "toy_prefixed.bam"
    _create_test_bam(bam_path, chromosome_prefix="GRCh_chr")

    data = utils_bam.read_bam(
        str(bam_path),
        bin_size=100,
        remove_noisy_bins=False,
        current_chromosome_name_prefix="GRCh_chr",
        desired_chromosome_name_prefix="chr",
    )
    assert data["unfiltered_counts"].shape == (2, 9)
    assert data["unfiltered_chromosome_stops"] == {"chr1": 4, "chr2": 8}

    sci = SCICoNE()
    sci.read_bam(
        str(bam_path),
        bin_size=100,
        remove_noisy_bins=False,
        current_chromosome_name_prefix="GRCh_chr",
        desired_chromosome_name_prefix="chr",
    )
    assert sci.data["filtered_chromosome_stops"] == {"chr1": 4, "chr2": 8}
