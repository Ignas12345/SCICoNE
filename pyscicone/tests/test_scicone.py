from scicone import SCICoNE
from scicone import utils_10x
import pandas as pd
import numpy as np

def test_scicone():
    sci = SCICoNE()
    sci.run_tests()

def test_read_coverage_csv_basic(tmp_path):
    csv_df = pd.DataFrame(
        [
            ["cellA", "GRCh38_chr1", 0, 0, 50000, 10],
            ["cellA", "GRCh38_chr1", 1, 50000, 100000, 20],
            ["cellA", "GRCh38_chr2", 0, 0, 50000, 30],
            ["cellA", "GRCh38_chr2", 1, 50000, 100000, 40],
            ["cellB", "GRCh38_chr1", 0, 0, 50000, 15],
            ["cellB", "GRCh38_chr2", 0, 0, 50000, 35],
            ["cellB", "GRCh38_chr2", 1, 50000, 100000, 45],
        ],
        columns=["CB", "chrom", "bin", "start", "end", "count"]
    )

    csv_path = tmp_path / "scicone_test_coverage.csv"
    csv_df.to_csv(csv_path, index=False)

    out = utils_10x.read_coverage_csv(
        csv_path,
        current_chromosome_prefix="GRCh38_chr",
        cells_to_keep=["cellB", "cellA"],
        bins_to_exclude=[1]
    )

    assert out["bin_size"] == 50000
    assert list(out["unfiltered_chromosome_stops"].keys()) == ["1", "2"]
    assert out["unfiltered_chromosome_stops"]["1"] == 1
    assert out["unfiltered_chromosome_stops"]["2"] == 3
    assert out["filtered_chromosome_stops"]["1"] == 0
    assert out["filtered_chromosome_stops"]["2"] == 2
    assert out["unfiltered_counts"].shape == (2, 4)
    assert out["filtered_counts"].shape == (2, 3)
    assert np.array_equal(out["excluded_bins"], np.array([1]))

    # cell order follows cells_to_keep and missing bin (chr1,1) for cellB is imputed as 0
    assert out["unfiltered_counts"][0, 1] == 0

def test_scicone_read_coverage_from_csv_wrapper(tmp_path):
    csv_df = pd.DataFrame(
        [
            ["cell1", "GRCh38_chr1", 0, 0, 1000, 1],
            ["cell1", "GRCh38_chr1", 1, 1000, 2000, 2],
            ["cell2", "GRCh38_chr1", 0, 0, 1000, 3],
            ["cell2", "GRCh38_chr1", 1, 1000, 2000, 4],
        ],
        columns=["CB", "chrom", "bin", "start", "end", "count"]
    )
    csv_path = tmp_path / "scicone_test_wrapper_coverage.csv"
    csv_df.to_csv(csv_path, index=False)

    sci = SCICoNE()
    sci.read_coverage_from_csv(
        csv_path,
        current_chromosome_prefix="GRCh38_chr",
        cells_to_keep=["cell1", "cell2"]
    )

    assert sci.data["filtered_counts"].shape == (2, 2)
