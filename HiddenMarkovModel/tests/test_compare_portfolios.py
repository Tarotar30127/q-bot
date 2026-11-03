"""Reject ambiguous notebook run selections before touching output files."""
from pathlib import Path

import pytest

from compare_portfolios import build


@pytest.mark.parametrize("window", ["2023", "2020"])
def test_custom_comparison_requires_all_allocations(window, tmp_path):
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="exactly these allocation"):
        build(window, run_dirs={"80/20": tmp_path / "run"}, output_dir=output)
    assert not output.exists()


def test_custom_comparison_rejects_reusing_one_run_for_all_allocations(tmp_path):
    output = tmp_path / "comparison"
    runs = {label: tmp_path / "run" for label in ("80/20", "90/10", "95/5")}
    with pytest.raises(ValueError, match="distinct run"):
        build("2023", run_dirs=runs, output_dir=output)
    assert not output.exists()


def test_custom_comparison_cannot_overwrite_an_input_report(tmp_path):
    runs = {label: tmp_path / label.replace("/", "_") for label in ("80/20", "90/10", "95/5")}
    with pytest.raises(ValueError, match="separate from the input"):
        build("2023", run_dirs=runs, output_dir=runs["80/20"])
    assert not any(Path(folder).exists() for folder in runs.values())
