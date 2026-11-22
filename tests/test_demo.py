import numpy as np

from spectra_probe.cli import main, run_demo
from spectra_probe.toy import toy_example


def test_decoded_list_drops_the_tail_and_probing_keeps_it():
    example = toy_example()
    decoded = example.direct_generation(3)
    assert int(np.sum(decoded > 0)) == 3
    assert np.all(example.likelihood.distribution > 0)
    assert example.js(example.likelihood.distribution) < example.js(decoded)


def test_demo_rows_and_cli(capsys):
    result = run_demo()
    by_method = {row["method"]: row for row in result["rows"]}
    assert by_method["Direct generation, top 3"]["tail_mass"] == 0.0
    assert by_method["SPECTRA, likelihood probing"]["categories_with_mass"] == 19
    assert main(["demo"]) == 0
    assert "SPECTRA, likelihood probing" in capsys.readouterr().out
