import subprocess
import pytest

COMMAND =  ["snakemake", "-F", "--cores", "1", "--configfile", "tests/test_config.yaml"]


def test_snakemake():
    try:
        # Run snakemake in the directory with Snakefile
        result = subprocess.run(COMMAND, check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Snakemake pipeline failed: {e}")
