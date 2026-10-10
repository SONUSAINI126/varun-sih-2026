import numpy as np
from varun.efi import efi
from varun.pipeline import run


def test_efi_range():
    e = efi(np.random.rand(20, 4, 4) + 1, np.random.rand(30, 4, 4))
    assert e.min() >= -1 and e.max() <= 1


def test_pipeline_runs():
    assert run(np.random.rand(10, 10) * 50)["members_kept"] >= 0
