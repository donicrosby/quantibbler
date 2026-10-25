"""Static fence: no CPU-defaulting tensor constructor in the capture/block path.

Every torch.zeros/ones/arange in these modules either names its device or is
exempt below with a reason. Device mismatches cost four fires in one day; a grep
is the check, and this test is the grep. The class must stop being representable.

The module list is the capture/block path: the calibration pass plus the
block-reconstruction stage modules.
"""

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "quantibbler" / "ptq"
MODULES = (
    "calibration.py",
    "block_recon.py",
    "_factorized.py",
    "_stats.py",
    "_probe.py",
    "_tuning.py",
    "_ste.py",
    "_block_stage.py",
)
CTOR = re.compile(r"torch\.(zeros|ones|arange|full|empty|eye|tensor)\(")
EXEMPT = (
    "nn.Parameter(torch.ones",  # module Parameter: moved by the module's own .to()
    "idx = torch.arange",  # index tensor: a CPU index on a CUDA tensor is legal
)


def test_no_cpu_defaulting_constructors() -> None:
    bad = []
    for f in MODULES:
        for i, line in enumerate((SRC / f).read_text().splitlines(), 1):
            if CTOR.search(line) and "device=" not in line and not any(p in line for p in EXEMPT):
                bad.append(f"{f}:{i}: {line.strip()[:90]}")
    assert not bad, (
        "CPU-defaulting constructors (name device= from the owning tensor/module, "
        "or add to EXEMPT with a reason):\n" + "\n".join(bad)
    )
