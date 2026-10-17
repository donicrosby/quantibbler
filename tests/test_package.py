import quantibbler
import quantibbler.kernels as k


def test_version():
    assert quantibbler.__version__ == "0.1.0"


def test_backend_is_known():
    assert k.backend() in {"rust", "numpy-reference"}
