from __future__ import annotations

from utils.device import get_device, mps_available


def test_mps_available_returns_bool() -> None:
    assert isinstance(mps_available(), bool)


def test_get_device_cpu_when_no_accel_preference() -> None:
    dev = get_device(prefer_cuda=False, prefer_mps=False)
    assert dev.type == "cpu"
