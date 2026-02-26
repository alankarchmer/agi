from __future__ import annotations

from utils.runtime_hygiene import (
    apply_runtime_warning_filters,
    detect_legacy_runtime_packages,
    import_gymnasium_clean,
    runtime_health_message,
)


def test_runtime_hygiene_helpers_execute() -> None:
    apply_runtime_warning_filters()
    gym, spaces = import_gymnasium_clean()
    assert gym is not None
    assert spaces is not None

    found = detect_legacy_runtime_packages()
    assert isinstance(found, dict)

    msg = runtime_health_message()
    assert isinstance(msg, str)
    assert len(msg) > 0
