from __future__ import annotations

import contextlib
import importlib
import io
import sys
import warnings
from importlib import metadata


_GYM_NOTICE_FRAGMENTS = (
    "Gym has been unmaintained since 2022",
    "Please upgrade to Gymnasium, the maintained drop-in replacement of Gym",
    "https://gymnasium.farama.org/introduction/migration_guide/",
)


def apply_runtime_warning_filters() -> None:
    """Filter known third-party deprecations unrelated to this codebase."""
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated\..*",
        category=FutureWarning,
    )


def _strip_known_gym_notice(stderr_text: str) -> str:
    if not stderr_text:
        return ""

    lines = stderr_text.splitlines()
    filtered = [line for line in lines if not any(fragment in line for fragment in _GYM_NOTICE_FRAGMENTS)]
    if not filtered:
        return ""
    return "\n".join(filtered).rstrip() + "\n"


def import_gymnasium_clean() -> tuple[object, object]:
    """
    Import gymnasium while suppressing the legacy Gym banner emitted by unrelated plugin packages.

    This keeps stderr clean without muting other unexpected import-time errors/warnings.
    """
    stream = io.StringIO()
    with contextlib.redirect_stderr(stream):
        gym = importlib.import_module("gymnasium")
        spaces = importlib.import_module("gymnasium.spaces")

    remainder = _strip_known_gym_notice(stream.getvalue())
    if remainder:
        sys.stderr.write(remainder)
    return gym, spaces


def detect_legacy_runtime_packages() -> dict[str, str]:
    """Return detected legacy packages that commonly cause noisy deprecation output."""
    found: dict[str, str] = {}
    for pkg in ("gym", "shimmy", "pynvml"):
        try:
            found[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return found


def runtime_health_message() -> str:
    found = detect_legacy_runtime_packages()
    if not found:
        return "Runtime hygiene check: clean (no legacy gym/pynvml packages detected)."

    pieces = [f"{name}=={ver}" for name, ver in sorted(found.items())]
    msg = [
        "Runtime hygiene check: legacy packages detected in active Python environment:",
        f"  {', '.join(pieces)}",
        "Recommended (inside this project venv):",
        "  pip uninstall -y gym shimmy pynvml",
        "  pip install -U gymnasium nvidia-ml-py",
    ]
    return "\n".join(msg)
