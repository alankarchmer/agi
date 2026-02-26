from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime_hygiene import detect_legacy_runtime_packages, runtime_health_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Check runtime hygiene for modern ASL usage")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if legacy packages are detected.",
    )
    args = parser.parse_args()

    print(runtime_health_message())

    found = detect_legacy_runtime_packages()
    if args.strict and found:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
