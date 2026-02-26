from __future__ import annotations

import sys
import types


def _install_pkg_resources_shim() -> None:
    try:
        import pkg_resources  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import importlib.metadata as importlib_metadata

    shim = types.ModuleType("pkg_resources")

    class _EntryPointAdapter:
        def __init__(self, ep):
            self._ep = ep

        def resolve(self):
            return self._ep.load()

    def iter_entry_points(group: str, name: str | None = None):
        eps = importlib_metadata.entry_points()
        selected = eps.select(group=group) if hasattr(eps, "select") else [ep for ep in eps if ep.group == group]
        if name is not None:
            selected = [ep for ep in selected if ep.name == name]
        return [_EntryPointAdapter(ep) for ep in selected]

    shim.iter_entry_points = iter_entry_points  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = shim


def main() -> None:
    _install_pkg_resources_shim()
    from tensorboard.main import run_main

    run_main()


if __name__ == "__main__":
    main()
