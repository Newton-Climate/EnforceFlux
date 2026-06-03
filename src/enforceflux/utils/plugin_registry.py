from __future__ import annotations

from importlib.metadata import entry_points
from typing import Type


def _select_entry_points(group: str):
    eps = entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return eps.get(group, [])


def list_plugins(group: str, base_cls: Type | None = None) -> dict[str, Type]:
    plugins: dict[str, Type] = {}
    for ep in _select_entry_points(group):
        try:
            cls = ep.load()
        except Exception as exc:  # pragma: no cover - import error surface
            raise RuntimeError(
                f"Failed to load plugin '{ep.name}' from group '{group}': {exc}"
            ) from exc
        if base_cls is not None and not issubclass(cls, base_cls):
            raise TypeError(
                f"Plugin '{ep.name}' in group '{group}' does not subclass {base_cls.__name__}"
            )
        plugins[ep.name] = cls
    return plugins


def get_plugin(group: str, name: str, base_cls: Type | None = None) -> Type:
    plugins = list_plugins(group, base_cls=base_cls)
    if name not in plugins:
        available = ", ".join(sorted(plugins.keys()))
        raise ValueError(
            f"Unknown plugin '{name}' in group '{group}'. Available: {available}"
        )
    return plugins[name]


def normalize_plugin_name(group: str, plugin_ref: str) -> str:
    prefix = f"{group}."
    if plugin_ref.startswith(prefix):
        return plugin_ref[len(prefix) :]
    return plugin_ref
