from pathlib import Path


def resolve_path(value: str | Path, base: Path) -> Path:
    """Resolve ``value`` against ``base`` unless it is already absolute."""
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()
