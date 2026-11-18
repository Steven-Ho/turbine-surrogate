from pathlib import Path, PurePosixPath, PureWindowsPath


def is_cross_platform_abs_path(path_str: str) -> bool:
    raw = str(path_str).strip()
    return (
        Path(raw).is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or PurePosixPath(raw).is_absolute()
    )


def normalize_relative_path(path_str: str) -> str:
    return str(path_str).strip().replace('\\', '/')


def resolve_path_from_base(base_dir: str, path_str: str) -> str:
    raw = str(path_str).strip()
    if is_cross_platform_abs_path(raw):
        return raw
    return str(Path(base_dir) / Path(normalize_relative_path(raw)))


def resolve_path_from_anchor(anchor_path: str, path_str: str) -> str:
    base_dir = str(Path(anchor_path).parent)
    return resolve_path_from_base(base_dir, path_str)


def normalize_path_for_match(path_str: str) -> str:
    return str(PurePosixPath(str(path_str).strip().replace('\\', '/')))


def path_endswith(candidate_path: str, mapped_path: str) -> bool:
    return normalize_path_for_match(candidate_path).endswith(normalize_path_for_match(mapped_path))
