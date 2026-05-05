import os
import re
from typing import Dict, Tuple, List


_FILE_MARKER_RE = re.compile(r"^\s*(#|//|/\*|<!--)\s*FILE:\s*(.+?)\s*(\*/|-->)?\s*$")
_END_MARKER_RE = re.compile(r"^\s*(#|//|/\*|<!--)\s*END\s*FILE\s*(\*/|-->)?\s*$", re.IGNORECASE)


def _extract_file_marker(line: str) -> str:
    match = _FILE_MARKER_RE.match(line)
    if not match:
        return ""
    path = match.group(2).strip().strip("\"'")
    return path


def _is_end_marker(line: str) -> bool:
    return bool(_END_MARKER_RE.match(line))


def parse_code_bundle(
    raw_text: str,
    target_folder: str,
    default_name: str,
) -> Tuple[Dict[str, str], bool]:
    """Parse multi-file bundle output into a file map.

    Returns (file_map, markers_found).
    """
    file_map: Dict[str, str] = {}
    current_path = ""
    buffer: List[str] = []
    markers_found = False

    for line in raw_text.splitlines():
        if _is_end_marker(line):
            continue
        marker_path = _extract_file_marker(line)
        if marker_path:
            markers_found = True
            if current_path:
                file_map[current_path] = "\n".join(buffer).strip("\n")
            current_path = marker_path.replace("\\", os.sep)
            buffer = []
            continue
        buffer.append(line)

    if current_path:
        file_map[current_path] = "\n".join(buffer).strip("\n")

    if not markers_found:
        file_map[os.path.join(target_folder, default_name)] = raw_text.strip()

    return file_map, markers_found


def _safe_join(base_path: str, rel_path: str) -> str:
    if os.path.isabs(rel_path):
        full_path = os.path.normpath(rel_path)
    else:
        full_path = os.path.normpath(os.path.join(base_path, rel_path))
    base_norm = os.path.normpath(base_path)
    if not full_path.startswith(base_norm + os.sep) and full_path != base_norm:
        raise ValueError(f"Unsafe path outside repo: {rel_path}")
    return full_path


def execute_plan_bundle(file_map: Dict[str, str], repo_path: str) -> List[str]:
    """Write a bundle of files to disk relative to repo_path."""
    written_paths: List[str] = []
    for rel_path, code in file_map.items():
        file_path = _safe_join(repo_path, rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        written_paths.append(file_path)
    return written_paths


def execute_plan(plan_code: str, target_folder: str, file_name: str = "new_feature.py"):
    os.makedirs(target_folder, exist_ok=True)
    file_path = os.path.join(target_folder, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(plan_code)
    return file_path