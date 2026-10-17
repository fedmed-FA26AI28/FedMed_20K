import re
from typing import Optional


def sanitize_run_name(name: Optional[str]) -> Optional[str]:
    """Sanitize a run name to ensure safe directory naming across Windows and Linux.

    Replaces illegal filesystem characters (:, *, ?, ", <, >, |, /, \\) and spaces
    with underscores, stripping leading/trailing dots/spaces.

    Args:
        name: User-provided run name string or None.

    Returns:
        Sanitized safe string for folder naming, or None if name is empty/None.
    """
    if name is None:
        return None
    name_str = str(name).strip()
    if not name_str:
        return None

    # Replace illegal filesystem chars and whitespace sequences with an underscore
    sanitized = re.sub(r'[\\/:*?"<>|\s]+', '_', name_str)
    # Strip leading/trailing dots, hyphens, and underscores
    sanitized = sanitized.strip('._-')
    return sanitized if sanitized else "run"
