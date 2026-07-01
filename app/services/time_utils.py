"""Time/period format normalization for cross-region robustness.

Harbin data is organized by ``YYYY-MM`` (e.g. ``2025-04``), while Haidian
uses ``YYYYMM`` (e.g. ``202601``). The API should accept either form and
fall back to the other, so that frontends do not need region-specific
date handling logic.
"""

import re
from typing import List, Optional


# 6-digit month: YYYYMM
_YYYYMM_RE = re.compile(r"^(\d{4})(\d{2})$")
# Hyphenated month: YYYY-MM
_YYYY_HYPHEN_MM_RE = re.compile(r"^(\d{4})-(\d{2})$")
# 8-digit date: YYYYMMDD (used by some Haidian raw scenes / embeddings)
_YYYYMMDD_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def normalize_month(month: Optional[str]) -> List[str]:
    """Return a list of equivalent month/date strings to try.

    The first entry is always the original value so that exact matches are
    preferred. If the value can be converted between ``YYYY-MM`` and
    ``YYYYMM`` forms, both are returned. ``YYYYMMDD`` is also collapsed to
    ``YYYYMM`` and ``YYYY-MM``.

    Examples:
        >>> normalize_month("2025-04")
        ["2025-04", "202504"]
        >>> normalize_month("202601")
        ["202601", "2026-01"]
        >>> normalize_month("20251201")
        ["20251201", "202512", "2025-12"]
    """
    if not month:
        return []

    result = [month]

    m_ymd = _YYYYMMDD_RE.match(month)
    if m_ymd:
        y, m, _ = m_ymd.groups()
        ym = f"{y}{m}"
        hyphen = f"{y}-{m}"
        if ym not in result:
            result.append(ym)
        if hyphen not in result:
            result.append(hyphen)
        return result

    m_hyphen = _YYYY_HYPHEN_MM_RE.match(month)
    if m_hyphen:
        y, m = m_hyphen.groups()
        compact = f"{y}{m}"
        if compact not in result:
            result.append(compact)
        return result

    m_compact = _YYYYMM_RE.match(month)
    if m_compact:
        y, m = m_compact.groups()
        hyphen = f"{y}-{m}"
        if hyphen not in result:
            result.append(hyphen)
        return result

    return result


def normalize_period(period: Optional[str]) -> List[str]:
    """Return equivalent period strings (e.g. for change detection).

    Supports ``<month>_vs_<month>`` and falls back to ``normalize_month``
    for single-valued periods.

    Examples:
        >>> normalize_period("2025-04_vs_2025-06")
        ["2025-04_vs_2025-06", "202504_vs_2025-06", "2025-04_vs_202506", "202504_vs_202506"]
        >>> normalize_period("202512_vs_202601")
        ["202512_vs_202601", "2025-12_vs_202601", ...]
    """
    if not period:
        return []

    if "_vs_" not in period:
        return normalize_month(period)

    left, right = period.split("_vs_", 1)
    left_variants = normalize_month(left)
    right_variants = normalize_month(right)

    result = []
    for lv in left_variants:
        for rv in right_variants:
            candidate = f"{lv}_vs_{rv}"
            if candidate not in result:
                result.append(candidate)
    return result


def normalize_quarter_date(date: str) -> List[str]:
    """Return candidate file stems for mosaic raw TIFF lookup.

    Accepts ``YYYY-MM``, ``YYYYMM`` and exact ``YYYYMMDD`` strings and
    returns all equivalent quarterly/compact forms that may match the
    underlying raw scene filenames.
    """
    result = [date]

    m_hyphen = _YYYY_HYPHEN_MM_RE.match(date)
    if m_hyphen:
        year, month = m_hyphen.groups()
        quarter = (int(month) - 1) // 3 + 1
        result.append(f"{year}Q{quarter}")
        compact = f"{year}{month}"
        if compact not in result:
            result.append(compact)
        return result

    m_compact = _YYYYMM_RE.match(date)
    if m_compact:
        year, month = m_compact.groups()
        quarter = (int(month) - 1) // 3 + 1
        result.append(f"{year}Q{quarter}")
        return result

    return result
