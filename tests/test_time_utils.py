"""Unit tests for time/period normalization utilities."""

import pytest

from app.services.time_utils import normalize_month, normalize_period, normalize_quarter_date


class TestNormalizeMonth:
    def test_compact_to_hyphen(self):
        assert "2026-01" in normalize_month("202601")
        assert "202601" in normalize_month("202601")

    def test_hyphen_to_compact(self):
        assert "202504" in normalize_month("2025-04")
        assert "2025-04" in normalize_month("2025-04")

    def test_ymd_collapses_to_months(self):
        variants = normalize_month("20251201")
        assert "20251201" in variants
        assert "202512" in variants
        assert "2025-12" in variants

    def test_arbitrary_value_preserved(self):
        assert normalize_month("Q1") == ["Q1"]

    def test_none_returns_empty(self):
        assert normalize_month(None) == []
        assert normalize_month("") == []


class TestNormalizePeriod:
    def test_compact_vs_hyphen(self):
        variants = normalize_period("2025-04_vs_2025-06")
        assert "2025-04_vs_2025-06" in variants
        assert "202504_vs_202506" in variants
        assert "202504_vs_2025-06" in variants
        assert "2025-04_vs_202506" in variants

    def test_single_period_treated_as_month(self):
        assert "202601" in normalize_period("202601")
        assert "2026-01" in normalize_period("202601")

    def test_none_returns_empty(self):
        assert normalize_period(None) == []
        assert normalize_period("") == []


class TestNormalizeQuarterDate:
    def test_hyphen_maps_to_quarter(self):
        variants = normalize_quarter_date("2025-04")
        assert "2025Q2" in variants
        assert "2025-04" in variants
        assert "202504" in variants

    def test_compact_maps_to_quarter(self):
        variants = normalize_quarter_date("202510")
        assert "2025Q4" in variants
        assert "202510" in variants

    def test_ymd_preserved(self):
        assert "20251201" in normalize_quarter_date("20251201")

    def test_quarter_preserved(self):
        assert "2025Q2" in normalize_quarter_date("2025Q2")
