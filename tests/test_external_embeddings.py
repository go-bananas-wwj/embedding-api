import numpy as np

from app.services import external_embeddings


def test_aef_month_falls_back_to_2025_annual_embedding(tmp_path, monkeypatch):
    root = tmp_path / "aef"
    annual = root / "haidian" / "2025"
    annual.mkdir(parents=True)
    expected = np.full((64, 4, 4), 2.5, dtype=np.float32)
    np.save(annual / "patch_000000.npy", expected)
    monkeypatch.setattr(external_embeddings, "AEF_EMBEDDING_DIR", root)

    actual = external_embeddings.load_aef_embedding(
        "haidian", "patch_000000", "202604"
    )

    np.testing.assert_array_equal(actual, expected)


def test_aef_exact_year_takes_precedence_over_2025_fallback(tmp_path, monkeypatch):
    root = tmp_path / "aef"
    for year, value in (("2025", 2.5), ("2026", 2.6)):
        annual = root / "haidian" / year
        annual.mkdir(parents=True)
        np.save(
            annual / "patch_000000.npy",
            np.full((64, 4, 4), value, dtype=np.float32),
        )
    monkeypatch.setattr(external_embeddings, "AEF_EMBEDDING_DIR", root)

    actual = external_embeddings.load_aef_embedding(
        "haidian", "patch_000000", "202604"
    )

    assert float(actual[0, 0, 0]) == np.float32(2.6)
