import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from pipelines.haidian.download_modelscope_assets import (
    _extract_deployment_archive,
)
from pipelines.harbin.download_modelscope_assets import verify_checksums


def test_haidian_deployment_archive_extracts_regular_files(tmp_path: Path):
    archive = tmp_path / "asset.tar"
    with tarfile.open(archive, "w") as tar:
        content = b"result"
        member = tarfile.TarInfo("data/haidian/tasks/example/result.png")
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))

    target = tmp_path / "target"
    _extract_deployment_archive(archive, target, force=False)

    assert (target / "data/haidian/tasks/example/result.png").read_bytes() == b"result"


def test_haidian_deployment_archive_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="Unsafe path"):
        _extract_deployment_archive(archive, tmp_path / "target", force=False)


def test_harbin_checksum_reader_accepts_historical_single_space(tmp_path: Path):
    asset = tmp_path / "asset.tar"
    asset.write_bytes(b"archive")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    (tmp_path / "checksums.sha256").write_text(
        f"{digest} asset.tar\n", encoding="utf-8"
    )

    assert verify_checksums(tmp_path) is True
