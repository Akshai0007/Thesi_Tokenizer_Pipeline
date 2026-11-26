import hashlib
import json
import stat
from zipfile import ZipFile, ZipInfo

import pytest

from experiments.mom_bpe import dgx_handoff as handoff


def zip_fixture(path, extra=None, corrupt=False):
    member, data = "source/code.py", b"committed source"
    identity = (len(data), hashlib.sha256(data).hexdigest())
    raw = json.dumps({"schema": handoff.SCHEMA, "profile": "smoke",
                      "files": [{"path": member, "bytes": identity[0], "sha256": identity[1],
                                 "identity": "git:pinned", "arm": None}]}).encode()
    with ZipFile(path, "x") as archive:
        archive.writestr(member, b"x" * len(data) if corrupt else data)
        archive.writestr(handoff.MANIFEST, raw)
        archive.writestr(handoff.SUMS, handoff.sums_bytes(raw, {member: identity}))
        if extra is not None:
            archive.writestr(extra, b"invalid")
    return handoff.file_digest(path)[1]


def test_fresh_extraction_and_independent_hash_verification(tmp_path):
    archive, target = tmp_path / "pack.zip", tmp_path / "restored"
    sha = zip_fixture(archive)
    assert handoff.extract_bundle(archive, target, sha)["profile"] == "smoke"
    assert handoff.verify_bundle(target)["files"][0]["path"] == "source/code.py"
    with pytest.raises(FileExistsError):
        handoff.extract_bundle(archive, target, sha)
    (target / "source/code.py").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Size or SHA-256"):
        handoff.verify_bundle(target)


@pytest.mark.parametrize("bad", ["../escape", "/absolute", "source\\nt", "source/aux.txt", "SOURCE/extra",
                                 "source/code.py/embedded"])
def test_rejects_unsafe_names_before_extracting(tmp_path, bad):
    archive = tmp_path / "pack.zip"
    sha = zip_fixture(archive, extra=bad)
    with pytest.raises(ValueError):
        handoff.extract_bundle(archive, tmp_path / "restored", sha)
    assert not (tmp_path / "restored").exists()


def test_rejects_zip_symlink_before_extracting(tmp_path):
    link = ZipInfo("source/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = tmp_path / "pack.zip"
    sha = zip_fixture(archive, extra=link)
    with pytest.raises(ValueError, match="Non-regular ZIP"):
        handoff.extract_bundle(archive, tmp_path / "restored", sha)
    assert not (tmp_path / "restored").exists()


def test_checks_trusted_archive_digest_and_extracted_payload(tmp_path):
    archive = tmp_path / "pack.zip"
    sha = zip_fixture(archive, corrupt=True)
    with pytest.raises(ValueError, match="Archive SHA-256"):
        handoff.extract_bundle(archive, tmp_path / "wrong", "0" * 64)
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(ValueError, match="recorded hash"):
        handoff.extract_bundle(archive, tmp_path / "restored", sha)


def test_preflight_rejects_cpu_torch_without_probing_gpu(tmp_path, monkeypatch):
    from experiments.mom_bpe import document_proxy

    bundle = tmp_path / "bundle"
    monkeypatch.setattr(handoff, "imports", lambda _: (document_proxy, None))
    monkeypatch.delenv("MLTOK_PLUGINS", raising=False)
    monkeypatch.setattr(document_proxy.torch.version, "cuda", None)
    monkeypatch.setattr(document_proxy, "gpu_preflight", lambda *_: pytest.fail("probed the GPU"))
    spec = {"storage": {"additional_free_scratch_bytes": 10}}
    with pytest.raises(ValueError, match="CPU-only"):
        handoff.preflight(bundle, tmp_path / "scratch", spec)
