"""Read-only DGX checks for the resource readiness notebook."""
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from experiments.mom_bpe import dgx_resources as gpu
from experiments.mom_bpe.dgx_handoff import file_digest
from mltok.lib.artifact_paths import unlinked_path
from mltok.lib.atomic import atomic_writer


# Raw identity.json and result.json bytes at private revision 60e3d255c7afd86ec568e5a728f98a8c6e433219.
PINS = {
    ("M48", "fixed_source_bytes"): ("b7080fbe2b272a8d7aa203e39408258af945debc3949405a462d5dffb8c83ffe", "9479c0d94196fbeec6e515440eff6303a285723219414eea4fdd74e111428734"),
    ("M48", "fixed_training_tokens"): ("237c4cd5a91e505cfb63b413268e87533a71c692159a22c19d57eb5612775646", "0526e1d225bcf0ceb6eb65eba29b035ea20d2c31a898c8b4c619805e0e1e3858"),
    ("F-space", "fixed_source_bytes"): ("b46a501f12ded18959714d734cebb433ce5c1b533b1a5e3269391bb631f12e10", "a5b02b5d6a0f0f84aaa0bb87ffa8acda2b0368adaeec430ca3d8607b4e6b91ec"),
    ("F-space", "fixed_training_tokens"): ("936dca3c82ac08bdbf62b1790eeef769a9b2f3cd1afbe118d3ebf526f0694841", "d0b0b3cab5322938e08582112461662e11b6f8e3b483ed4132ef29c37ee8d59c"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def seed17(root, action):
    from experiments.mom_bpe import document_proxy as driver

    results = {}
    for (arm, mode), pins in PINS.items():
        folder = unlinked_path(root / "runs/dp/seed17" / arm / mode)
        for name, expected in zip(("identity.json", "result.json"), pins):
            require(file_digest(unlinked_path(folder / name))[1] == expected, f"{folder}/{name} differs")
        result, _ = driver.receipt(folder / "result.json")
        identity = json.loads((folder / "identity.json").read_bytes())
        require(identity == result["identity"], "Conflicting seed-17 identity")
        ref = result["checkpoint"]
        entry = {"bpb": result["evaluation"]["bpb"], "checkpoint": ref}
        results[f"{arm}/{mode}"] = entry
        if action == "identity":
            continue
        final = identity["forecast"]["optimizer_updates"]
        interval = identity["config"]["checkpoint_updates"]
        expected_steps = set(range(interval, final, interval)) | {final}
        found = set()
        files = {}
        for path in sorted(folder.glob("step-*.pt")):
            match = driver.GENERATION.fullmatch(path.name)
            require(match and unlinked_path(path).is_file(), f"Invalid generation {path}")
            step = int(match[1])
            size = path.stat().st_size
            if action == "hashes":
                size, digest = file_digest(path)
                require(digest == match[2], f"Generation checksum differs at {path}")
            files[path.name] = size
            found.add(step)
        entry.update(files=files, missing_steps=sorted(expected_steps - found), stored_bytes=sum(files.values()))
        if action == "hashes":
            require(files.get(ref["path"]) == ref["size"] and driver.GENERATION.fullmatch(ref["path"])[2] == ref["sha256"],
                    f"Missing or conflicting final checkpoint at {folder}")
    return results


def inspect_devices():
    from subprocess import check_output

    fields = ("--query-gpu=index,uuid,name,memory.free,utilization.gpu", "--query-compute-apps=gpu_uuid,pid")
    return {field: check_output(["nvidia-smi", field, "--format=csv,noheader,nounits"], timeout=10, text=True)
            for field in fields}


def preflight(attempt, uuid, shared, allocated):
    require(allocated is True, "Confirm the institutional allocation before driver preflight")
    policy = gpu.GpuAdmission(5120, 10, True) if shared else gpu.GpuAdmission()
    initial = policy.admit(uuid)
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from experiments.mom_bpe import document_proxy as driver

    hardware = driver.gpu_preflight(policy.min_free_mib, max_gpu_util_pct=policy.max_gpu_util_pct,
                                    allow_shared_gpu=policy.allow_shared_gpu)
    require("A100" in hardware["type"] and hardware["physical_index"] == initial["physical_index"], "Expected allocated A100")
    code = {"driver": driver.code_digest(driver.__file__), "resources": driver.code_digest(gpu.__file__)}
    with gpu.ResourceTelemetry(attempt, policy, initial, driver.torch.cuda, driver.torch.device("cuda"), code) as journal:
        journal.restored(0, None)
    return {"hardware": hardware, "policy": asdict(policy), "diagnostic_attempt": journal.path.relative_to(attempt).as_posix()}


def store(path, value):
    with atomic_writer(path, overwrite=False) as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode())


def latest(session, action):
    dirs = sorted(session.glob(action + "-*"))
    require(dirs, f"Run the {action} cell")
    path = unlinked_path(dirs[-1] / "verified.json")
    require(path.is_file(), f"Latest {action} attempt is incomplete. Inspect {path.parent}")
    return path.parent, json.loads(path.read_bytes())


def telemetry(session):
    from experiments.mom_bpe import document_proxy as driver

    folder, record = latest(session, "admission")
    path = unlinked_path(folder / record["diagnostic_attempt"])
    start, resume, end = (json.loads(unlinked_path(path / name).read_bytes())
                          for name in ("start.json", "resume.json", "end.json"))
    require(start["schema"] == "mom-dgx-resource-telemetry-v1" and start["admission"] == record["policy"]
            and start["operational_code_sha256"] == {"driver": driver.code_digest(driver.__file__),
                                                     "resources": driver.code_digest(gpu.__file__)}
            and resume == {"starting_update": 0, "checkpoint": None} and end["status"] == "training_finished",
            "Telemetry schema, admission or diagnostic completion differs")
    for sample in (start["initial"], end["gpu"]):
        require(sample and sample["uuid"] == record["hardware"]["uuid"]
                and sample["physical_index"] == record["hardware"]["physical_index"] != 7
                and sample["free_mib"] >= 0 and 0 <= sample["utilization_pct"] <= 100, "Missing GPU sample")
    require(all(type(end[k]) is int and end[k] >= 0 for k in ("peak_allocated_bytes", "peak_reserved_bytes"))
            and end["training_seconds"] >= 0, "Missing diagnostic counters")
    return {"admission_receipt": folder.name, "initial": start["initial"], "end": end}


def run(action, root, session, uuid="", shared=False, allocated=False):
    attempt = unlinked_path(session / f"{action}-{time.time_ns()}-{uuid4().hex}")
    attempt.mkdir(parents=True, mode=0o700)
    if action in ("identity", "inventory", "hashes"):
        result = seed17(root, action)
    elif action == "devices":
        result = inspect_devices()
    elif action == "admission":
        result = preflight(attempt, uuid, shared, allocated)
    elif action == "telemetry":
        result = telemetry(session)
    elif action == "report":
        evidence = {key: latest(session, key)[0].name for key in
                    ("identity", "inventory", "hashes", "devices", "admission", "telemetry")}
        check = telemetry(session)
        for dependency, consumer in (("identity", "inventory"), ("inventory", "hashes"), ("devices", "admission")):
            require(int(evidence[dependency].split("-")[1]) <= int(evidence[consumer].split("-")[1]),
                    f"Rerun {consumer} after the latest {dependency} check")
        require(check["admission_receipt"] == latest(session, "telemetry")[1]["admission_receipt"],
                "Rerun telemetry for the latest admission")
        result = {"verified_receipts": evidence, "scope": "readiness checks only", "admission": latest(session, "admission")[1],
                  "free_storage_bytes": shutil.disk_usage(root).free}
    elif action == "storage":
        result = {"seed17": seed17(root, "inventory"), "session_receipts": sorted(p.name for p in session.iterdir()),
                  "free_storage_bytes": shutil.disk_usage(root).free}
    else:
        raise ValueError("Unknown readiness check")
    store(attempt / "verified.json", result)
    print(json.dumps(result, indent=2), flush=True)
