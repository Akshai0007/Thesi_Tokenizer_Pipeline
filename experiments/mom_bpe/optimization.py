"""Fit and measure one matched boundary-optimization arm."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import yaml

from experiments.mom_bpe.diagnose import AUDIT, HERE, format_json, immutable, sha
from experiments.mom_bpe.orthographic_errors import audit
from experiments.mom_bpe.performance import checked_canonical, from_files, measure
from experiments.mom_bpe.pilot import peak_rss_bytes
from experiments.mom_bpe.verify_frozen import checked_member, load_rows, verify
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.picky_backend import ARTIFACT
from mltok.snapshots import _lock, file_identity

ARMS = ("T1", "T3", "T4", "T5", "T8")
EXPLORATORY = "EXPLORATORY_WIKIMEDIA_PROSE"
CAP = 2 * 1024**2


def verified_input(path, normalizer, frozen=False):
    path = path.resolve()
    content = path.read_bytes()
    manifest = json.loads(content)
    digest = sha(content)
    if frozen:
        gate = verify(path, require_morphology=False)
        if gate["manifest_sha256"] != digest:
            raise ValueError("Frozen manifest changed during verification")
        files = {key: [checked_member(path.parent, entry) for entry in manifest["splits"][source]]
                 for key, source in (("train", "train"), ("validation", "eval"))}
        files["orthography"] = [checked_member(path.parent, manifest["orthography_panel"])]
    else:
        if (manifest["schema"] != "mom-boundary-development-v1"
                or manifest["qualification"] != EXPLORATORY or manifest["verified_freeze"]):
            raise ValueError("Expected explicitly exploratory development input")
        if manifest["recipe"]["normalizer_fingerprint"] != registry.fingerprint("normalizer", normalizer):
            raise ValueError("Development normalizer fingerprint differs")
        files = {}
        for split in ("train", "validation"):
            entry = manifest["splits"][split]
            if entry["size"] > 2 * CAP or not 0 < entry["canonical_bytes"] <= CAP:
                raise ValueError("Exploratory input exceeds the local CPU cap")
            files[split] = [checked_member(path.parent, {**entry, "path": entry["file"]})]
        prior_families, prior_groups, prior_digests, prior_ids = set(), set(), set(), set()
        for split in ("train", "validation"):
            families, groups, digests, ids = set(), set(), set(), set()
            count, nbytes = 0, 0
            for row in load_rows(files[split][0]):
                count += 1
                nbytes += row["canonical_bytes"]
                if row["id"] in ids or row["dedup_group"] in groups or row["canonical_sha256"] in digests:
                    raise ValueError("Development contains duplicate records")
                families.add(row["family"])
                groups.add(row["dedup_group"])
                digests.add(row["canonical_sha256"])
                ids.add(row["id"])
            if (count != manifest["splits"][split]["rows"] or nbytes != manifest["splits"][split]["canonical_bytes"]
                    or families & prior_families or groups & prior_groups or digests & prior_digests or ids & prior_ids):
                raise ValueError("Development split exposure or separation differs")
            prior_families, prior_groups, prior_digests, prior_ids = families, groups, digests, ids
    return files, digest


def fit_arm(manifest, out, arm, vocab=7744, *, frozen=False, fit_only=False):
    registry.load_plugins()
    cfg_bytes = (HERE / "optimization-arms.yaml").read_bytes()
    cfg = yaml.safe_load(cfg_bytes)
    if frozen and vocab != cfg["primary_scientific_vocab"]:
        raise ValueError("Use 48K for the primary frozen evaluation")
    files, manifest_sha = verified_input(manifest, cfg["normalizer"], frozen)
    normalizer = registry.make("normalizer", cfg["normalizer"])
    pre_spec = cfg["pretokenizer"]
    spec = cfg["arms"][arm]
    algo = registry.make("algorithm", spec)
    canonical_train = sum(len(checked_canonical(r, normalizer).encode()) for r in from_files(files["train"]))
    identity = {"data_sha256": manifest_sha, "mode": "frozen" if frozen else "development", "arm": arm, "vocab": vocab,
                "config_sha256": sha(cfg_bytes), "normalizer": registry.fingerprint("normalizer", cfg["normalizer"]),
                "pretokenizer": registry.fingerprint("pretokenizer", pre_spec),
                "algorithm": registry.fingerprint("algorithm", spec),
                "code": {p: sha((HERE / p).read_bytes().replace(b"\r\n", b"\n"))
                         for p in ("optimization.py", "orthographic_errors.py", "performance.py")}}
    fit_path, model_path = out / "fit.json", out / ARTIFACT
    with _lock(out / ".arm.lock"):
        immutable(out / "identity.json", format_json(identity))
        if fit_path.exists():
            fit = json.loads(fit_path.read_bytes())
            if fit["identity"] != identity or fit["model_sha256"] != file_identity(model_path)[1]:
                raise ValueError("Completed fit differs from its identity")
            model = algo.load(out, normalizer)
        else:
            start = perf_counter()
            model, info = algo.train(TrainContext(
                vocab, DEFAULT_SPECIALS, normalizer, registry.make("pretokenizer", pre_spec), out, None,
                list(map(str, files["train"])), {"merge_checkpoint_every": 512},
            ))
            if info["canonical_bytes"] != canonical_train:
                raise ValueError("Fitted arm consumed different training bytes")
            fit = {"training": info, "seconds": perf_counter() - start, "peak_rss_bytes": peak_rss_bytes(),
                   "model_sha256": file_identity(model_path)[1], "identity": identity}
            immutable(fit_path, format_json(fit))
        if fit_only:
            return fit
        if fit["training"]["occupied_vocab"] != vocab:
            raise ValueError("Underfilled arm. Matched evaluation was withheld.")
        result_path = out / "validation.json"
        if result_path.exists():
            result = json.loads(result_path.read_bytes())
            digest = result.pop("result_sha256")
            if result["fit"] != fit or sha(canonical(result)) != digest:
                raise ValueError("Completed measurement failed verification")
            return {**result, "result_sha256": digest}
        rows = list(from_files(files["validation"]))
        strata = {}
        for key in ("src", "domain"):
            groups = defaultdict(list)
            for row in rows:
                groups[row[key]].append(row)
            strata[key] = {k: measure(model, group, normalizer, vocab) for k, group in sorted(groups.items())}
        if frozen:
            orthography = [r for r in from_files(files["orthography"]) if r["status"] == "adjudicated"]
        else:
            orthography = json.loads((AUDIT / "fidelity-cases.json").read_bytes())["cases"]
        result = {
            "fit": fit, "qualification": "FROZEN" if frozen else EXPLORATORY,
            "validation": measure(model, rows, normalizer, vocab), "by_stratum": strata,
            "boundary_attribution": audit(model, rows),
            "fidelity_probes": measure(model, orthography, normalizer, vocab, require_hashes=False),
            "special_ids": model.special_ids,
        }
        result["result_sha256"] = sha(canonical(result))
        immutable(result_path, format_json(result))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--vocab", type=int, default=7744)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    report = fit_arm(args.manifest, args.out, args.arm, args.vocab, frozen=args.frozen, fit_only=args.fit_only)
    print(json.dumps({"arm": args.arm, "occupied": report.get("fit", report)["training"]["occupied_vocab"],
                      "validation_tokens": report.get("validation", {}).get("ids")}))
