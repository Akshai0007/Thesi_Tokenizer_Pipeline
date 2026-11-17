# mltok — Malayalam tokenizer pipeline (plug-in architecture)

Train, compare and select a **from-scratch Malayalam tokenizer** on Kaggle (cheap selection) and then the DGX (final run).
Output: a plain Hugging Face `tokenizer.json` (byte-fallback BPE, loads with `transformers` — no custom code).

**Everything swappable is one file.** Dataset, filters, normalizer, pre-tokenizer, training algorithm, metrics, proxy model,
selection rule and checkpoint store are *plugins* that depend only on a small fixed contract, never on each other. Replace one file (or one config entry)
and the rest of the pipeline is untouched. **Read [`PIPELINE.md`](PIPELINE.md)** for the full file map, the contracts, the swap recipes
and exactly what each change recomputes.

```
sources ─► filters ─► dedup/split ─► normalizer ─► pretokenizer ─► algorithm ─► metrics ─► proxy LM (bits/byte) ─► selector ─► best/
```

## Swap things in 3 commands
```bash
python -m mltok plugins                                      # what is registered, and from which file
python -m mltok new-plugin source my_crawl --dir my_plugins  # writes a WORKING starter file (all 8 kinds supported)
python -m mltok check --file my_plugins/my_crawl.py          # proves it fits the contract BEFORE any run
# then in the config:  plugins_dir: [my_plugins]   and   sources: [{type: my_crawl, id: crawl}]
```
Datasets need no code at all if they are files: `sources: [{type: local_files, id: news, path: /data/*.jsonl.gz}]`.
A file that registers an existing name in your `plugins_dir` **replaces** the built-in one.

## Quick start

### GitHub or Kaggle Dataset?
**Use GitHub.** The notebook needs *Internet = On* anyway (corpus + baseline tokenizers), so `git clone` is free, a fix is `git push` + `git pull`
(no new Dataset version), and the DGX pulls the same repo so both machines run identical code. Private repo: Kaggle Secret `GITHUB_TOKEN`.
Use the zip only for an offline/frozen snapshot; for runs you want to cite, pin the commit (`COMMIT` in the notebook).

### Kaggle (Accelerator = GPU T4 x2, Internet = On) — `notebooks/kaggle_runner.ipynb`
The notebook works in **gated rounds**. Set `ROUND` in the first cell and *Run All*: cells for rounds <= `ROUND` run, later ones are skipped. Each gate prints PASS/FAIL; do not go on until it passes.

| ROUND | runs | gate | time |
|---|---|---|---|
| 1 | get code (GitHub), install, `selftest`, `pytest` | 2 GPUs visible, selftest + tests green | ~5-8 min |
| 2 | dry run on a small real sample (`kaggle_dryrun.yaml`) | datasets download, baselines load, both GPUs used, report written | ~10-20 min (estimate) |
| 3 | full selection (`kaggle_small.yaml`); `FULL_STAGES`: `intrinsic` (CPU) then `proxy` (GPU) | report reviewed | hours |
| 4 | variations from `RUN_VARIATIONS`: `seeds`, `vocab`, `zwnj`, `algos`, `bigproxy`, `datamix` (config-only `--set`; snapshots kept) | compare snapshots; ties = gap within seed spread | per variation |
| 5 | freeze checklist -> `freeze.json` (commit, winner, plugin hashes, data fingerprints) + `results.zip` | lossless + verify + seeds + beats baselines by more than noise | minutes |

New session on a fresh machine? Keep `ROUND` and set `DONE_ROUNDS = [1, 2]` for gates that already passed: their runs are skipped (get-code + install always re-run) and attached previous output is restored.
`proxy.seeds: [0,1,2]` reports mean ± std BPB per variant and flags a top-2 gap inside the seed spread as a **tie**.

### Checkpoints to Hugging Face (so nothing is lost if a session dies)
Every finished unit (word table, tokenizer, metrics, each proxy seed, report, `best/`) is pushed to a **new private** Hugging Face model repo, and the long proxy-LM training saves mid-run checkpoints that resume after a crash.
1. Create a Kaggle Secret (Add-ons → Secrets) named `HF_TOKEN` with **write** access and attach it to the notebook.
2. In the notebook settings set `HF_REPO = "<you>/<new-repo-name>"` — use a *new* name, not an existing model repo (the store only writes under its own `mltok/` folder, but a dedicated repo is cleaner).
3. Gate 1 runs `store-check` (push → pull → delete a tiny file) so a bad token fails in seconds, not after hours.
A fresh session restores automatically (`restore`) and the cache reuses everything still valid. Details and limits: PIPELINE.md §8.

### What uses CUDA
* **Proxy language models: both GPUs.** One worker process per GPU (`CUDA_VISIBLE_DEVICES` pinned) takes a share of the tokenizer variants (`proxy.gpus: auto`). fp16+GradScaler on T4/P100, bf16 on A100 (chosen from compute capability; `is_bf16_supported` is true on T4 via slow emulation).
* **Tokenizer training, word counting, encoding, metrics: CPU.** BPE training is sequential and hash-table bound; there is no practical GPU speed-up. They use all CPU cores (`resources.workers`).
* `proxy.micro_batch` accumulates gradients so large vocabularies fit a 16 GB T4.

### DGX
Edit `configs/dgx_full.yaml` (`components`, `search.factors` = Kaggle winners), then `scripts/run_dgx.sh [GPU_INDEX]`.
It waits until ≥ 60 GB of that A100 is free and utilisation is low, and caps its own CUDA allocator at 60 GB. `proxy.gpus` is an explicit list there — `auto` would use every GPU in the box.

## How to read results
`runs/<name>/report.md` ranks every variant and ends with a provenance block (plugin file hashes, data fingerprints).
**Do not choose by fertility alone**: published analyses find token-count/entropy metrics do not reliably predict model quality.
The deciding number is **proxy-LM bits-per-byte (BPB)** — the same small GPT on the same text, scored on held-out text; BPB is comparable across tokenizers.
The winner is picked among **eligible** variants only (lossless on eval docs *and* on a stress set of emoji / other scripts / ZWNJ / odd whitespace) — a tokenizer without byte fallback can score a good BPB and still be unusable.
Confounds: fixed context holds more text for low-fertility tokenizers; bigger vocabularies add embedding parameters (`n_nonembed` is logged). Confirm the Kaggle winner on the DGX proxy before the final run.

## Real runs
See `RESULTS_LOG.md`. Run 1: SentencePiece Unigram beat our constrained BPE and stock BPE (script-integrity diagnostics did not predict that); it exposed a data-mix bug. Run 2 (balanced data): Unigram still wins by a clear margin over four BPE implementations; the constraint is neutral on BPB; vocabulary 32K → 48K → 64K improves BPB with diminishing returns.

## Verified vs not verified (honest status)
Verified in the dev sandbox (CPU, synthetic data): **103 tests pass** (2 GPU tests skip without CUDA): Python and HF normalizers/pre-tokenizers agree;
exported tokenizer equals a slow reference BPE encoder; lossless round-trip incl. byte fallback and old-style chillu; loads in `transformers.AutoTokenizer` with chat template;
every built-in plugin passes its contract check and the independence rule; every starter template is a valid plugin; a brand-new dataset works as one file; replacing a plugin
invalidates exactly its dependents (and nothing else); resumable runner; multi-process counting equals single-process; per-GPU worker fan-out (with CPU workers).

**Not verified — check on the first Kaggle run:** the Hugging Face dataset ids/config names in `configs/*.yaml` (the sandbox had no Hub access; failed sources are skipped and logged);
Hub baseline tokenizers (some are gated); the GPU paths on real CUDA devices (fp16/GradScaler on T4, bf16 on A100) and 64k-vocab memory on the DGX;
all numbers from *synthetic* data prove mechanics only, not which design is best for real Malayalam; MinHash near-duplicate removal is not included (add a `filter`-stage plugin or
pre-dedup before the final run if sources overlap).

## Planned experiments (not implemented)
Conjunct-level seeds (remaining akshara breaks are inside conjuncts) · late-stage multiword phase (published Malayalam numbers suggest small gains) ·
morpheme boundaries for `mlbpe_morph` (file format `word<TAB>offset,offset`; deriving surface boundaries from `mlmorph` needs an alignment step) ·
bilingual Malayalam+English retrain · extension-pack export for adapting an existing LLM.
