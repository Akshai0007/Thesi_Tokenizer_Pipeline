# PIPELINE.md — what every file does, and which ones you can swap

**Rule of the whole design:** every swappable piece is **one file** that depends only on a small fixed
contract (`mltok/contracts.py`) and the shared helpers (`mltok/lib/`). It never imports another plugin.
Replace that file (or edit one config entry) and nothing else changes. `python -m mltok check` proves a new
file fits *before* you run anything, and the cache is keyed on the file's source code, so a swap recomputes
exactly the results that depended on it.

```
config (YAML) ──► ENGINE (never edited) ──► asks the registry for plugins by name
                      │
  sources ─► filters ─► [dedup+split] ─► normalizer ─► pretokenizer ─► algorithm ─► metrics ─► proxy model ─► selector ─► best/
  (plugin)  (plugin)      (engine)        (plugin)      (plugin)       (plugin)    (plugin)    (plugin)      (plugin)
```

## 1. The nine swap points (plugins)

| # | Kind | You replace this when you want to… | Built-in files (`mltok/plugins/<kind>/`) | Config key |
|---|---|---|---|---|
| 1 | **source** | change/add a **dataset** | `hf_stream.py` (Hugging Face), `local_files.py` (your jsonl/txt/parquet), `synthetic.py` (tests) | `sources:` |
| 2 | **filter** | change **what counts as a good document** | `length.py`, `ml_ratio.py`, `line_repetition.py` | `filters:` |
| 3 | **normalizer** | change **text normalization** | `identity.py` (`none`), `nfc.py`, `malayalam.py` (`ml`) | `components.normalizer`, `search.factors.normalizer` |
| 4 | **pretokenizer** | change **how text is cut before merging** | `whitespace.py` (`ws`), `word.py` (`word`) | `components.pretokenizer` |
| 5 | **algorithm** | change **how the tokenizer is trained** | `mlbpe.py` (`mlbpe`, `mlbpe_plain`, `mlbpe_morph`), `hf_bpe.py`, `sentencepiece.py` (`spm_unigram`, `spm_bpe`) | `components.algorithm` |
| 6 | **metric** | add/replace an **intrinsic measurement** | `basic.py`, `structure.py`, `roundtrip.py`, `stress.py` | `metrics:` |
| 7 | **proxy_model** | change the **test language model** | `gpt.py` | `proxy.model` |
| 8 | **selector** | change **how the winner is decided** | `default.py` | `selector:` |
| 9 | **store** | change **where checkpoints are pushed / restored from** (Hugging Face, a directory, S3 ...) | `hf_hub.py`, `local_dir.py`, `none.py` | `store:` |

Anything not in this table (engine, lib) is meant to stay as is.

## 2. Every file in the repo

### Engine — do not need to change when you swap things
| File | Role |
|---|---|
| `mltok/runner.py` | Orchestrates stages, one-factor-at-a-time search, caching/invalidation keys, per-GPU proxy workers, report. Knows no dataset, language or algorithm. |
| `mltok/ingest.py` | Runs `source` plugins into raw shards; applies `filter` plugins, exact dedup, deterministic train/eval split, sharding. |
| `mltok/proxy_train.py` | Trains/scores the proxy LM (data encoding, optimiser, mixed precision, micro-batching, bits-per-byte). Architecture comes from a `proxy_model` plugin. |
| `mltok/sync.py` | Checkpoint manager: batches/throttles pushes to whatever `store` plugin is configured, never lets a store error stop a run, restores finished work in a fresh session. |
| `mltok/registry.py` | `@register`, plugin discovery (built-in + your directories), source-code fingerprints. |
| `mltok/contracts.py` | The interface each plugin kind must satisfy (this file *is* the contract). |
| `mltok/contract_checks.py`, `mltok/templates.py` | `check` and `new-plugin` commands. |
| `mltok/cli.py`, `mltok/__main__.py` | Commands. |
| `mltok/gpu.py`, `mltok/selftest.py`, `mltok/verify.py` | Shared-GPU guard (60 GB cap), GPU self-test, exported-tokenizer compatibility check. |

### Library — shared helpers that plugins may import (keep the API stable)
| File | Role |
|---|---|
| `mltok/lib/bpe_core.py` | Language-agnostic constrained-BPE trainer + export to a plain HF `tokenizer.json`. The Malayalam rules are *arguments* (`token_ok`, `seed_pairs`), supplied by the algorithm plugin. |
| `mltok/lib/backends.py` | `HFBackend`, `SPMBackend`, `HubBackend`, tokenizer saving. |
| `mltok/lib/io.py` | Shard reading, word counting (multiprocess), alphabet selection, data fingerprints. |
| `mltok/lib/marker.py` | The word-start marker `▁` (owned by the engine so normalizers/pre-tokenizers stay unaware of it). |
| `mltok/lib/unicode_ml.py` | Malayalam character classes and the akshara (syllable) regex. |
| `mltok/lib/synth.py` | Synthetic text for smoke tests. |

### Plugins — the replaceable files: see the table in section 1.

### Config and run scripts
`configs/*.yaml` (smoke, scale_test, kaggle_dryrun, kaggle_small, dgx_confirm, dgx_final), `scripts/run_dgx.sh`, `scripts/build_zip.sh`, `notebooks/kaggle_runner.ipynb`.

## 3. Dependency rules (enforced by tests)

* plugin → `mltok.contracts`, `mltok.registry`, `mltok.lib.*`, third-party libraries. **Nothing else.** (`mltok check` and `tests/test_contracts.py` fail a plugin that imports another plugin, an engine module, or uses a relative import.)
* `lib` → never imports plugins (test).
* engine → never imports plugins; it only asks the registry by name (test).

That is why you can replace any plugin file without touching its neighbours.

## 4. Swap recipes

**Change the dataset** (no code): edit `sources:` — e.g.
```yaml
sources:
  - {type: local_files, id: my_news, path: /data/news/*.jsonl.gz, text: text}
  - {type: hf_stream,  id: wiki,     hf: wikimedia/wikipedia, name: 20231101.ml, split: train, text: text}
```
**Change the dataset** (new loader, one file): `python -m mltok new-plugin source my_crawl --dir my_plugins`, fill in `iter_docs()` (yield `{"text":..., "src":...}`), then `python -m mltok check --file my_plugins/my_crawl.py`, add `plugins_dir: [my_plugins]` and `{type: my_crawl, id: crawl}` to the config.

**Change a normalizer / pre-tokenizer / algorithm / metric / proxy model / selector / filter:** same three steps with `new-plugin <kind> <name>`. To *replace a built-in* keep its name (e.g. a file registering `normalizer` `ml` in your `plugins_dir`): a later registration with the same (kind, name) wins, and `mltok plugins` prints the override.

**Try variants side by side:** list them under `search.factors`; values are names or `{type: name, param: value}`.
```yaml
search:
  factors:
    normalizer: [none, ml, {type: ml, zw: drop}]
    algorithm:  [mlbpe, mlbpe_plain, spm_unigram, my_algo]
```
Useful commands: `python -m mltok plugins` (what is registered, from which file), `python -m mltok check [--file F | --kind K --name N]`.

## 5. What a change recomputes (content-addressed cache)

Every result lives under a key built from the **source code + parameters** of each plugin it depends on, plus data fingerprints. Nothing is reused unless every ingredient is identical; nothing is recomputed unless an ingredient changed.

| You change… | Word table | Tokenizers | Metrics | Proxy LMs |
|---|---|---|---|---|
| a **dataset / filter / split** | all | all | all | all |
| the **normalizer** or **pretokenizer** file/params | affected variants | affected variants | affected | affected |
| the **algorithm** file/params | reused | that algorithm's variants | those | those |
| a **metric** plugin or the `metrics:` list | reused | **reused** | all | reused |
| the **proxy_model** file or proxy settings (incl. each new seed) | reused | reused | reused | all (finished seeds are kept: `proxy_seed<N>.json`) |
| the **selector** | reused | reused | reused | reused (only the choice changes) |
| `vocab`, a factor value | only new variants | only new variants | new | new |

(Tested in `tests/test_plugins.py`: editing one normalizer file retrains only the variant that uses it; adding a metric retrains no tokenizer.)

Further efficiency built in: multi-process word counting (`resources.workers`), the Rust trainer for baselines, tokenizers shared across metric/proxy stages, one worker process per GPU for proxy runs (`proxy.gpus`), micro-batching for 16 GB GPUs, resumable runs (finished work is skipped), and a provenance block (plugin file hashes + data fingerprints) in `best/SELECTION.json` and `report.md`.

## 6. Contracts at a glance (full text in `mltok/contracts.py`)

| Kind | You implement | Must hold |
|---|---|---|
| source | `iter_docs()` → dicts with `text` (and optional `src`) | non-empty strings |
| filter | `keep(text) -> bool` | pure function of the text |
| normalizer | `normalize(text)` and `hf()` | the two give identical output (checked on 8 samples incl. empty string, old-style chillu, ZWNJ, emoji) |
| pretokenizer | `split(marked_text)` and `hf()` | pieces concatenate to the input; equals the HF pre-tokenizer |
| algorithm | `train(ctx) -> (Backend, info)`, `load(dir, normalizer)`; flags `needs_word_counts`, `uses_pretokenizer` | lossless round-trip; ids within vocab; `load` reproduces `train` |
| metric | `compute(backend, docs) -> dict` | flat, JSON-serialisable |
| proxy_model | `build(vocab, seq_len)` → `nn.Module` | `forward(idx[B,T]) -> logits[B,T,V]`, finite |
| selector | `eligible`, `score`, `proxy_score` | lower is better |
| store | `push(files, message)`, `pull(prefix, dir)`, optional `delete(paths)` | push→pull restores bytes exactly; pull never overwrites local files; unknown prefix → 0, no error; writes only under its own prefix |

Standard result keys other plugins may rely on: `fertility`, `bytes_per_token`, `roundtrip_ok`, `stress_roundtrip_ok`, `unk_rate`, `mark_start_rate`, `akshara_break_rate`, `bpb`. If you replace a metric and drop a key the selector needs, `selector` will raise a clear `KeyError` naming it; add the key or change the selector's `rank_by`.

## 7. Known limits of the plug-in design (honest)

* Plugins are independent of *each other*, not of the contract: if the contract itself changes (`mltok/contracts.py`), plugins may need an update. The contract is small and covered by tests.
* `lib` is a shared API. Changing it (rare) can affect several plugins — the test-suite exists to catch that.
* Plugins that need extra Python packages (e.g. `pyarrow` for parquet) must have them installed; the registry reports import failures in `mltok plugins` instead of crashing the run.
* The akshara-level constraint and seeds live in `plugins/algorithm/mlbpe.py` (Malayalam-specific by design). A different language needs its own algorithm/normalizer/filter files, not engine changes.

## 8. Checkpoints and resume (`store` + `sync`)

Config: `store: {type: hf_hub, repo: user/name, private: true, prefix: kaggle}` (default `none`) and
`sync: {min_interval_s: 120, ckpt_push_interval_s: 900, restore: true, include_cache: true, include_checkpoints: true, include_data: false}`.
The Hugging Face token is read **only from the environment** (`HF_TOKEN`); it cannot be put in a config file and is masked in notebook logs.

| Level | What is saved | When | How a restart uses it |
|---|---|---|---|
| finished unit | word table, trained tokenizer, `metrics.json`, each `proxy_seed<N>.json`, report, `best/`, `freeze.json` | right after the unit finishes (batched, at most every `min_interval_s`; forced at stage end, on error and on exit) | `restore` pulls them; the content-addressed cache reuses every one whose key still matches |
| inside a proxy-LM run | model + optimizer + scaler + RNG every `proxy.checkpoint_every_steps` steps | local at every interval; pushed at most every `ckpt_push_interval_s`; only the newest is kept remotely and it is deleted when the run finishes | training resumes from that step and reaches the same result as an uninterrupted run (tested) |
| cleaned data | shards | only with `sync.include_data: true` (can be GBs) | otherwise re-ingest + re-clean; cleaning is deterministic, so cache keys still match |

* A store failure (network, token, rate limit) is logged and the files stay queued; the run continues. It is retried at the next flush.
* The `hf_hub` store writes only under its `prefix` folder and creates the repo private if missing; other files in the repo are never touched.
* Not checkpointed (honest limits): **mid-way through one BPE training** and **mid-way through word counting of one shard** — those are minutes-scale units that are simply redone. Everything before and after them is saved.
* Every checkpoint is a new LFS object in the repo history; for large DGX checkpoints consider `squash_every` (irreversible history squash) or a dedicated repo.

## 9. Data mixing, SentencePiece export, known limits (added after the first real run — see RESULTS_LOG.md)

* **Source mixing.** `split.balance: true` (default) interleaves raw sources by `split.weights` (default equal) so a `max_train_mb` cap is *shared*; `false` reads files one after another and the first source fills the cap alone. Per-source training bytes and `train_share` are written to `clean_stats.json`; the freeze checklist fails if one source exceeds 90 %. The eval split mirrors the raw pool (per-source fertility is reported, so out-of-domain effects are visible).
* **`resources.max_word_types`** caps the word table used by `mlbpe` only; a warning prints if the kept types cover < 97 % of the text (algorithms that stream all text are then not comparable).
* **SentencePiece Unigram -> Hugging Face.** `spm_unigram` writes `tokenizer.json` next to `spm.model` (rebuilt piece table, byte fallback, digit splitting, our normalizer). Each export is parity-checked against SentencePiece on sample text + built-in edge cases: *identical ids* or an *exact Viterbi score tie* are accepted, anything else rejects the export (`info.hf_export`). For an older run: `python -m mltok export-hf --spm <best>/spm.model --normalizer ml`. `spm_bpe` is not converted (HF BPE is a different model class).
* **Known limitations.** Text containing special tokens does not round-trip exactly (marker added per segment); plain-text pretraining is unaffected. SentencePiece and the converted tokenizer can break an exact Viterbi tie differently (rare, equally probable, lossless).

## 10. Sync details learned from a real run
* Pushes run in a **background thread**; training never waits on the network. Stage ends, errors and exits use `flush(force=True)` and wait.
* A mid-training checkpoint is pushed only after a **full** `ckpt_push_interval_s` (default 900 s) since the process started, so short jobs never upload one; only long jobs (e.g. DGX) do.
* A file modified while it is uploading stays queued and is re-pushed.
* `proxy_summary.json` stores per-variant seed statistics; the freeze checklist uses the **standard error of the difference** between two seed-averaged BPBs (>= 2 required, 2-3 labelled borderline).

## 11. Pinning a decision, and the two DGX steps
* `python -m mltok pin -c <cfg> --variant <name> --reason "<why>"` copies a finished, **eligible** variant to `best/` and writes `SELECTION.json` with `pinned: {reason, proxy_winner, bpb_gap_pct, z_vs_proxy_winner}` + provenance. The reference ("proxy winner") is the best *eligible* variant, so a lossy tokenizer can never be the benchmark. The notebook's freeze gate then requires the pinned choice to give up < 1 % BPB.
* `configs/dgx_confirm.yaml` — step 1: Unigram 32K/48K/64K and the best BPE on a `small` proxy (3 seeds, 8 GB balanced text). `configs/dgx_final.yaml` — step 2: train the decided tokenizer on all data, no proxies. `scripts/run_dgx.sh <gpu> <config>`.
* `tests/test_configs.py` validates every shipped config (plugins exist, params construct, vocab is a multiple of 64, DGX configs never use `gpus: auto` and stay within the 60 GB cap).
