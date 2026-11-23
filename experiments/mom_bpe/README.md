# MOM-BPE development and validation

The later independent review records the corrected implementation, fallback attribution and source-assisted morphology probe. [Independent review](REVIEW.md)

MOM-BPE is a CPU-validated implementation of Malayalam orthographic units and confidence-weighted morphological guidance. Training on the thesis corpus is blocked. The 48K evaluation foundation defines the protocol, but the replacement corpus has yet to be frozen and its annotation queue has zero independently verified references. The existing 48K artifact and historical results remain available at their original identities. [Evaluation foundation](../../audit/48k-baseline/FOUNDATION.md) · [Evaluation protocol](../../audit/48k-baseline/evaluation-protocol-v1.json)

## Components

| Component | Implementation |
| --- | --- |
| Normalization | `ml_conservative` applies NFC and optionally maps the five standardized legacy chillus in terminal, unconjoined contexts. It leaves explicit ZWNJ, samvruthokaram, archaic characters and ambiguous spellings intact. `terminal_chillu: false` selects NFC alone. |
| Pretokenization | `script_runs` resolves Unicode Script and Script_Extensions for a configured script list. Compatible letters and attached marks form runs. Whitespace, punctuation, numbers and nonconfigured scripts remain reversible. The algorithm coalesces cuts inside either protected unit definition consistently across the arms. |
| Orthographic units | Codepoint, Unicode extended grapheme cluster (EGC), or the new Malayalam grammar. The grammar groups virama-linked bases, chillu conjuncts, vowel signs and final virama including samvruthokaram. ZWNJ ends a conjunct. The grammar retains dot reph and historical chillus and falls back to EGC for other material. The earlier akshara reference and EGC measurements remain separate. |
| Morphology | Exact surface projections require complete concatenation, compatible analyses, independent reviewer identifiers, legal offsets and explicit fixed-point confidence. Non-identical underlying and surface parts need a supplied allomorph or rewrite. Unresolved or in-unit boundaries supply no merge penalty. Optional Mlmorph observations only select exact readings already present in the reviewed rules. The analyzer's ranking weights are unused. |
| Weighted BPE | A pair's score is its occurrence count minus `λ × Σ(confidence)` over crossed sites, accounting for corpus multiplicity. There is one global weight and per-site confidence in thousandths. Integer arithmetic makes ties reproducible. Alphabet members are protected initial units and omitted members fall back to exact UTF-8 bytes. Inference starts from the same units and replays the saved rank order without running an analyzer. |
| Persistence | The versioned artifact contains preprocessing, initial symbols, merge ranks, special IDs, runtime and encoder fingerprints. Loading rejects a different encoder identity. A same-volume atomic, exclusive write preserves an existing artifact. Plain-text encoding preserves `▁` and registered-control spellings as literal text. Trusted controls use their separate reserved IDs. |

The serialized normalizer and pretokenizer use Hugging Face Tokenizers primitives. MOM-BPE inference uses the project backend, allowing exact arbitrary initial-unit and fallback behavior. A stock `AutoTokenizer` cannot load the resulting MOM-BPE artifact directly. [Unicode Malayalam specification](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-12/) · [Unicode text segmentation](https://www.unicode.org/reports/tr29/)

## CPU verification

The experiment script fitted five **384-slot diagnostic artifacts** on 32 small authored training records. It loaded and SHA-256-verified the immutable T0 artifact and evaluated every arm on 54 authored ordinary probes plus separate fidelity probes. The table shows exact non-fallback Malayalam transitions under the historical akshara reference and EGC.

| Arm | Eligible transitions | Akshara breaks | EGC breaks | Ordinary byte-fallback IDs |
| --- | ---: | ---: | ---: | ---: |
| T0 published | 133 | 6 | 6 | See original audit |
| T0-matched | 982 | 606 | 606 | 1,631 |
| T1 grapheme | 24 | 0 | 0 | 6,092 |
| T2 akshara | 24 | 0 | 0 | 6,092 |
| T3-off | 24 | 0 | 0 | 6,092 |
| T3 weighted | 19 | 0 | 0 | 6,092 |

T2 and T3-off produced exactly the same initial symbols, merge sequence and IDs. T3 used four supported synthetic forms and changed the merge order. A separate regression compared each weighted merge with an independent pair recount and checked the final segmentations against rank replay. All new arms passed normalized round-trip on 54/54 ordinary probes, 26/26 Unicode stress probes and 11/11 explicit fidelity cases. Raw fidelity was 54/54, 21/26 and 9/11, respectively. The authored stress panel had zero eligible measured transitions in T1–T3, so its zero break count supplies no orthographic efficacy measurement. Large fallback counts and changing coverage also make the ordinary rates unsuitable for comparing production quality.

Four of five synthetic forms projected five weighted surface boundaries. The fused sandhi example abstained. The optional pinned Mlmorph run returned analyses for four forms and none for one. Projection from the analyzer abstained on all five because the supplied rule set did not resolve every reading. There are zero independently adjudicated Malayalam alignment results for this milestone. The scorer records exact boundary precision/recall and token-ID consistency across repeated surface stems when approved reference forms become available. [Diagnostic evidence](diagnostic-results-v1.json) · [Synthetic alignment rules](diagnostic_alignments.json) · [Existing annotation queue](../../audit/48k-baseline/morphology-annotation-v1.tsv)

### Reproduction

The full CPU regression run passed 246 tests with two CUDA skips and zero failures. The final focused rerun passed 150 tests with zero failures or skips. Both runs reported two dependency deprecation warnings from the existing SentencePiece contract test. All three notebook code cells executed independently.

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python -m pytest -q -p no:cacheprovider -ra
python experiments/mom_bpe/diagnose.py --out "$env:TEMP/mom-bpe-diagnostics"
```

To reproduce the optional analyzer column, use the previously verified local wheel directory as `PYTHONPATH` and select a fresh output directory with `--mlmorph`. The diagnostic script writes its completion record only after checking artifact reload, input fingerprints and the zero-weight control. The notebook exposes independent cells for tests, diagnostics and freeze preflight. [Dependency manifest](../../audit/48k-baseline/morphology-dependencies.json) · [CPU notebook](../../notebooks/mom_bpe_cpu.ipynb)

## Gates and DGX matrix

`verify_frozen.py` checks the protocol and normalizer fingerprints, ordered shard sizes and SHA-256, raw and canonical document identities, source and domain totals, exact train/eval leakage, annotation coverage and held-out family separation from training alignments. Two-maintainer review, license rights and near-duplicate clustering still need independent confirmation. The verifier checks the referenced review artifacts' integrity. `prepare.py` emits fingerprints, verified data inputs and resolved arm specifications. With an existing MOM-BPE artifact it also computes exposure under the two budgets without training.

Use a `mom-frozen-inputs-v1` manifest with `protocol_sha256`, `normalizer_fingerprint`, `reviewers`, `sources`, `source_counts` and ordered `splits.train` and `splits.eval`. Each file entry gives a relative `path`, `size` and `sha256`. Include entries for the two review receipts (`near_duplicate_review` and `train_eval_separation_review`), `orthography_panel`, `morphology_panel` and `training_alignments`. Each natural text JSONL record has `id`, `group`, `src`, `domain`, `text`, `sha256`, `canonical_sha256` and `canonical_bytes`. Keep the synthetic manifest used by the unit tests separate from the reviewed corpus.

```powershell
python experiments/mom_bpe/verify_frozen.py path/to/frozen-inputs.json
python experiments/mom_bpe/prepare.py --manifest path/to/frozen-inputs.json --out path/to/plan.json
```

| Arm | Primary comparison |
| --- | --- |
| T0 | Pinned published 48K, descriptive comparator on the common held-out set |
| T0-matched | Active 48K codepoint control with shared preprocessing and fallback |
| T1 | EGC initialization compared with T0-matched |
| T2 | Malayalam orthographic initialization compared with T1 |
| T3-off | T2 plus the same morphology inputs as T3 and zero weight |
| T3 | T3-off with a proposed morphology weight of 0.4 |

1. After the freeze and annotation gates pass, verify full vocabulary occupancy, fallback exposure, fidelity and throughput on limited development inputs. Confirm T2 and T3-off still yield equivalent IDs.
2. After explicit DGX authorization, use one eligible GPU for a matched pilot. Verify physical IDs and availability, excluding physical GPU 7. Keep CPU tokenizer diagnostics separate.
3. Compare five matched arms across the two budgets and three paired seeds (30 downstream runs). The planned GPT at 512 context tokens has 29,692,800 parameters, including 18,628,608 embedding parameters, verified by CPU instantiation. T0 can be added descriptively on the same evaluation freeze. Archive the config, manifests, software and hardware identities, checkpoints and actual training exposure for each run.
4. Under fixed source bytes use the same ordered document set. Under fixed non-padding predicted tokens report actual source bytes and incomplete document coverage. Given BOS, predict text tokens and one EOS per complete document. A partial UTF-8 fallback span leaves its byte count unknown. The diagnostics exercised both ledgers, including 3,456 canonical bytes and 408 predicted tokens under the source budget, and an exact 120-token cap under the token budget.
5. Use paired seeds and document-family confidence intervals, identical canonical evaluation bytes and a preregistered BOS/EOS policy. Report compute, context bytes, token fertility, bytes per token, unknown rate, vocabulary use, per-domain results, fidelity, boundary incidence and exclusions. Keep the proposed BPB margin of +0.005 and linguistic thresholds subject to preregistration before held-out scoring. Archived BPB belongs to its original validation dataset.

Full-corpus and downstream runs are **not ready**. The approved corpus and independent annotations are missing. At production scale, measure alphabet coverage and worst-case inference throughput, and validate interrupted tokenizer fitting. A completed runner stage is reusable. An interruption during fitting can require repeating that fit. The existing proxy checkpoint path is unchanged, and its current stream packing needs separate validation against the planned per-document downstream protocol before a BPB comparison. No live Hugging Face credentials or repositories were used in this milestone. [Ablation configuration](arms.yaml)

## Research contribution and prior work

Grapheme Pair Encoding establishes grapheme initialization, and the Indic evaluation already includes Malayalam. Script-aware orthographic processing and EGC initialization are adaptations. [Grapheme Pair Encoding](https://aclanthology.org/2025.coling-main.400/) · [Indic grapheme tokenization evaluation](https://aclanthology.org/2026.findings-emnlp.1093/)

MorphTok uses sandhi-aware segmentation and constrained BPE for Hindi and Marathi. MorphBPE constrains training merges with morphological boundaries. The contribution to test here is the combination of a Malayalam-specific written-unit grammar, explicitly reviewed surface rewrites with abstention, and a graded crossing penalty compared directly with the same zero-weight pipeline. The implementation establishes that the mechanisms and controls operate, while comparative linguistic efficacy, research novelty and downstream gains remain open. [MorphTok](https://arxiv.org/abs/2504.10335) · [MorphBPE](https://aclanthology.org/2026.findings-acl.2068/) · [MorphScore evaluation](https://arxiv.org/abs/2507.06378)
