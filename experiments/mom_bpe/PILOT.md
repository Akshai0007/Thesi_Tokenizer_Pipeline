# Bounded MOM-BPE pilot and research readiness

GitHub's `research/mom-bpe-review` reference and the local review commit both resolve to `31a43a88f1b76483ec5115a08a9828e1b8d441d9`. The branch was already published when checked. The additions described here are local working-tree changes.

## Corpus identity and provenance

The historical DGX evaluation remains an archived result under `c1321250cd78`, with 22,120 documents from the FineWeb-2, Sangraha and Wikimedia source pool. The historical selection records a shard-name-and-size fingerprint. It cannot establish content identity without the original shards or trusted ordered document hashes. The raw private backup was identified through metadata only. The original evaluation has yet to be recovered.

`malayalam-eval-protocol-v1` fixes the plan for separately sourced, pinned, licensed and deduplicated natural and orthographic panels. The repository contains authored diagnostics, protocol and review inputs. It has no verified replacement corpus manifest. The 54 ordinary probes and the Unicode cases remain engineering diagnostics. No historical result was reassigned to them. [Foundation audit](../../audit/48k-baseline/FOUNDATION.md) · [Evaluation protocol](../../audit/48k-baseline/evaluation-protocol-v1.json)

## Candidate controls and resumption

| Arm | Initial symbols | Readiness |
| --- | --- | --- |
| A | Unicode codepoints | CPU fixture tests passed. A reviewed freeze is required for the representative pilot. |
| B | Unicode extended grapheme clusters | Same freeze and initial alphabet/fallback policy as A, applied to grapheme symbols. |
| C | Malayalam akshara units | Same freeze and controls as A and B. Projection and weights remain disabled. |
| Later T3 and T3-off | Malayalam akshara with matched morphology input | Requires independently adjudicated training alignments and a separate held-out morphology panel. |

Preparation verifies the natural and orthography freeze, then materializes the longest prefix of complete training documents that fits the canonical UTF-8 budget. Every arm reads that same fingerprinted local shard in the same order. The requested 48,000 slots include the same seven control IDs and 256 fallback byte IDs. Report the actual occupied vocabulary as well as the target. An underfilled scientific arm retains its fit receipt and stops before evaluation. The public T0 tokenizer remains a descriptive reference at its original revision. [Separate engineering mode](RECOVERY_AND_ENGINEERING.md)

Metrics report canonical whitespace-word fertility, canonical UTF-8 bytes per token, exact raw and canonical round-trip counts, eligible transitions, EGC and historical akshara violations, and excluded transitions. The same byte-ID test measures byte fallback in every arm. Per-domain fallback is the number of byte-fallback IDs divided by all emitted text IDs in that domain. Orthography remains a separate panel. Encoder batch output includes every text ID, including all fallback bytes. Literal strings such as `<pad>` remain text. The downstream caller supplies the reserved control IDs separately.

Each process fits one arm and saves immutable merge generations. An interruption can resume at the latest compatible verified generation. Recovery rebuilds pair statistics from the saved symbol sequences and boundary flags. A damaged newest generation can fall back to the prior verified generation. Generations remain on disk. Completed fits and metrics are hash-checked on reuse. Preparation and input counting can repeat after an interruption. Timing and peak process RSS describe the successful fit attempt, so a resumed measurement excludes work lost during prior attempts. Compare resource use with fresh attempts under the same runtime.

The focused offline suite passed 187 tests with zero failures and zero skips. One existing SentencePiece contract test emitted two dependency deprecation warnings. Tests exercised identical merge ranks after interruption, corrupt-generation fallback, identical source exposure across arms, cached result reuse, canonical BPB accounting and simulated Hub restoration. The protocol-only input was rejected before creating a pilot directory.

The default pilot cap is 4 MiB canonical training text and the hard cap is 8 MiB. The natural evaluation and separate orthography inputs each have a 16 MiB cap. Run one CPU process per arm sequentially for comparable peak RSS. Allow memory for the word table, pair statistics and one serialized checkpoint. Each arm can write up to 24 ordinary checkpoint generations at the 2,048-merge interval and final save. Size scratch space from the first generation, allowing for all three arms and temporary writes. A provisional starting allocation is 4 GiB of available RAM, with disk needs measured locally. Production-scale time and RSS have yet to be measured.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$freeze = 'path/to/approved/frozen-inputs.json'
$pilot = 'path/to/new/local-pilot'
python experiments/mom_bpe/pilot.py prepare --manifest $freeze --out $pilot --train-bytes 4194304
python experiments/mom_bpe/pilot.py run --plan "$pilot/plan.json" --arm A
python experiments/mom_bpe/pilot.py run --plan "$pilot/plan.json" --arm B
python experiments/mom_bpe/pilot.py run --plan "$pilot/plan.json" --arm C
```

The `prepare` command does not train. After the gate passes, repeat a `run` command to resume or reuse that arm. Store the run root outside Git. Keep older prototype artifacts with their pinned review checkout because the encoder identity checks source and runtime versions.

## Canonical BPB reference

The reference scorer verifies the frozen canonical SHA-256 and byte length for each document, validates lossless encoding and token ID range, and makes one prediction call per document. BOS provides context. Every text ID and one EOS are targets. It sums their negative log-likelihoods in nats, converts to bits by dividing by `ln(2)`, and divides by the total canonical UTF-8 input bytes. Tests cover NFC length changes, literal control spellings, fallback IDs and distinct per-document losses. This verifies the accounting independently of any model.

The existing proxy smoke runner concatenates streams and measures raw bytes. Its BPB remains separate. Before a research BPB result, validate a model-backed caller's document resets, context windows, padding masks and complete loss coverage. Use identical canonical documents and the same BOS/EOS policy across arms. [Reference scorer](../../mltok/lib/canonical_bpb.py)

## Separate morphology preparation

The blind packet contains 96 earlier isolated prompts and 12 contextual desk prompts. Reviewer sheets suppress analyzer suggestions and prior decisions. Open the evidence only after independent review, then preserve unresolved and abstained cases in the adjudication sheet. On changes from raw to canonical text, check the actual normalized surface before entering canonical offsets. The supplied packet contains zero authorized training labels.

The evidence pins Mlmorph 1.4.3, SFST 1.10.0 and the 18,619,083-byte compiled transducer at SHA-256 `f6eb3d3fb73fdc7e1f422d5a0b14e20e33fc736d4a830e4395213577611dfa72`. Its ranked readings and the source-assisted judgments are retained as review aids. The separate morphology preflight requires adjudicated training alignments that pass the reviewed surface projection engine and are disjoint from held-out forms and families. The zero-weight T3-off uses the same verified input as T3. [Packet manifest](morphology-review-v2/manifest.json) · [Original reviewer guidance](../../audit/48k-baseline/FOUNDATION.md)

```powershell
python experiments/mom_bpe/morphology_packet.py --out path/to/new-review-packet
python experiments/mom_bpe/prepare.py --manifest path/to/approved/morphology-freeze.json --out path/to/morphology-plan.json
```
