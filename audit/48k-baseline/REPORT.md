# Audit of the published Malayalam 48K tokenizer

## Executive summary

The published BPE tokenizer at revision `cff0d31c84b598436087f21cc8183021993831e1` is byte-for-byte identical to the archived DGX confirmation winner. The archived selection records 22,120 evaluation documents and fingerprint `c1321250cd78`. The original shards were unavailable in the archive locations inspected. The 54 authored diagnostic sentences therefore support failure reproduction and comparison between methods. They do not reproduce the historical benchmark.

- **Two normalized fidelity failures reproduced.** A literal `▁` decodes as a space. An inline registered chat special token can introduce a space before following text. All 54 ordinary diagnostic sentences and the pipeline's fixed stress suite passed normalized round-trip.
- **Six exact-offset akshara violations in 133 eligible transitions.** All six occur after virama inside a pretokenized piece. None of the corresponding adjacent token pairs has a merge rule or a joined vocabulary entry. The current reconstructed-surface metric counts two additional violations spanning emoji byte fallback. This identifies a metric-bias mechanism on the diagnostic corpus and leaves its contribution to the historical rate unknown.
- **The vocabulary and merge graph pass structural checks.** The audit found no repeated serialized keys, shared vocabulary IDs, gaps, or missing merge dependencies. It flagged five pieces with unexpected consecutive Malayalam vowel signs and two pieces with repeated virama. Their origin and utility need corpus evidence.
- **The strongest thesis direction is an independently reviewed Malayalam surface-boundary benchmark and a controlled confidence-weighted training ablation.** Grapheme-based tokenization, morphology-constrained BPE, and intermediate-vocabulary refinement already have published precedents. Fix the two fidelity issues under a shared policy before attributing gains to linguistic changes.

## 1. Identity and data provenance

| Evidence | Value |
|---|---|
| Published tokenizer | `Akzhai/malayalam-mlbpe-tokenizer` |
| Pinned revision | `cff0d31c84b598436087f21cc8183021993831e1` |
| Archived selection | `backup_runs_final/dgx_confirm/best/SELECTION.json` at archive revision `4d018390ee2c07e9665cb1e384214ddb4e03c704` |
| Historical winner | `mlbpe-ml-word-v48000` |
| Baseline file count and size | 3 files, 4,789,744 bytes |
| Tokenizer JSON SHA-256 | `2f4b7e8b0706007f7ab3f0b8b2cc1b169094072c334585dabc4081d455e69ff5` |
| Historical evaluation identity | 22,120 documents, 4,602,360 whitespace words, 122,510,792 UTF-8 bytes, legacy fingerprint `c1321250cd78` |
| Diagnostic fixture SHA-256 | `a712126a68d444ad31a86a7da40a10e58c4b61b223d112a7591c58a874aa69d1` |

The archived winner's `tokenizer.json` has the exact published JSON digest. The baseline originals remain in the revision-keyed [original directory](original/cff0d31c84b598436087f21cc8183021993831e1/). Size and SHA-256 were checked before and after diagnostics. Each file's digest is in [baseline-artifacts.json](baseline-artifacts.json).

The archive head inventory had 301 files and no named evaluation-shard group. Thirteen metadata files were inspected at that pinned revision. Eight recent archive revisions also exposed no named eval shards in their inspected inventories. The DGX selection records the expected old fingerprint, which was based on shard basenames and byte lengths. The record establishes the reported dataset identity without supplying the bytes needed to independently recompute it. Historical metric fingerprints also differ from the fingerprints reported by the current registry for the default metric specs, so the archived values require the original metric implementation and settings for exact replication. The archive inspection downloaded only selected metadata and one tokenizer file. [Retrieval record](retrieval.json) · [Historical evidence](historical-evidence.json)

## 2. Reproduced evidence and new measurements

**Reproduced identity** means the archived tokenizer matched the published file by SHA-256. The historical values below were recovered from that archived selection. No proxy model or tokenizer was trained in this audit.

| Measure | Archived historical result | New 54-document diagnostic result |
|---|---:|---:|
| Token fertility per whitespace word | 1.758184 | 2.129371 |
| UTF-8 bytes per token | 15.140120 | 11.765189 |
| Characters per token | 5.559772 | 4.402299 |
| Rényi efficiency, existing `basic` metric | 0.426450 | 0.377902 |
| Observed vocabulary share | 97.3792% | 0.9833% (472 IDs) |
| Unknown tokens | 0% | 0/609 |
| Mark-start rate, current token-head diagnostic | 0.2897% historical report | 0.7595% (3/395) |
| Akshara breaks, reported/reconstruction metric | 6.9482% historical report | 5.9259% with the current reconstruction metric (8/135 after tracing) |
| Akshara breaks, new exact-offset metric | Unavailable | 4.5113% (6/133) |
| Extended grapheme breaks, new exact-offset metric | Unavailable | 4.5113% (6/133) |
| Normalized document round-trip | 100% | 54/54 |
| Raw document round-trip | Unavailable | 54/54 |
| Fixed stress round-trip | 100% | 100% with the current suite |
| Proxy BPB across three seeds | 0.472178 ± 0.000408 population SD | Not measured |

The diagnostic corpus has 286 whitespace words, 7,165 UTF-8 bytes, and 609 tokens. It is authored and deliberately includes difficult examples. Neither column estimates a gain over the other. The three archived BPBs are 0.471896, 0.472756, and 0.471884. [Measurements](measurement-v3/measurements.json) · [Historical evidence](historical-evidence.json)

### Efficiency by authored register

Every row contains six authored examples. The regional probes are illustrative and have no dialect-population sampling claim.

| Register | Words | Tokens | Fertility | Bytes/token | Akshara breaks / eligible |
|---|---:|---:|---:|---:|---:|
| News style | 41 | 53 | 1.293 | 21.094 | 0/6 |
| Literary style | 35 | 60 | 1.714 | 15.783 | 0/19 |
| Encyclopedic style | 30 | 58 | 1.933 | 15.517 | 1/22 |
| Technical style | 37 | 71 | 1.919 | 14.732 | 0/22 |
| Conversation style | 30 | 46 | 1.533 | 14.543 | 0/7 |
| Regional-register probes | 20 | 42 | 2.100 | 10.452 | 2/14 |
| Code mixing and loanwords | 37 | 70 | 1.892 | 8.729 | 0/3 |
| Rare and complex words | 23 | 63 | 2.739 | 14.984 | 2/34 |
| Numbers, punctuation, and other scripts | 33 | 146 | 4.424 | 3.363 | 1/6 |

On this machine the median of five warmed 864-document batches was about 278,190 encoded and 1,121,999 decoded tokens per second with tokenizer parallelism disabled. These are local tokenizer-library timings. The causal-LM collator and data loading are excluded from the timed region. A 135,167-codepoint input (356,351 bytes) emitted 28,672 tokens and round-tripped exactly after lexical normalization. [Domain analysis](measurement-v3/domain-analysis.json) · [Tokenization traces](measurement-v3/diagnostic-tokenizations.jsonl)

## 3. Full vocabulary and merge integrity

The tokenizer is BPE with byte fallback, no merge dropout, no continuing-subword prefix or end-of-word suffix, and a word-oriented split regex. The seven registered special tokens are `<pad>`, `<s>`, `</s>`, `<unk>`, `<|im_start|>`, `<|im_end|>`, and `<|endoftext|>`. The first four IDs are 0, 1, 2, and 3. The declared Hugging Face normalizer serializes NFC, six legacy-chillu substitutions, an invisible/control-character removal rule and the marker transport suffix. The decoder replaces marker glyphs with spaces, resolves fallback bytes, joins pieces and strips one leading space. Ordered merges begin with consonant-plus-virama seeds. [Tokenizer configuration](measurement-v3/tokenizer-configuration.json)

| Full-vocabulary check | Count | Share of 48,000 |
|---|---:|---:|
| Special tokens | 7 | 0.0146% |
| Byte-fallback tokens | 256 | 0.5333% |
| Reserved padding fillers | 0 | 0% |
| Repeated serialized keys, ID aliases, or missing IDs | 0 each | 0% |
| NFC- or declared lexical-normalization-equivalent groups preserving word-start context | 0 each | 0% |
| Legacy chillu pieces and legacy/atomic duplicate pairs | 0 each | 0% |
| Pieces containing atomic chillus | 5,474 | 11.4042% |
| Pieces starting with a Malayalam combining mark or joiner | 25 | 0.0521% |
| Pieces containing an alphabetic letter from outside Malayalam script | 2,878 | 5.9958% |
| Pieces with both Malayalam and non-Malayalam letters | 6 | 0.0125% |
| Unexpected control/format or unassigned-codepoint flags under this audit | 0 each | 0% |
| Pieces with suspicious consecutive Malayalam vowel signs | 5 | 0.0104% |
| Pieces with repeated virama | 2 | 0.0042% |
| Pieces ending in virama | 10,164 | 21.1750% |
| Produced merge nodes consumed by a later merge | 13,190 | 27.4792% |
| Intermediate nodes unobserved in the 54 fixtures | 12,962 | 27.0042% |
| All vocabulary IDs unobserved in the 54 fixtures | 47,528 | 99.0167% |
| IDs observed exactly once in the 54 fixtures | 429 | 0.8938% |

The 46,411 merges have 46,411 unique outputs. There are zero duplicated merge pairs, missing inputs, missing outputs, or operands unavailable at their merge rank under reconstruction from non-output symbols. Contextual prefix variants such as a word-start-marked form and its unmarked counterpart serve different positions. Pieces used for byte fallback, signs, multilingual text, or legitimate virama-final forms are not vocabulary defects. The five consecutive-sign flags include `ത്തിെ` and `▁വെെ`. Both repeated-virama examples are observable pieces (`▁മുസ്്`, `▁മുസ്്ലിം`). Their presence is confirmed. Inferring erroneous source documents or estimating training-time utility would require corpus evidence. [Vocabulary summary](measurement-v3/vocabulary-analysis.json) · [All 48,000 entries and flags](measurement-v3/vocabulary.jsonl)

## 4. Unicode and orthography

The 26 Unicode fixtures cover composed and decomposed vowel signs, atomic and legacy chillus, virama, samvruthokaram, conjuncts, ZWJ/ZWNJ, anusvara, visarga, candrabindu, dot reph, archaic codepoints, leading marks and malformed mark sequences. The serialized lexical normalizer produced NFC output, was idempotent and matched the current Python `ml` reference in all 26 cases. The check excludes the exact two serialized transport steps (prepend `▁`; replace space with `▁`) from the lexical normalization comparison. This follows the distinction between canonical normalization and application-specific transforms. [Unicode UAX #15](https://unicode.org/reports/tr15/) · [Unicode case results](measurement-v3/unicode-analysis.json)

| Verified case | Observed behavior | Source |
|---|---|---|
| Literal `ക▁ക` | Decodes as `ക ക`; lexical-normalized input still contains the literal marker | Transport uses the same symbol for literal glyph and whitespace; the decoder converts it to a space |
| `മലയാളം<|im_start|>വാക്ക്` | Decodes with a space inserted before `വാക്ക്` | The registered special separates text spans; the following span receives a new marker prefix |
| Legacy chillu and canonical decompositions | Canonicalized on the requested comparison policy | Explicit chillu replacements and NFC |
| ZWJ after a matching consonant+virama inside a larger context | The probe U+0D15 U+0D4D U+200D U+0D37 normalizes to `ൿഷ` | Replacement matches the substring independent of context; orthographic intent needs review by a Malayalam reader |
| ZWNJ, archaic probes and malformed mark sequences | Recorded against lexical expectations case by case | Malformed orthography can be preserved as Unicode text |
| Three Python strings containing unpaired surrogates and one truncated UTF-8 sequence | All four rejected at the tokenizer binding or strict decoder boundary | Ill-formed Unicode input was not counted as an orthographic defect |

Across the 26 cases, raw and normalized round-trip passed 15 and 24 respectively. The two normalized failures are exact reproductions on this probe set. The other raw differences follow declared transforms such as NFC, removal of selected invisibles or CR, and legacy chillu substitution. A joiner transform can be semantically sensitive even when it implements the declared policy. Annotate legacy-chillu and deliberate-conjunct contexts separately before changing that policy. [Unicode case results](measurement-v3/unicode-analysis.json)

### Boundary root cause

The offset audit evaluates two distinct references. One is the pipeline's Malayalam akshara regex. The other is the installed `regex` module's extended grapheme cluster segmentation. It counts only genuine consecutive lexical token transitions with an exact shared codepoint offset and Malayalam or joiners on both sides. It excludes byte and special transitions, and reports ambiguous offset exclusions. The 54 diagnostics excluded 24 byte or special transitions and zero ambiguous transitions. The Unicode probes produced different counts for the two rules, including 11 akshara and 13 EGC violations over 32 eligible transitions on lexically normalized inputs. [Unicode UAX #29](https://www.unicode.org/reports/tr29/) · [Boundary analysis](measurement-v3/boundary-analysis.json)

| Input excerpt | Observed pieces | Expected |
|---|---|---|
| പ്രകാശസംശ്ലേഷണത്തിലൂടെ | `ശ്` → `ലേ` | Keep the position inside `ശ്ലേ` free of token boundaries |
| എന്തൂട്ടാ | `▁എന്` → `തൂ` | Keep `ന്തൂ` together according to both references at that position |
| നിഷ്കർഷാപൂർവം | `▁നിഷ്` → `കർ` | Keep the `ഷ്` + `ക` conjunct boundary internal |
| പ്രത്യുത്പന്നമതിത്വം | `ത്` → `പന്ന` | Keep the `ത്` + `പ` conjunct boundary internal |

All six observed exact-offset breaks were after virama, inside a piece produced by the pretokenizer, with a starting letter on the right. Each adjacent pair lacks both a direct merge and a concatenated entry in the serialized vocabulary. This is consistent with the current Malayalam BPE constraint protecting the **start** of a piece from a combining mark while still permitting the piece on the left to end with virama. A pretokenization boundary did not force these six cuts. The current legacy metric joins token strings after dropping byte tokens. On the authored emoji family example this makes adjacent joiners appear connected across dropped emoji bytes and adds two apparent Malayalam boundary violations. This explains the difference between the two metrics on these fixtures. Historical contributions from training data, frequency, word-table coverage, or this metric artifact remain unverified without the original data and historical implementation. [Root-cause traces](measurement-v3/boundary-root-causes.json)

## 5. Morphological behavior

Mlmorph 1.4.3 with SFST 1.10.0 returned candidate analyses for 21 of the 24 forms, and 11 forms had multiple candidates. Examples include plural genitive for `കുട്ടികളുടെ`, multiple possible analyses for `കടൽക്കര`, and a locative analysis among the alternatives for `വീട്ടിൽ`. `ക്ഷേത്രോത്സവം`, `നിഷ്കർഷാപൂർവം`, and `പ്രത്യുത്പന്നമതിത്വം` received no analysis in this probe set. Candidates and weights are retained as analyzer evidence and have no asserted surface offsets. Mlmorph is an analysis/generation transducer, so its lemma and tags cannot be used directly as a splitter over surface text. [Thottingal's Mlmorph paper](https://aclanthology.org/W19-6801/) · [Morphology analysis](measurement-v3/morphology-analysis.json)

- **Manually specified concatenative fixtures** had 1 matched reference boundary out of 16 across 13 forms. Three eligible predicted boundaries give precision 1/3, recall 1/16, and F1 0.1053 against these fixtures. Three exploratory surface hypotheses had one match and remain separate.
- **Repeated-stem consistency** was assessable only when the exact surface stem was isolated by a token boundary across forms. None of four eligible repeated-stem groups isolated it across every form. The audit therefore reports zero fully isolatable groups and leaves consistency conditional on isolation unmeasured.
- **Sandhi and gemination** had five explicitly unscored forms, including `മരക്കൊമ്പ്` and `വീട്ടിൽ`. These cases need adjudicated surface alignment. A single-piece common form can be useful for compression while missing a manual morpheme boundary.

These scores characterize authored probes, not a gold Malayalam benchmark. A first thesis dataset should separate concatenative forms, auditable surface allomorphs, and ambiguous or non-invertible sandhi. Preserve alternative analyses and reviewer agreement alongside boundaries.

## 6. Hugging Face compatibility

The original JSON loaded locally through both Tokenizers and Transformers `AutoTokenizer` with remote code disabled. There were zero vocabulary-ID mismatches. Batch IDs and decoded strings matched, an in-memory serialization reload produced identical batch IDs, and the standard causal-LM collator produced a 3×10 input with padding labels masked and unpadded labels equal to input IDs. The chat template rendered the assistant generation prompt. The compatibility checks do not override the two measured inline-literal fidelity failures. [HF compatibility results](measurement-v3/hf-compatibility.json)

## 7. Contribution and experiment priorities

| Candidate idea | What prior work establishes | Thesis decision |
|---|---|---|
| Script normalization and pretokenization | Indic studies already evaluate script normalization, algorithm and vocabulary size, and vocabulary construction strategy | Treat reliable transport and explicit normalization policy as correctness work shared by comparable arms |
| Grapheme or akshara-aware merges | GPE and recent Indic grapheme-tokenizer studies establish grapheme units, including experiments covering Malayalam | Include a grapheme unit comparator. Test whether a tailored Malayalam rule adds value over EGC on reviewed joiner, chillu and conjunct cases |
| Morphology-constrained BPE | MorphBPE constrains training merges. MorphTok evaluates morphology, sandhi and script-aware constrained BPE for Hindi and Marathi | Prioritize a Malayalam-specific **surface-aligned, confidence-weighted** comparison with hard constraints and abstention. Establish novelty against up-to-date prior work before making a claim |
| Picky vocabulary refinement | PickyBPE resolves undertrained intermediate tokens by refining merges | Defer. Reconsider only if representative held-out counts show consequential allocation waste and preservation of all merge dependencies is established |
| Morphological consistency as the objective | MorphScore work reports that morphology alignment explains limited variance in downstream model performance | Use it as a linguistic diagnostic alongside fertility, lossless fidelity, and matched-budget BPB |

Sources: [Indic tokenizer comparison](https://aclanthology.org/2026.findings-acl.1632/) · [Grapheme Pair Encoding](https://aclanthology.org/2025.coling-main.400/) · [Indic grapheme tokenizer evaluation](https://aclanthology.org/2026.findings-emnlp.1093/) · [MorphBPE](https://aclanthology.org/2026.findings-acl.2068/) · [MorphTok](https://arxiv.org/abs/2504.10335) · [PickyBPE](https://aclanthology.org/2024.emnlp-main.925/) · [MorphScore multilingual evaluation](https://arxiv.org/abs/2507.06378)

### Proposed controlled matrix

| Arm | Change being tested | Role |
|---|---|---|
| Historical 48K BPE | Frozen published artifact | Descriptive reference, including backward compatibility and archived results |
| Matched 48K BPE control | Same BPE recipe under the fixed, documented fidelity policy | Required causal reference |
| EGC control | Use extended-grapheme units or demonstrably prevent in-EGC boundaries, with a declared fallback | Tests whether a standard Unicode reference is sufficient |
| Malayalam orthography variant | Malayalam-aware protection for reviewed conjunct/joiner cases versus the EGC control | Isolates custom orthographic value |
| Confidence-weighted morphology variant | Add penalties only for defensibly aligned surface boundaries. Compare soft weights against a hard/high-confidence subset on a small pilot | Tests whether Malayalam morphology improves alignment without an unacceptable efficiency or BPB cost |

Optional external comparator after the control is stable: a 48K SentencePiece Unigram run on identical cleaned corpus bytes. It helps distinguish a BPE-specific trade-off from a broadly useful linguistic metric. Defer PickyBPE and joint morphology/orthography combinations unless single-factor evidence supports them.

1. Recover the original eval shards through a bounded export from the DGX if they still exist. Record exact names, sizes, document count and SHA-256. If unavailable, publish a frozen replacement benchmark with source, revision, license, deduplication and split manifests. Label it distinctly from the archived benchmark.
2. Establish immutable controls and an explicit fidelity contract, including reversible literal-marker behavior and handling of special tokens in plain text versus chat. Add reported fixture failures to regression tests and validate chillu/conjunct ambiguities with Malayalam reviewers.
3. Prepare a small independently adjudicated surface-boundary set. Record alternative analyses, abstentions, reviewer agreement, and stratified results. Ensure held-out inflections or lexical families cannot leak into configuration selection.
4. Pilot the arms as independently replaceable plugins. Hold vocabulary budget, corpus order, text byte budgets, seeds, and special IDs fixed. Report raw and normalized round-trip, exact-offset akshara and EGC violations, morphology coverage and alignment, compression, training time, and checkpoint recovery.
5. Run expensive training only after the pilot decision and explicit approval. Use a verified eligible DGX GPU and keep physical GPU 7 excluded. Use equal training and evaluation bytes, paired seeds and paired BPB differences with uncertainty. Record embedding/non-embedding parameters and the decoded text span per context. Reject an improvement claim if only a diagnostic metric improves.

## 8. Engineering verification and reproducibility

- Stage A was committed locally as `b8e531709bf2b78280cbe8b5a2155d688e943cb7`. The read-only GitHub branch check returned `d75f72dc4ac5e4c3b6b73ad36e14a5f9c126606b`. The authorized push could not authenticate noninteractively. The previous 181-test suite concerns the Stage A code. The added boundary plugin passed its exact-offset unit test and its plugin contract test (2 passed).
- Stage B passed 15 live checks on the private `Akzhai/mltok-storage-test` repository. Two snapshots with six artifacts each were read back by SHA-256 and restored from their pinned revisions. Upload interruption, retries, interrupted downloads, compatibility checks, and fallback were injected locally. No published data was corrupted or removed. [Live Hub verification](hub-verification.json)
- Authoritative new results are in [measurement-v3](measurement-v3/). [configuration.json](measurement-v3/configuration.json) records the fixture and code fingerprints, versions, and definitions. [COMPLETE.json](measurement-v3/COMPLETE.json) lists sizes and hashes for the completed run.
- To rerun without changing the original baseline or completed result directory, use a new output subdirectory. Morphology measurements used the local, hash-checked PyPI wheels for Mlmorph 1.4.3 and SFST 1.10.0. The tokenizer-only checks also run without the optional analyzer. [Mlmorph release](https://pypi.org/project/mlmorph/1.4.3/) · [SFST release](https://pypi.org/project/sfst/1.10.0/)

```powershell
$env:PYTHONPATH=(Resolve-Path 'audit/48k-baseline/.deps/installed').Path
$env:PYTHONUTF8='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python audit/48k-baseline/diagnose.py --output audit/48k-baseline/reproduction-01
```

No production tokenizer repository was modified. There was no tokenizer retraining, proxy training on thesis data, GPU use, or full-corpus download. The live backup checks verify integrity and recovery against observed Hub operations. They remain dependent on trusted repository writers and the Hub retaining the pinned revisions.
