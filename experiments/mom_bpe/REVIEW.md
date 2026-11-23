# Independent MOM-BPE review

The initial workspace contained 21 prototype implementation, test and documentation files plus the line-ending metadata and the archived diagnostic JSON. All 23 files were reviewed. The three published baseline files still match the pinned manifest at revision `cff0d31c84b598436087f21cc8183021993831e1`. The baseline tokenizer JSON SHA-256 remains `2f4b7e8b0706007f7ab3f0b8b2cc1b169094072c334585dabc4081d455e69ff5`. The original pipeline configurations, previous commits and diagnostic evidence were preserved. [Baseline manifest](../../audit/48k-baseline/baseline-artifacts.json) · [Original diagnostic evidence](diagnostic-results-v1.json)

## Findings and corrections

| Area | Finding |
| --- | --- |
| Legacy compatibility | Existing plugins retain their registration, cache fingerprints, default configuration and artifact labels. The runner's compact label is opt-in for MOM-BPE and hashes the full algorithm specification. This avoids invalid Windows characters and excessive path lengths when a reviewed alignment file is passed by absolute path. |
| Normalization and orthography | The conservative rules reproduce their Python and serialized Hugging Face outputs on the regression cases and are idempotent on those cases. Legacy chillu mapping stays context limited. The review added a regression and grammar fix for Malayalam combining anusvara above (U+0D00), which previously detached from its base. Akshara and extended grapheme clusters continue to have separate definitions. |
| Script extensions | A shared leading letter could previously be claimed by the first configured script. A run now requires a primary-script letter. Shared prefixes follow the next compatible primary script and shared suffixes remain with the current run. Both Japanese script orders and the mixed Malayalam/Latin cases are tested. |
| Weighted merges | The prototype applied the minimum pair count to the weighted score. The raw count now determines candidate eligibility in MOM-BPE. Confidence changes the ranking of the eligible pairs. An enabled run also fails clearly if sampling or alphabet pruning leaves none of its projected sites between encodable initial units. The legacy scoring path retains its behavior. |
| Rank replay and disabled control | The independent recount and rank-replay regression checks emitted merges. The diagnostic notebook verifies identical initial symbols, ranks and IDs for T2 and T3-off. The morphology file and its hash are identical for T3-off and T3. |
| Review and freeze gates | Whitespace and case aliases can no longer authorize the same reviewer twice. Duplicate held-out text after canonicalization is rejected even if its IDs differ. Planning checks that the manifest bytes are the same bytes verified by preflight. Reviewer independence, rights and near-duplicate adjudication still require external confirmation. |
| Encoder and backup | A tiny CPU integration test exercises the normal runner, a weighted alignment passed by absolute path, the custom encoder, one proxy update, and an offline revisioned Hugging Face store. Restore followed by runner reload checks token IDs and checkpoint bytes and reuses the completed proxy result. The custom artifact requires the project encoder. Stock `AutoTokenizer` does not load that artifact. |

The Unicode review used the assigned character chart for the historical combining mark and the Unicode script property guidance for neutral characters with script extensions. [Malayalam character chart](https://www.unicode.org/Public/UCD/latest/charts/nameslist/0d00/) · [Unicode Script Property](https://www.unicode.org/reports/tr24/)

## Why the diagnostic fell back to bytes

The independent count used the exact archived training and evaluation fixtures with their declared preprocessing. The 384-slot diagnostic reserves seven control slots and 256 byte slots, leaving 121 lexical slots and a 60-unit initial-alphabet cap. All seen initial units fit under that cap.

| Arm | Seen training unit types | Held-out unit types | Fallback byte IDs | From units absent in training | From seen units excluded by the cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0-matched codepoints | 44 | 153 | 1,631 | 1,631 | 0 |
| T1 graphemes | 49 | 394 | 6,092 | 6,092 | 0 |
| T2 aksharas | 49 | 394 | 6,092 | 6,092 | 0 |

Increasing the vocabulary with these same training fixtures would still leave every unseen cluster absent. The diagnostic fallback counts are explained by the tiny corpus and the declared whole-unit byte fallback, which preserves the bytes of each unseen unit. There was no unexplained fallback in these probes. This is not an estimate for a 48K trained model. Report unit coverage and byte-fallback incidence on representative held-out data before comparing boundary rates. [Measured attribution](review-results-v1.json)

## Mlmorph and manual surface review

The hash-pinned local Mlmorph 1.4.3 and SFST 1.10.0 installation loaded its 18,619,083-byte compiled transducer. The analyzer returns ranked tagged lemma readings. Its parser provides roots and tags, while its generator reproduces surface strings. Neither provides character offsets. On the original five forms it returned four nonempty reading lists and one empty list. Each nonempty list contained additional readings absent from the prototype's mapping, so abstention was expected. Selecting the lowest analyzer rank would change the research rule and could force a false analysis.

The source-assisted desk review examined 12 real Malayalam forms in explicit contexts. It used transparent plurals, a surface stem alternation, y before the locative, a fused plural–locative junction, and the original ambiguous or missing readings. These are judgments from this review and are separate from the synthetic examples and the independently adjudicated research panel.

| Outcome | Count |
| --- | ---: |
| Forms with successful analyzer parses | 10 of 12 |
| Candidate readings parsed | 18 |
| Selected readings generating the exact surface | 6 |
| Defensible character-offset surface alignments | 5 |
| Correct surface abstentions | 7 |
| Incorrect alignments emitted on the desk set | 0 |
| New forms authorized for training | 0 |

The five checked splits were `കിളി|കൾ`, `കുട|കൾ`, `കസേര|കൾ`, `വീടു|കൾ` and `പൂച്ച|യിൽ`. The selected reading for `കടകളിൽ` regenerates that form, but a full three-part projection would cut the written unit `ളി` and abstains. The old synthetic division `വീ|ട്ടിൽ` is unsuitable as a morphology label because it assigns written stem material to the locative. The exact geometric checker alone cannot detect that linguistic mistake when given a supplied split. Reviewer authorization remains mandatory, so none of the desk judgments produced training weights. Confidence values are fixed-point judgments, and the analyzer's ranks have no probability calibration. [Desk judgments](morphology-desk-review.json) · [Observed readings](review-results-v1.json) · [Mlmorph system paper](https://aclanthology.org/W19-6801/) · [Malayalam grammar](https://www.languageinindia.com/nov2012/ravisankarmalayalamgrammar.pdf) · [Malayalam inflection study](https://www.mdpi.com/2226-471X/8/1/29)

## Reproducibility and experiment readiness

The read-only probe uses the previous local wheel environment and writes a new immutable result. Wheel hashes were checked against the existing lock. The notebook cells ran in separate processes. The saved evidence fingerprints the compiled transducer and manual input. The reviewed diagnostic output has a separate identity and includes the saved model fingerprints and zero-weight equivalence result. [Reviewed CPU diagnostics](diagnostic-results-v2.json)

The targeted offline integration suite passed 234 tests with zero failures and zero skips. One existing SentencePiece contract test emitted two protobuf deprecation warnings. The subsequent six-test freeze rerun also passed and covered the added regression for manifest replacement after verification.

```powershell
$env:PYTHONPATH=(Resolve-Path 'audit/48k-baseline/.deps/installed').Path
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
python experiments/mom_bpe/review_probe.py --out "$env:TEMP/mom-bpe-independent-review.json"
```

The ablation configuration preserves the public T0 as a descriptive reference, matched codepoint control, T1, T2, T3-off and T3, with 48K occupied slots required for new primary models. The first controlled experiment after the data freeze should compare the matched codepoint, grapheme and akshara arms on a bounded, identical source-byte development slice. Check 48K occupancy, fallback by domain, Unicode fidelity, reference-break incidence and coverage before adding approved morphology. Then verify zero-weight equivalence again and run the paired T3-off/T3 comparison.

There is still no verified frozen dataset or independently adjudicated morphology panel. The runner proxy smoke test verifies interchange and recovery. The existing proxy packs a continuous stream and counts raw bytes, so it cannot establish the planned per-document canonical-byte BPB outcome. Full-vocabulary training throughput and interrupted BPE fitting remain unverified. No actual corpus experiment, live Hugging Face backup, GPU operation or downstream BPB comparison was performed in this review. [Ablation configuration](arms.yaml) · [Freeze protocol](../../audit/48k-baseline/evaluation-protocol-v1.json)
