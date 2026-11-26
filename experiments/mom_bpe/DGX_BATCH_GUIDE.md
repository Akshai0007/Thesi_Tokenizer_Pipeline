# DGX batch guide for the T5 and T8 downstream ablation

The batch runs two matched pairs at seed 17. The trainer, optimizer, model, tokenizers, normalizer, data and canonical BPB scorer come from the frozen inputs and operational commit `9b872ffb5a66bdf4d23df58acebbba7e788915d8`. The queue manifest SHA-256 is `d0bfb6b3994cff4ca2c0159718e0a29da91e1f8aa029f5fa3d7169541905a5b7`. The original 25-file T5/T8 input manifest remains `4266825c1132cb44555313b0b68cd489bbfe55566a508be2a8ed606a0bad7750`.

| Order | Arm and protocol | Canonical training bytes | Predicted targets | Updates | Exposure plan SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | T5 fixed source bytes | 49,998,152 | 5,383,363 | 325 | `e7ca4eedbbd72817d7a31d8335004dcd4ab11709a53b58f8b2b70cf745f6846b` |
| 2 | T8 fixed source bytes | 49,998,152 | 5,386,510 | 325 | `87b197f9954b1d2f10f65a6e2d3c660a5495426c51a7ff297fc49e7a6c7a2732` |
| 3 | T5 fixed training tokens | 46,504,717 | 5,000,000 | 302 | `d4c7ce984c0c07dd1a5955408de475e9c2574017b92ce349629ba31ba794b7bd` |
| 4 | T8 fixed training tokens | 46,478,001 | 5,000,000 | 302 | `3acd4b0575113a6428115bc2cb98786966ae4dd12e1ba717851383fabf4cf11a` |

Each job uses the 38,799,936-parameter GPT, context 512 and all 186 ordered development documents. Fixed source bytes selects the same 10,355 complete documents under a 50,000,000-byte budget. Fixed training tokens uses the frozen token budget and tokenizer-specific prefix endpoint. The old fixed-source jobs keep their original output paths. The fixed-token jobs use separate subdirectories under the new batch root.

## Stage on DGX

Extract the source-only deployment archive in an approved private JupyterLab folder. Open `notebooks/mom_bpe_dgx_batch.ipynb` and set the kernel working directory to the extracted `experiments/mom_bpe` directory. The notebook checks the controller, queue manifest, and the two frozen input companions by SHA-256.

Reuse `/jupyter/socse.user63/mom-bpe-private-recovery`, a user-owned directory with mode 0700. Stage the eight exact historical reference JSONs under `t5-t8-seed17-reference-v1/seed17/{M48,F-space}/{fixed_source_bytes,fixed_training_tokens}/{identity,result}.json`. The authenticated local copy is under `experiments/mom_bpe/runs/dp/postrun-v1/60e3d255c7afd86ec568e5a728f98a8c6e433219/`. The historical checkpoints are referenced by the original JSONs. The comparisons use those JSONs and the frozen data fingerprints.

| Cell | Operation |
| --- | --- |
| 1 | Verify or recover the exact operational Git checkout. A fresh fetch is staged in a private sibling directory before verification and promotion. |
| 2 | Verify the existing pinned Python 3.12.0 environment and dependencies. Restore the approved environment if it was removed. |
| 3 | Optionally restore 25 exact members from Package B into `t5-t8-frozen-runs-v1/`. Reuse that tree if it already verifies. |
| 4 | Verify SHA-256 for frozen members and all eight historical JSONs. First verification checks both tokenizer fits and the frozen development identity. |
| 5 | Verify the four exposure plans, including the fixed-token plans. |
| 6 | Verify CPU readiness and the full-batch private scratch floor. |
| 7 | Inspect physical GPU UUIDs, memory, utilization, compute processes and scratch. |
| 8 | Run the approved A100 diagnostic and record the session authorization. |
| 9 | Run T5 then T8 under fixed source bytes. |
| 10 | Verify both first-pair results and checkpoints and write the first comparison. |
| 11 | Run T5 then T8 under fixed training tokens. This requires full-queue approval and the verified first-pair receipt. |
| 12 | Require all four verified results and final checkpoints. |
| 13 | Write the per-protocol scientific analysis and paired document comparisons. |
| 14 | Optionally back up verified artifacts. |
| 15 | Write the immutable final session report. |

Package B is the existing 91,141,457-byte archive with SHA-256 `fcc1957a3456774ccbc00c08e18738c7b142250d22150bbbf405206616156364`. Cell 3 is opt-in. Set `restore_approved = True` only when using the verified archive at the configured path. A successful receipt avoids reopening the archive while all selected files exist. If a staged member is missing, rerun Cell 3 against that same archive and then verify inputs. Conflicting bytes block the stage. The fixed-token plans reuse the same 25 members.

To activate the GPU after institutional allocation and explicit execution approval, set a fresh kernel label such as `%env MOM_BPE_BATCH_SESSION=dgx-batch-001`. In Cell 8, set `smoke_approved = True` and `allocated = True`, enter the approved full physical A100 UUID and select `scope = "all"` for both pairs. The diagnostic uses zero training updates. Default admission is 8192 MiB free, at most 5% utilization and exclusive use. With separate institutional shared-use approval, `shared_approved = True` selects the reviewed 5120 MiB and 10% profile. Both profiles reject physical GPU 7 and require verifiable compute-process information. A failed diagnostic attempt blocks reuse of an earlier approval under that label.

Set `enable_training = True` in Cell 9 after Cell 8 succeeds. Cell 10 must verify and compare both first-pair results before Cell 11 can start. Enable Cell 11 separately while the allocation is still valid. To resume, rerun the same training cell on the original GPU and runtime. Keep the same approval receipt while its allocation remains valid. A later allocation requires a fresh diagnostic. The worker admits the approved device before each job and the pinned trainer checks admission again immediately before training. On a failure the controller stops the queue and retains logs, checkpoints and receipts.

Cell 13 separates tokenization compression and Malayalam orthographic measurements from downstream BPB. It reads tokenizer-only metrics from the frozen validation artifacts and reports canonical bytes per scored prediction alongside the paired language-model results. Per-document rows and 10,000 paired bootstrap draws use the same ordered development documents. The historical checkpoint references come from authenticated result JSONs. Historical GPU peaks and elapsed times are unavailable from those JSONs. This seed-17 evaluation leaves independent held-out quality and training-seed uncertainty open.

In Cell 14 supply the institutionally approved private backup mount and set `backup_approved = True`. Choose `kind = "all"` or a category such as `checkpoints`, `results`, `identities`, `configuration`, `telemetry`, `manifests`, `receipts`, `analysis` or `logs`. The destination is a mounted directory outside the recovery root. Every copied file is reopened and independently hashed at the destination. The content-addressed backup receipt is the exact revision and includes the verified member manifest. A full verified backup marks the available jobs `BACKED_UP`. Category-specific backups have separate receipts. Cell 15 reports verified and pending jobs, metrics, backups, scratch, current-process observations and the requirement for separate cleanup approval. Review backup coverage and ownership before any proposed deletion.

## Recovery and capacity

The batch state root is `runs/dp/t5-t8-seed17-batch-v1/`. Numbered state receipts progress through input verification, plan verification, readiness, execution, completion, result verification and backup. An interrupted worker releases the OS lock. The next controller invocation records the interruption and resumes from a compatible retained generation on the original GPU UUID and runtime. A job is skipped only after checking its identity, evaluation and final checkpoint. A missing or damaged final artifact of a previously verified job requires exact recovery from approved storage. Earlier corrupt or interrupted generations remain available for investigation. Elapsed training time and GPU allocator peaks come from per-attempt telemetry. An abrupt crash can leave the last measurement interval incomplete.

Allow **8 GiB free for the first pair** or **15 GiB for all four**, plus checkout and input storage. Seven generations per job at about 466 MB each imply 28 retained generations for all four. The estimated untouched four-job requirement is 13,999,434,240 bytes including two write/restore allowances. The controller enforces the higher 15 GiB floor before an untouched four-job batch. The selected input members occupy 109,430,168 bytes and the archive occupies about 91 MB. Plan backup capacity separately. The 8 GiB VRAM admission floor is a policy threshold. Actual peak VRAM and runtime are measured on the approved device.

An alternative selection or ordering of the reviewed jobs can use `--manifest FILE --manifest-sha SHA256` with the controller CLI. The controller verifies exact membership against the approved job definitions, uses a separate journal namespace and preserves the original job output identities. New tokenizer, model, seed or input identities require reviewed plans and the corresponding adapter. The shipped notebook pins only this four-job queue.

## Offline verification

The clean pinned checkout restored exactly 25 files from Package B, authenticated all eight historical reference JSONs, verified both protocol-specific plans for each arm and rejected unapproved smoke, execution and backup. The local verification receipts are under `runs/dp/t5-t8-batch-verification-v1/`. The four-plan receipt has file SHA-256 `fdc09d63e110ede1fb27a08682e4566826e36e902bd244799aa1bbc9c4f445e2`. The notebook and controller remain local deliverables. DGX storage, checkpoint availability and institutional GPU allocation require on-host confirmation.

The focused Windows CPU suite passed 101 deployment, recovery, resource and integrity tests. It used `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` and `PYTHONDONTWRITEBYTECODE=1`.

```powershell
python -m pytest -q -ra -p no:cacheprovider tests/test_dgx_batch.py tests/test_dgx_resources.py tests/test_document_proxy.py tests/test_dgx_handoff.py tests/test_dgx_readiness.py tests/test_notebook.py tests/test_dgx_session_upgrade.py
```
