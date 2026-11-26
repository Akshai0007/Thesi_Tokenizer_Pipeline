# DGX batch guide for the T5 and T8 downstream ablation

The batch runs two matched pairs at seed 17. The trainer, optimizer, model, tokenizers, normalizer, data and canonical BPB scorer come from the frozen inputs and operational commit `9b872ffb5a66bdf4d23df58acebbba7e788915d8`. The queue manifest SHA-256 is `d0bfb6b3994cff4ca2c0159718e0a29da91e1f8aa029f5fa3d7169541905a5b7`. The original 25-file T5/T8 input manifest remains `4266825c1132cb44555313b0b68cd489bbfe55566a508be2a8ed606a0bad7750`.

| Order | Arm and protocol | Canonical training bytes | Predicted targets | Updates | Exposure plan SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | T5 fixed source bytes | 49,998,152 | 5,383,363 | 325 | `e7ca4eedbbd72817d7a31d8335004dcd4ab11709a53b58f8b2b70cf745f6846b` |
| 2 | T8 fixed source bytes | 49,998,152 | 5,386,510 | 325 | `87b197f9954b1d2f10f65a6e2d3c660a5495426c51a7ff297fc49e7a6c7a2732` |
| 3 | T5 fixed training tokens | 46,504,717 | 5,000,000 | 302 | `d4c7ce984c0c07dd1a5955408de475e9c2574017b92ce349629ba31ba794b7bd` |
| 4 | T8 fixed training tokens | 46,478,001 | 5,000,000 | 302 | `3acd4b0575113a6428115bc2cb98786966ae4dd12e1ba717851383fabf4cf11a` |

Each job uses the 38,799,936-parameter GPT, context 512 and all 186 ordered development documents. Fixed source bytes selects the same 10,355 complete documents under a 50,000,000-byte budget. Fixed training tokens uses the frozen token budget and tokenizer-specific prefix endpoint. The old fixed-source jobs keep their original output paths. The fixed-token jobs use separate subdirectories under the new batch root.

## GitHub deployment

Retrieve `notebooks/mom_bpe_dgx_batch.ipynb` from the exact published deployment commit on `research/mom-bpe-dgx-resources`. Verify the notebook SHA-256 from the publication report before opening it. Set `MOM_BPE_DEPLOY_REVISION` to that full 40-character commit in the kernel environment. Cell 2 retrieves that commit and the separate immutable operational commit directly from GitHub. Cell 3 checks the GitHub revision and byte hashes of all seven deployment files, including the notebook, then records the verified runtime. Keep the verified checkout pristine. The notebook can be edited and run from the separate copy fetched to JupyterLab.

Reuse `/jupyter/socse.user63/mom-bpe-private-recovery`, a user-owned directory with mode 0700. The existing isolated Python 3.12.0 environment must contain the pinned scientific dependencies and `huggingface_hub`. Use an approved credential provider, runtime environment injection, or the masked prompt in Cell 4.

## Pinned artifact inventory

| Required files | Trusted origin |
| --- | --- |
| Eight seed-17 M48/F-space result and identity JSON files | Private model repo `Akzhai/mom-bpe-dgx-results-private`, revision `60e3d255c7afd86ec568e5a728f98a8c6e433219` |
| Twenty-five T5/T8 tokenizer, cache, exposure ledger and corpus identity files | Exact original individual files identified by `t5_t8_inputs_v1.json`. The inspected Hugging Face receipts do not contain these hashes. |

The private thesis evidence snapshots in `Akzhai/mltok-storage-test` at `9db79dbd64572bab10c44c7ec8660d8dc103b443` and `78737cb615e81ba24878f6e7ae7ac02ca6d36d52` do not supply the 25 required training files. Reuse already staged originals or arrange approved individual-file staging into `t5-t8-frozen-runs-v1/` following the exact relative paths, sizes and SHA-256 hashes in the frozen manifest. The fixed-training-token plans are recomputed from those frozen caches and checked against the reviewed plan pins.

Cell 5 writes `artifact-inventory.json` and immutable per-attempt restoration receipts under `individual-restorations/` in the batch directory. Each entry includes the local destination, expected size and SHA-256, pinned source if verified, and its status. Verified local files are reused. The eight historical files are placed in the separate `t5-t8-seed17-reference-v1/` tree. Temporary download state is kept under `downloads/` for retry and resume. A conflicting existing file is preserved. If the exact remote member is available, its verified replacement is saved separately and the cell reports the conflict. Resolve the conflict within approved staging storage before resuming. Missing training files without a verified Hub source block training.

## Notebook order

| Cell | Operation |
| --- | --- |
| 1 | Check the pinned Python environment and Hub client. |
| 2 | Retrieve the exact deployment and operational commits from GitHub. |
| 3 | Verify the source commit, notebook and companion hashes. |
| 4 | Authenticate to the pinned private Hub reference revision. |
| 5 | Restore individual files and review the per-file receipt. |
| 6 | Verify all frozen input and historical reference hashes. |
| 7 | Verify the four exposure plans. |
| 8 | Check CPU readiness and private scratch. |
| 9 | Inspect GPU resources and storage. |
| 10 | Run the approved A100 diagnostic. |
| 11 | Run T5 then T8 under fixed source bytes. |
| 12 | Verify both first-pair results and checkpoints and write their comparison. |
| 13 | Run T5 then T8 under fixed training tokens. |
| 14 | Require four verified results and final checkpoints. |
| 15 | Generate the paired scientific analysis. |
| 16 | Optionally back up verified outputs to a separately approved private Hub repository. |
| 17 | Write the immutable final session report. |

After institutional allocation and explicit execution approval, set a fresh `MOM_BPE_BATCH_SESSION` label. In Cell 10 set `smoke_approved = True` and `allocated = True`, enter the full physical A100 UUID, and keep `scope = "all"` for both pairs. Default admission requires 8192 MiB free, at most 5% utilization and exclusive use. Separate institutional shared-use approval permits the reviewed 5120 MiB and 10% profile. Both profiles reject physical GPU 7 and require verifiable compute-process information. The diagnostic uses zero training updates.

Enable Cell 11 after the smoke succeeds. Enable Cell 13 separately after Cell 12 verifies both first-pair results. The controller runs one job at a time and stops on a failed check. Resume by rerunning the same training cell on the original GPU and runtime. A later allocation needs a fresh smoke receipt. Scientific comparisons continue to be seed-17 development evidence.

For Cell 16 supply the approved private model repository and explicitly enable `backup_approved`. The default upload set is verified result and identity JSON files. Final checkpoints require the separate `checkpoint_backup_approved` switch. The stage checks destination privacy, uses a parent-pinned commit, and independently re-downloads and hashes each approved member at the resulting revision. Existing conflicting remote files stop backup. Hub backup receipts live in `hub-backup-receipts/`. Reversible caches and corpus data remain in their approved storage.

## Recovery and capacity

The batch state root is `runs/dp/t5-t8-seed17-batch-v1/`. Numbered state receipts progress through input verification, plan verification, readiness, execution, completion, result verification and backup. An interrupted worker releases the OS lock. The next controller invocation records the interruption and resumes from a compatible retained generation on the original GPU UUID and runtime. A job is skipped only after checking its identity, evaluation and final checkpoint. A missing or damaged final artifact of a previously verified job requires exact recovery from approved storage. Earlier corrupt or interrupted generations remain available for investigation. Elapsed training time and GPU allocator peaks come from per-attempt telemetry. An abrupt crash can leave the last measurement interval incomplete.

Allow **8 GiB free for the first pair** or **15 GiB for all four**, plus checkout and input storage. Seven generations per job at about 466 MB each imply 28 retained generations for all four. The estimated untouched four-job requirement is 13,999,434,240 bytes including two write/restore allowances. The controller enforces the higher 15 GiB floor before an untouched four-job batch. The selected input members occupy 109,430,168 bytes and the archive occupies about 91 MB. Plan backup capacity separately. The 8 GiB VRAM admission floor is a policy threshold. Actual peak VRAM and runtime are measured on the approved device.

An alternative selection or ordering of the reviewed jobs can use `--manifest FILE --manifest-sha SHA256` with the controller CLI. The controller verifies exact membership against the approved job definitions, uses a separate journal namespace and preserves the original job output identities. New tokenizer, model, seed or input identities require reviewed plans and the corresponding adapter. The shipped notebook pins only this four-job queue.

## Verification

The frozen batch and operational trainer retain their original pins. The deployment notebook loads the seven reviewed source files from an exact GitHub commit and verifies their byte identities. The private Hub inventory separates the eight pinned historical JSONs from the 25 frozen local training artifacts.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q -ra -p no:cacheprovider
```
