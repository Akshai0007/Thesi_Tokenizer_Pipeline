# Malayalam tokenizer development

1. Preserve the independent plugin architecture and its public interfaces. Keep plugins replaceable through configuration and free of imports from other plugins.
2. Keep changes focused and avoid unnecessary changes to working components.
3. Keep Jupyter cells isolated and independently rerunnable. Give each cell explicit inputs and prerequisites, and persist stage outputs so reruns do not depend on hidden kernel state.
4. Save completed stages as immutable artifacts with stable identities. Preserve existing experiment outputs and use a new artifact identity for changed inputs.
5. Support interrupted training and checkpoint-based resumption. Validate checkpoint compatibility before resuming.
6. Support private Hugging Face repositories for experiment storage.
7. Obtain credentials securely at runtime through an approved secret provider or environment injection. Never hardcode tokens.
8. Never log or commit credentials, including through notebook output, command arguments, or serialized configuration.
9. Never delete checkpoints until replacement artifacts and their backups have been verified.
10. Preserve original tokenizer baselines and their recorded results.
11. Maintain reproducibility with fingerprints for resolved configuration, datasets, plugins, and artifacts. Record the identities used for each evaluation.
12. Never use physical GPU 7 on the university DGX A100. Verify physical GPU assignments before applying process-local device masks.
13. Check GPU availability before starting DGX experiments.
14. Run CPU diagnostics separately from GPU training.
15. Obtain explicit approval before destructive operations, remote deletions, or publishing changes.
16. Favor scientific correctness, reproducibility, and experimental quality over speed.

## Branch & Worktree Discipline

17. Perform all development, auditing, and testing on a dedicated research branch or isolated Git worktree. Never make direct edits or commits on `main`.
18. Never run `git reset --hard`, `git clean`, or force-push (`git push --force` / `--force-with-lease`) without explicit user approval.

## Artifact & Frozen Config Protection

19. Never delete, overwrite, or clean untracked local artifacts (including `runs/`, raw or cleaned corpora, backup archives, and checkpoint directories).
20. Never modify frozen experiment configurations, manifests, pin files, or pinned deployment artifacts (`experiments/mom_bpe/optimization-arms.yaml`, `experiments/mom_bpe/t5_t8_inputs_v1.json`, `experiments/mom_bpe/t5_t8_batch_v1.json`, `experiments/mom_bpe/document-proxy.yaml`, `experiments/mom_bpe/document_proxy.py`, `mltok/lib/*`, or `DEPLOYMENT_FILES`) without explicit user approval.

## Strict DGX & Hugging Face Security Rules

21. Never bypass `experiments/mom_bpe/dgx_resources.py` GPU admission guards, memory/utilization limits, or institutional allocation checks, and never run unallocated GPU jobs.
22. Enforce least-privilege Hugging Face token scope (use read-only tokens for restoration and verification; use write tokens only for explicitly approved private backups). Never print, echo, log, cache in git config, or commit credentials, and never use production HF repositories for connectivity or smoke tests.

## Required Task Handoff Report

23. End every implementation or audit task with a structured 10-point report: (1) Task description, (2) Files inspected, (3) Files modified, (4) Reason, (5) Tests executed, (6) Test results, (7) Generated artifacts, (8) Git commit, (9) Remaining issues, and (10) Recommended next task.
