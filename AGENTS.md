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
