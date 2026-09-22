# Project working agreements

- User preference (2026-09-20): deliver and deploy code through Git commits and the configured origin. Do not create or transfer source tar/zip packages for deployment. Data, checkpoints and experiment result archives remain separate and ignored by Git.
- Preserve server/local uncommitted changes. Do not force-push, reset, clean or automatically stash to make deployment succeed. Use fast-forward updates only and keep a running experiment on its original code/configuration.
- Current combined entry: `docs/all_preparation.md` (closure-v4 then Torch comparison). Detailed bounded closure protocol: `docs/preparation_closure.md`. Historical full reproduction and source-package instructions are not the default workflow.
