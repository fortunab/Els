# Migration notes

The original repository contained several standalone Python files exported from notebooks. This restructure converts them into importable modules and preserves the main experimental ideas.

## Changes made

1. Renamed files to descriptive snake_case module names.
2. Moved reusable code into `src/hydra_aixia/`.
3. Added `examples/` scripts for direct execution.
4. Added `requirements.txt`, `pyproject.toml`, `.gitignore`, and `.env.example`.
5. Removed hard-coded API credentials and replaced them with `OPENAI_API_KEY` environment-variable loading.
6. Added basic tests to validate package imports.

## Recommended next steps

- Add dataset-specific documentation.
- Add experiment result tables.
- Add model checkpoints only through releases or external storage, not Git.
- Add CI tests after stabilizing dependencies. 
