# Repository Guidelines

This project will grow into a federated LoRA experimentation toolkit. Keep contributions incremental, reproducible, and documented; update this guide whenever the workflow changes.

## Project Structure & Module Organization
- `src/` houses core training code: client/server logic, dataset adapters, and LoRA helpers.
- `configs/` stores YAML or JSON run definitions; keep secrets in environment files instead.
- `scripts/` holds orchestration utilities (cluster launchers, data prep).
- `tests/` mirrors `src/` modules and uses identical package names for clarity.
- Reserve `notebooks/` for exploratory work and factor durable logic back into `src/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated development environment.
- `pip install -r requirements.txt` synchronizes dependencies; regenerate the file when versions change.
- `pytest` executes automated tests; add `-k name` for targeted subsets.
- `python src/main.py --config configs/default.yaml` is the expected entry point for local experiments; ensure sample configs stay runnable.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, comprehensive type hints, and descriptive class names (`ClientTrainer`, `AggregatorService`). Format via `black .`, lint with `ruff`, and sort imports using `isort`. Keep source files ASCII-only; never commit Chinese text or other non-ASCII characters in code. Modules stay snake_case, constants SHOUT_CASE, and configuration files use kebab-case identifiers.

## Testing Guidelines
Place PyTest modules in `tests/` named `test_<module>.py`, with functions like `test_<behavior>()`. Cover both local and federated execution paths using fixtures for synthetic clients. Maintain >=85% statement coverage; explain any deltas in pull requests. Prefer event-driven waits over sleeps to keep tests stable.

## Commit & Pull Request Guidelines
Adopt Conventional Commits (`feat:`, `fix:`, `docs:`) written in the imperative mood. Reference issue IDs (e.g., `FED-17`) and document config changes or migrations. Pull requests must include a summary, validation evidence (`pytest` output or sample run logs), and screenshots when behavior changes are user-visible.

## Security & Configuration Tips
Keep API keys and federation endpoints in `.env` files; add `.env.example` entries so others can bootstrap quickly. Before pushing, scrub artefacts from `data/` or `outputs/`, rotate any demo credentials, and document secrets handling in `README.md`.
