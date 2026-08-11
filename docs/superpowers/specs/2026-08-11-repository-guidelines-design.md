# Repository Guidelines Design

## Goal

Create a concise `AGENTS.md` contributor guide for the current repository. The
document must be professional, actionable, and approximately 250–350 words.

## Content

The guide will be titled **Repository Guidelines** and cover:

- Source, test, configuration, asset, script, documentation, and runtime-data
  locations.
- Windows and macOS environment setup, dashboard startup, engine startup, and
  pytest commands already supported by the repository.
- Python conventions: four-space indentation, snake_case functions/modules,
  PascalCase classes, package-path imports, type hints, and focused modules.
- Pytest conventions: `tests/test_*.py`, isolated tests, mocked external
  services/models, and explicit opt-in for hardware or audible tests.
- Commit conventions inferred from history: short imperative subjects with
  optional conventional prefixes; pull requests must explain scope, testing,
  configuration changes, and UI evidence when applicable.
- Security rules for `.env`, Telegram and ElevenLabs credentials, known-face
  images, model weights, and `data/logs/` runtime artifacts.

## Constraints

Use only commands and paths present in the repository. Do not claim a formatter,
linter, or coverage threshold that is not configured. Do not modify or stage
unrelated working-tree changes.
