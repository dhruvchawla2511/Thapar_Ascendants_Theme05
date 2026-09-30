# ARCHIVED — not the active codebase

This `ascendants/` directory is a leftover snapshot of the repo from
BEFORE the Dhruv/Shivansh module merge (it only contains
`agent/fast_path.py`, `agent/tool_engine.py`, `agent/perception.py`, and
their tests — none of the controller, reasoner, orchestrator, or tool
manifest work exists here).

It is **not imported by anything**, and `python3 -m unittest discover -s
tests` (run from the repository root, i.e. one level up from this
directory) never touches it — `discover` only walks the `tests/` folder
it was pointed at, and this directory is a sibling of that, not a child
of it.

The real, current codebase is at the repository root: `../agent`,
`../tests`, `../demo`, `../README.md`, `../AGENTS.md`.

This directory has been left in place (not deleted) per this repo's own
rule in AGENTS.md — "Do not delete or blindly overwrite existing files" —
but it is safe to delete whenever a human decides to clean it up; nothing
depends on it.
