"""viz -- read-only lineage visualizer + static HTML export (Phase 3).

Consumes the artifacts the optimizer already writes (runs/<campaign>/lineage.jsonl
plus the step_/reflection_ side files); it never imports the core loop, so it has
zero blast radius on the search itself.
"""
