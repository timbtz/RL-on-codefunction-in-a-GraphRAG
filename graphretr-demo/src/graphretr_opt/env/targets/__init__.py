"""env/targets -- SearchTarget adapters.

FakeSearchTarget (in-memory, deterministic) and SubprocessSearchTarget (spawns
the real graphsearch service over Neo4j). Importing this package pulls neither
Neo4j nor LangChain -- the heavy import lives inside the subprocess worker.
"""
