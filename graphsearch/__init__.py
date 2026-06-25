"""graphsearch -- in-repo copy of the agentic graph search service plus the
offline MCQ dataset generator (mcq_gen). The runnable service lives under
src/ (kept off this package's import path so the original absolute imports
resolve via sys.path.insert(0, 'graphsearch/src')); mcq_gen is a normal
`python -m graphsearch.mcq_gen.build` entrypoint run from the repo root.
"""
