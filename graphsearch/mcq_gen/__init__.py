"""mcq_gen -- the ONE-TIME, offline MCQ dataset generator.

NOT imported by the optimizer hot loop. It converts the free-text gold Q&A into
a multiple-choice gold set with graph-plausible distractors and an explicit
"cannot be determined from context" option, then runs a one-time closed-book
baseline pass (answerer only, no graph) to stamp `closedbook_correct` on each
item. Output: graphsearch/data/dataset.json.

Run:  python -m graphsearch.mcq_gen.build --in graphsearch/data/gold_qa.json \
                                          --out graphsearch/data/dataset.json
"""
