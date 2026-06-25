"""graphsearch.qa -- the company-KG MCQ question-set loader.

`QaSubstrate` satisfies the engine's `Substrate` contract
(`graphretr_opt.data.substrate_proto`). Carved out of the engine
(`graphretr_opt.data.qa_substrate`) on 2026-06-25 so the company service owns its
own data port, mirroring `graphsearch.reward`.
"""
from .qa_substrate import QaSubstrate  # noqa: F401
