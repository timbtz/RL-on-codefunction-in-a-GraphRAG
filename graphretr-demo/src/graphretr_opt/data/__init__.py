"""data -- the QA/KB data layer.

`Substrate` (the STaRK-prime substrate) pulls heavy deps (stark_qa, torch), so it
is exposed LAZILY: `from graphretr_opt.data import Substrate` still works, but
merely importing the package (e.g. to reach `qa_substrate.QaSubstrate` on the
graph_search path) does not drag in stark_qa. This is what lets the graph_search
campaign run on a machine without the function-campaign's dataset deps.
"""


def __getattr__(name):  # PEP 562 lazy attribute
    if name == "Substrate":
        from .substrate import Substrate
        return Substrate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
