"""data -- the engine's dataset CONTRACT layer.

After the carve-out (2026-06-25) the concrete substrates live with their service
(`starksearch.qa.Substrate`, `graphsearch.qa.QaSubstrate`); what stays here is
`substrate_proto.Substrate`, the `typing.Protocol` both conform to. Keeping the
contract in the engine lets the optimizer core depend on one named type without
importing either service.
"""
