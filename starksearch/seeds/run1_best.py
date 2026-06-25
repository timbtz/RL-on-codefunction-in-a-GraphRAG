"""Anchor-and-expand retrieval with answer-type inference.

A whole-query embedding only reaches textually-similar nodes, so it misses
RELATIONAL answers (the targets of a drug, genes expressed in a tissue, the
cellular components of a gene) and generically-named ones (nucleus, cytoplasm).
So we: (1) treat the top vector hits as the query's named entities (anchors)
and pull their graph neighbours -- a node touching several anchors is a
'common to A and B' answer; (2) infer the answer NODE TYPE from the leading
noun of the query, pulling typed vector hits and boosting that type; (3) for
relational chains (transported-by, lacks-expression, side-effect-of,
contraindication, interacts-with) take one extra, frequency-ranked hop; and
(4) -- the key correction -- strongly boost EXPANSION candidates whose node
type matches the inferred answer, because the gold is usually a typed neighbour
(an anatomy that lacks expression, a drug with a side effect) that scores near
zero on text alone and would otherwise fall out of the candidate pool. Direct
vector hits are always kept, so we never regress below the vector-only
baseline. No try/except: fan-out is kept modest because a timed-out graph call
zeroes the whole query.

And (5) when the query NAMES a relation (synergistic, PPI, transporter, side
effect, expression-absent, target), expand the anchors along THAT typed edge.
The anchors are high-degree hubs, so the generic both-direction call in (1)
buries the one gold neighbour reached by the named relation against the fan-out
cap; a typed call returns it directly (Dexfenfluramine's synergistic partners,
TMEM185A's PPI partners). Typed neighbours feed the same intersection/type/text
machinery, and a flat boost keeps them past the pool cut even on one anchor.

And (6) when the answer is a DRUG that 'interacts with' / inhibits / targets a
named gene or protein, the gold drug is reached along a DRUG-PROTEIN edge
(target / enzyme / carrier / transporter), NOT the gene-gene 'interacts_with'
edge the generic 'interact' keyword maps to; we follow all four so the specific
gold drug enters the pool and is type/text re-ranked.

And (7) ONTOLOGY-HIERARCHY answers (precursor/subsequent/subtype diseases 'in
the disease ontology hierarchy') are reached along the 'parent_child' edge,
which no keyword previously mapped; we now follow it so the gold parent/child
(simian AIDS -> monkey disease) enters via the relation-typed expansion. This
also fires on GENERALITY phrasing -- 'more general and specific conditions',
'broader/narrower' -- where the answer is a parent/child disease but the query
never names the ontology (bone remodeling disease off hyperostosis).

And (8) the EXPRESSION-ABSENT chain: 'which anatomical structures LACK the
expression of genes implicated in X' is gene -> expression_absent -> anatomy,
but the query LEADS with 'anatomical structures', so the whole-query anchors
are all anatomies and never touch the implicated genes. We independently pull
the genes (typed gene vector hits -- the query describes their function),
follow expression_absent PER GENE, and accumulate the distinct-gene count in
conn so the high-degree shared anatomies (cerebellar vermis, deltoid) that lack
expression across MANY implicated genes -- and score ~zero on text -- rank top.

And (8b) the REVERSE expression-absent chain: 'which gene is unexpressed in
BOTH the corpus callosum AND the pancreas' is anatomy -> expression_absent ->
gene, but the query LEADS with 'gene/protein', so every anchor is a gene and
the named anatomies are never touched. Symmetric to (8): we pull the anatomies
(typed anatomy vector hits) and follow expression_absent PER anatomy,
accumulating the distinct-anatomy count in conn so the gold gene unexpressed in
BOTH named anatomies ranks top by that intersection count (NPFFR1).

And (9) the DRUG-FROM-GENE chain ('investigational drugs targeting the ADA
enzyme', 'compound that acts upon genes with a death effector domain'): the
gold drugs are experimental DrugBank stubs with ~zero query text, reachable
ONLY from the NAMED gene along a drug-protein edge. But the query LEADS with
'drug', so every anchor is a drug and the gene is never one. As in (8) we
independently pull the gene as a typed gene hit and follow target / enzyme /
carrier / transporter PER gene to the gold drugs.

And (10) the GENE-GENE SHARED-TRAIT intersection ('a gene that interacts with
NAMED gene AND shares a phenotype/disease/effect with it' -- AGT/REN,
REST/NBEA, CAV3/JPH2, GLRX2/...): the gold is the named gene's ppi/
interacts_with partner that ALSO shares one of its phenotype/disease neighbours
-- the SECOND constraint the generic fan-out ignores, so the gold ranks on text
alone (~0) and is buried among hub partners. We pull the named gene (top typed
gene hit), collect its interaction partners (A) and the genes sharing one of
its phenotype/disease/assoc neighbours (B), and strongly boost A intersect B --
the exact gold shape, narrow enough not to flood the pool.
"""

ANCHORS = 12         # top vector hits treated as query entities
EXPAND_LIMIT = 200   # 1-hop fan-out cap
HOP2_ANCHORS = 8     # intermediates expanded on the 2nd hop
HOP2_LIMIT = 150     # 2-hop fan-out cap
POOL = 200           # cap on candidates sent to rank_by_text / filter_nodes
DIRECT_W = 1.0       # raw query->node similarity
TYPED_VEC_W = 0.6    # type-restricted vector similarity
CONN_W = 0.6         # per distinct anchor a candidate connects to
HOP2_W = 0.3         # per intermediate a 2-hop candidate connects to
TEXT_W = 0.35        # candidate text similarity to the query
TYPE_W = 0.6         # candidate is of the inferred answer type
REL_TYPE_W = 0.7     # EXPANSION candidate is of the inferred answer type
REL_HOP_W = 0.5      # candidate reached by the query's NAMED relation type

# Drug<->protein edge types. The generic 'interact' keyword maps to the
# gene-gene 'interacts_with' relation, which never reaches a drug answer; when
# the answer IS a drug, the gold is linked to the named gene along one of these.
DRUG_PROTEIN_RELS = ('target', 'enzyme', 'carrier', 'transporter')

# Expression-absent chain: independently expand implicated genes (the query
# describes their function) to the anatomies that LACK their expression, PER
# gene so a hub gene's edges aren't truncated against the others.
EXP_GENES = 12       # gene hits expanded individually for the expression chain
EXP_LIMIT = 80       # PER-GENE expression-absent fan-out cap

# Drug-from-gene chain (9): pull the NAMED gene as a typed gene hit and follow
# drug-protein edges PER gene to the gold (experimental, ~zero-text) drugs.
DP_GENES = 8         # typed gene hits expanded individually to reach drugs
DP_LIMIT = 80        # PER-gene drug-protein fan-out cap

# Gene-gene shared-trait chain (10): the gold partner of the named gene that
# ALSO shares one of its phenotype/disease neighbours.
GG_GENES = 4         # named-gene candidates used for the shared-trait chain
GG_LIMIT = 120       # PER-call fan-out cap for the shared-trait chain
GG_TRAITS = 12       # named gene's phenotype/disease neighbours expanded back
GG_INTERSECT_W = 1.2 # partner that ALSO shares a trait -- the exact gold shape

# (query substring, rel_type) -- if the query names a relation, expand the
# anchors along THAT typed edge so the specific gold neighbour is not truncated
# by the generic both-direction fan-out. Most-specific keywords first; the
# first 2 distinct rel_types matched (in list order) are used.
REL_KEYWORDS = [
    ('synergistic', 'synergistic_interaction'),
    ('ppi', 'ppi'),
    ('protein-protein', 'ppi'),
    ('side effect', 'side_effect'),
    ('contraindic', 'contraindication'),
    ('off-label', 'off_label_use'),
    ('off label', 'off_label_use'),
    # Ontology-hierarchy answers (precursor / subsequent / subtype diseases)
    # are reached along the parent_child edge.
    ('hierarchy', 'parent_child'),
    ('ontology', 'parent_child'),
    ('precursor', 'parent_child'),
    ('subsequent', 'parent_child'),
    ('subtype', 'parent_child'),
    ('subclass', 'parent_child'),
    ('lineage', 'parent_child'),
    ('ancestor', 'parent_child'),
    ('descendant', 'parent_child'),
    # Generality phrasing -- 'more general and specific conditions', broader/
    # narrower -- asks for a parent/child disease without naming the ontology.
    ('more general', 'parent_child'),
    ('general and specific', 'parent_child'),
    ('broader', 'parent_child'),
    ('narrower', 'parent_child'),
    ('transporter', 'transporter'),
    ('carrier', 'carrier'),
    ('lack', 'expression_absent'),
    ('not expressed', 'expression_absent'),
    ('absent', 'expression_absent'),
    ('expressed in', 'expression_present'),
    ('express', 'expression_present'),
    ('enzyme', 'enzyme'),
    ('target', 'target'),
    ('transport', 'transporter'),
    ('indicat', 'indication'),
    ('associated with', 'associated_with'),
    ('interact', 'interacts_with'),
]

# (keyword, vector_search label, filter_nodes ntype) -- order is a fallback;
# the EARLIEST occurrence in the query wins (questions lead with the answer).
TYPES = [
    ('cellular structure', 'cellular_component', 'cellular_component'),
    ('cellular component', 'cellular_component', 'cellular_component'),
    ('organelle', 'cellular_component', 'cellular_component'),
    ('gene', 'gene_protein', 'gene/protein'),
    ('protein', 'gene_protein', 'gene/protein'),
    ('drug', 'drug', 'drug'),
    ('medication', 'drug', 'drug'),
    ('pathway', 'pathway', 'pathway'),
    ('disease', 'disease', 'disease'),
    ('disorder', 'disease', 'disease'),
    ('syndrome', 'disease', 'disease'),
    ('phenotype', 'effect_phenotype', 'effect/phenotype'),
    ('symptom', 'effect_phenotype', 'effect/phenotype'),
    ('side effect', 'effect_phenotype', 'effect/phenotype'),
    ('biological process', 'biological_process', 'biological_process'),
    ('molecular function', 'molecular_function', 'molecular_function'),
    ('anatom', 'anatomy', 'anatomy'),
    ('tissue', 'anatomy', 'anatomy'),
    ('exposure', 'exposure', 'exposure'),
    ('condition', 'disease', 'disease'),
    ('pharmaceutical', 'drug', 'drug'),
    ('medicine', 'drug', 'drug'),
    ('compound', 'drug', 'drug'),
]


def infer_type(ql):
    best = None
    best_at = len(ql) + 1
    for kw, label, ntype in TYPES:
        at = ql.find(kw)
        if 0 <= at < best_at:
            best_at = at
            best = (label, ntype)
    return best


def search(q, G):
    ql = q.lower()
    typ = infer_type(ql)
    label = typ[0] if typ else None
    ntype = typ[1] if typ else None

    base = G.vector_search(q, k=100)
    scores = {}
    for nid, sim in base:
        scores[nid] = DIRECT_W * sim

    # Typed vector hits: surface correct-type nodes the full-query ANN buries.
    if label:
        for nid, sim in G.vector_search(q, k=50, label=label):
            scores[nid] = scores.get(nid, 0.0) + TYPED_VEC_W * sim

    # 1-hop expansion: per neighbour, count how many DISTINCT anchors it
    # touches (>=2 == 'common to both entities' answer).
    anchors = [nid for nid, sim in base[:ANCHORS]]
    aset = set(anchors)
    conn = {}
    if anchors:
        for s, r, d in G.get_neighbors(anchors, direction='both', limit=EXPAND_LIMIT):
            if s in aset and d not in aset:
                conn.setdefault(d, set()).add(s)
            if d in aset and s not in aset:
                conn.setdefault(s, set()).add(d)

    # Relation-typed expansion: when the query NAMES a relation (synergistic,
    # PPI, transporter, side effect, expression-absent, parent_child...), follow
    # THAT edge type from the anchors. The untyped call above buries a high-
    # degree hub's specific gold neighbour against the fan-out cap; a typed call
    # returns it. Typed neighbours are merged into conn so they inherit the
    # intersection count, pool cut, relc type-filter, and text/type re-rank.
    relhit = set()
    rels = []
    for kw, rt in REL_KEYWORDS:
        if kw in ql and rt not in rels:
            rels.append(rt)
    if anchors:
        for rt in rels[:2]:
            for s, r, d in G.get_neighbors(anchors, rel_type=rt, direction='both', limit=EXPAND_LIMIT):
                if s in aset and d not in aset:
                    conn.setdefault(d, set()).add(s)
                    relhit.add(d)
                if d in aset and s not in aset:
                    conn.setdefault(s, set()).add(d)
                    relhit.add(s)

    # Drug-from-gene chain (9): when the answer is a DRUG that targets /
    # inhibits / acts upon a named gene or protein, the gold drug (an
    # experimental DrugBank stub with ~zero query text) hangs off the NAMED
    # gene along a drug-protein edge. But the query LEADS with 'drug', so every
    # anchor is a drug and the gene is never one -- the old block expanded the
    # drug anchors and only ever reached the genes they target. We instead pull
    # the gene as a typed gene vector hit and follow target / enzyme / carrier /
    # transporter PER gene (tiny per-gene fan-out) to the gold drugs, merged
    # into conn/relhit for the drug type/text re-rank below.
    if anchors and ntype == 'drug' and any(k in ql for k in (
            'target', 'inhibit', 'enzyme', 'bind', 'block', 'act',
            'affinity', 'agonist', 'antagonist', 'substrate', 'transport',
            'interact', 'metaboli', 'modulat', 'studied', 'associat',
            'effect')):
        dgenes = [g for g, sim in G.vector_search(q, k=40, label='gene_protein')]
        dgset = set(dgenes)
        for g in dgenes[:DP_GENES]:
            for rt in DRUG_PROTEIN_RELS:
                for s, r, d in G.get_neighbors([g], rel_type=rt, direction='both', limit=DP_LIMIT):
                    if s == g and d not in dgset:
                        conn.setdefault(d, set()).add(g)
                        relhit.add(d)
                    elif d == g and s not in dgset:
                        conn.setdefault(s, set()).add(g)
                        relhit.add(s)

    # Gene-gene shared-trait intersection (10): 'a gene that interacts with
    # NAMED gene AND shares a phenotype/disease/effect with it'. The gold is the
    # named gene's ppi/interacts_with partner that ALSO shares one of its
    # phenotype/disease neighbours -- the second constraint the generic fan-out
    # ignores, so the gold ranks on text alone (~0). We pull the named gene (top
    # typed gene hits), collect its interaction partners (A) and the genes
    # sharing one of its phenotype/disease/assoc neighbours (B), and strongly
    # boost A intersect B. The intersection is narrow, so unlike a blanket ppi
    # expansion it does not flood the pool.
    if anchors and ntype == 'gene/protein' and any(k in ql for k in (
            'interact', 'engage', 'bind', 'partner', 'ppi')) and any(
            k in ql for k in ('phenotyp', 'effect', 'disease', 'associat',
                              'common', 'mutual', 'shar', 'role', 'damage')):
        igenes = [g for g, sim in G.vector_search(q, k=20, label='gene_protein')]
        igset = set(igenes)
        partners = {}
        traits = set()
        for g in igenes[:GG_GENES]:
            for rt in ('ppi', 'interacts_with'):
                for s, r, d in G.get_neighbors([g], rel_type=rt, direction='both', limit=GG_LIMIT):
                    p = d if s == g else (s if d == g else None)
                    if p is not None and p not in igset:
                        partners.setdefault(p, set()).add(g)
            for rt in ('phenotype_present', 'associated_with'):
                for s, r, d in G.get_neighbors([g], rel_type=rt, direction='both', limit=GG_LIMIT):
                    t = d if s == g else (s if d == g else None)
                    if t is not None:
                        traits.add(t)
        sharers = set()
        if traits:
            for s, r, d in G.get_neighbors(sorted(traits)[:GG_TRAITS], direction='both', limit=GG_LIMIT):
                sharers.add(s)
                sharers.add(d)
        for p in partners:
            conn.setdefault(p, set()).update(partners[p])
            relhit.add(p)
            if p in sharers:
                scores[p] = scores.get(p, 0.0) + GG_INTERSECT_W

    # Expression-absent chain: 'which anatomical structures LACK expression of
    # genes implicated in X' is gene -> expression_absent -> anatomy, but the
    # query LEADS with 'anatomical structures', so the whole-query anchors are
    # all anatomies and never touch the implicated genes. Independently pull the
    # genes (typed gene vector hits) and follow expression_absent PER gene,
    # accumulating in conn the count of distinct implicated genes each anatomy
    # lacks expression in -- the high-degree shared anatomies (cerebellar
    # vermis, deltoid) that score ~zero on text then rank top by that count.
    if anchors and ntype == 'anatomy' and any(k in ql for k in (
            'lack', 'absent', 'not express', 'do not express',
            'fail to express', 'without express')):
        genes = [g for g, sim in G.vector_search(q, k=40, label='gene_protein')]
        gset = set(genes)
        for g in genes[:EXP_GENES]:
            for s, r, d in G.get_neighbors([g], rel_type='expression_absent', direction='both', limit=EXP_LIMIT):
                if s == g and d not in gset:
                    conn.setdefault(d, set()).add(g)
                    relhit.add(d)
                elif d == g and s not in gset:
                    conn.setdefault(s, set()).add(g)
                    relhit.add(s)

    # Reverse expression-absent chain (8b): 'which gene/protein is consistently
    # unexpressed in BOTH the corpus callosum AND the pancreas' is anatomy ->
    # expression_absent -> gene, but the query LEADS with 'gene/protein', so
    # every anchor is a gene and the named anatomies are never touched.
    # Symmetric to the block above: pull the anatomies (typed anatomy vector
    # hits) and follow expression_absent PER anatomy, accumulating in conn the
    # count of distinct named anatomies each gene lacks expression in -- the
    # gold gene unexpressed in BOTH (NPFFR1) ranks top by that intersection.
    if anchors and ntype == 'gene/protein' and any(k in ql for k in (
            'unexpress', 'not express', 'never express', 'lack', 'absent',
            'do not express', 'fail to express', 'without express',
            'not exhibit')):
        anats = [a for a, sim in G.vector_search(q, k=40, label='anatomy')]
        anset = set(anats)
        for a in anats[:EXP_GENES]:
            for s, r, d in G.get_neighbors([a], rel_type='expression_absent', direction='both', limit=EXP_LIMIT):
                if s == a and d not in anset:
                    conn.setdefault(d, set()).add(a)
                    relhit.add(d)
                elif d == a and s not in anset:
                    conn.setdefault(s, set()).add(a)
                    relhit.add(s)

    cand = sorted(conn, key=lambda n: len(conn[n]), reverse=True)[:POOL]
    for n in cand:
        scores[n] = scores.get(n, 0.0) + CONN_W * len(conn[n])
    # The named relation is itself strong evidence: keep typed neighbours past
    # the pool cut even when they touch only one anchor.
    for n in relhit:
        scores[n] = scores.get(n, 0.0) + REL_HOP_W

    # 2-hop for relational chains (transported-by, lacks-expression,
    # side-effect-of, contraindication, interacts-with): the ubiquitous answers
    # are the highest-FREQUENCY second-hop neighbours.
    do_hop2 = any(k in ql for k in (
        'interact', 'engage', 'affected by', 'transport', 'express',
        'lack', 'side effect', 'consequence', 'contraindic', 'caused by'))
    freq = {}
    if do_hop2 and cand:
        h2 = cand[:HOP2_ANCHORS]
        h2set = set(h2)
        for s, r, d in G.get_neighbors(h2, direction='both', limit=HOP2_LIMIT):
            if s in h2set and d not in h2set and d not in aset:
                freq[d] = freq.get(d, 0) + 1
            if d in h2set and s not in h2set and s not in aset:
                freq[s] = freq.get(s, 0) + 1
        for n in sorted(freq, key=lambda n: freq[n], reverse=True)[:POOL]:
            scores[n] = scores.get(n, 0.0) + HOP2_W * freq[n]

    # Relational answers of the inferred type: the gold node is often a typed
    # neighbour (an anatomy that lacks expression, a drug with a side effect, a
    # gene that interacts) with near-zero text similarity. Boost expansion
    # candidates whose node type matches the answer, so they survive the pool
    # cut and reach the text/type re-rank below.
    if ntype:
        relc = sorted(conn, key=lambda n: len(conn[n]), reverse=True)
        relc = relc + sorted(freq, key=lambda n: freq[n], reverse=True)
        relc = relc[:POOL]
        if relc:
            for nid in G.filter_nodes(relc, ntype=ntype, limit=POOL):
                scores[nid] = scores.get(nid, 0.0) + REL_TYPE_W

    # Text re-rank of the top candidate pool.
    pool = sorted(scores, key=lambda n: scores[n], reverse=True)[:POOL]
    if pool:
        for nid, sim in G.rank_by_text(pool, q, top=POOL):
            scores[nid] = scores.get(nid, 0.0) + TEXT_W * sim

    # Boost candidates of the inferred answer type.
    if ntype and pool:
        for nid in G.filter_nodes(pool, ntype=ntype, limit=POOL):
            scores[nid] = scores.get(nid, 0.0) + TYPE_W

    return scores
