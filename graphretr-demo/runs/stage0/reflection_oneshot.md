# PROMPT

You are optimizing a Python retrieval program for STaRK-prime, a biomedical knowledge-graph QA benchmark. Given a natural-language query, the program must return candidate node ids ranked by relevance. The metric being optimized is recall@20 (secondary: hit@1, MRR), averaged over held-out queries. Gold answer sets often contain MANY nodes (median ~10), and answers can be of any node type.

## Current program (the incumbent)
```python
"""Seed program: naive vector-only retrieval (the frozen baseline).

The optimizer mutates copies of this file; this copy never changes.
"""


def search(q, G):
    hits = G.vector_search(q, k=100)
    return {nid: sim for nid, sim in hits}
```

## Execution contract (hard-enforced by an AST gate -- violations are auto-rejected)
- The module must define `def search(q, G)`; helper functions and module-level
  constants are allowed. q is the query string, G the graph API below.
- Return a non-empty dict {node_id(int): score(float)}; higher score = more
  relevant. Only the ranking matters. Unreturned candidates rank last.
- NO imports. NO attribute names starting with underscore. Only these
  builtins: ['abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'float', 'int', 'len', 'list', 'max', 'min', 'range', 'reversed', 'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip'].
- Each graph call has a 2 s server-side timeout; primitives cap fan-out.
  A query crashing or timing out zeroes that query's score.

## Graph API
G is a RetrievalGraph over the STaRK-prime biomedical knowledge graph
(129,375 nodes / 8,100,498 directed typed edges in FalkorDB). All methods are
read-only, capped (fan-out <= 200, k-hop depth <= 3), and each
underlying graph query has a 2000 ms timeout (a timeout raises -- catch
nothing, just keep fan-out modest). Invalid arguments raise ValueError.

G.vector_search(text, k=20, label=None) -> list[(node_id:int, similarity:float)]
    ANN over node-text embeddings (all-MiniLM-L6-v2, cosine). similarity ~[0,1],
    sorted descending. label=None searches all 10 node-type labels and merges;
    label='disease' (etc.) restricts to one type.
G.get_neighbors(ids, rel_type=None, direction='out', limit=50) -> list[(src_id, rel_type:str, dst_id)]
    1-hop edges of the given node id(s). direction: 'out'|'in'|'both'.
G.k_hop_expand(ids, k=2, rel_type=None, max_nodes=200) -> list[node_id]
    BFS neighborhood out to depth k (deduplicated; includes start ids).
G.filter_nodes(ids, ntype=None, text_contains=None, limit=200) -> list[node_id]
    Keep ids whose node matches the exact ntype and/or whose document contains
    the substring (case-insensitive).
G.shortest_path(a, b, max_len=3) -> list[node_id]
    Shortest undirected path between two ids ([] if none within max_len,
    which is hard-capped at 3).
G.rank_by_text(ids, query_text, top=20) -> list[(node_id, similarity)]
    Re-rank candidate ids by embedding similarity to query_text, descending.
G.get_text(ids, limit=50) -> dict[node_id, str]
    Node documents (name + type + details), for inspection/filtering.

Allowlists (use these EXACT strings):
  label / node-type labels: ['anatomy', 'biological_process', 'cellular_component', 'disease', 'drug', 'effect_phenotype', 'exposure', 'gene_protein', 'molecular_function', 'pathway']
  ntype strings:            ['anatomy', 'biological_process', 'cellular_component', 'disease', 'drug', 'effect/phenotype', 'exposure', 'gene/protein', 'molecular_function', 'pathway']
  rel_type strings:         ['associated_with', 'carrier', 'contraindication', 'enzyme', 'expression_absent', 'expression_present', 'indication', 'interacts_with', 'linked_to', 'off_label_use', 'parent_child', 'phenotype_absent', 'phenotype_present', 'ppi', 'side_effect', 'synergistic_interaction', 'target', 'transporter']


## Worst failures from the latest rollout (train queries the incumbent missed)
[1] query: What are the pathways with both a 'parent-child' hierarchy related to the TCR signaling pathway and interactions with the CD101 gene or protein?
    recall@20=0.00 hit@1=0 mrr=0.00  gold ids: [128074]
    MISSED gold 128074: - name: Generation of second messenger molecules
- type: pathway
- source: REACTOME
- details:
  - dbId: 202433
  - displayName: Generation of second messenger molecules
  - stId: R-HSA-202433
  - stIdVersion: R-HSA-202433.4
  - isInDisease: False
  - isInferred: False
  - name: ['Generation of seco
    retrieved top-20 ids: [128859, 91013, 6374, 26164, 78570, 82130, 34568, 61278, 91014, 78975, 58304, 57241, 4307, 26165, 2732, 11043, 6105, 12455, 5928, 13497]
[2] query: Which diseases are subclasses of specific developmental disorders and associated with the CLN6 gene?
    recall@20=0.00 hit@1=0 mrr=0.00  gold ids: [94883]
    MISSED gold 94883: - name: developmental coordination disorder
- type: disease
- source: MONDO
- details:
  - mondo_name: developmental coordination disorder
  - mondo_definition: A disorder characterized by an impairment in the development of an individual's motor coordination skills; this impairment in motor develop
    retrieved top-20 ids: [8262, 33420, 13925, 2589, 9154, 28175, 30003, 92, 59560, 11838, 11594, 60652, 38805, 2051, 2851, 25206, 7953, 27618, 57922, 12672]
[3] query: Please find genes or proteins that interact with the GTPase function, participate in the TBC/RABGAPs pathway, are mainly expressed in CD8+ T cells, belong to the small GTPase superfamily, specifically the Rab family, and may be involved in vesicle transport.
    recall@20=0.00 hit@1=0 mrr=0.00  gold ids: [7668]
    MISSED gold 7668: - name: RAB33A
- type: gene/protein
- source: NCBI
- details:
  - query: RAB33A
  - alias (other gene names): RabS10
  - genomic_pos (genomic position): {'chr': 'X', 'end': 130184870, 'ensemblgene': 'ENSG00000134594', 'start': 130171962, 'strand': 1}
  - name (gene name): RAB33A, member RAS oncogene
    retrieved top-20 ids: [58340, 58553, 58547, 100657, 58557, 6342, 2077, 40769, 110892, 4423, 55000, 55639, 7731, 53779, 10184, 3975, 46403, 50306, 47492, 5658]
[4] query: Identify the gene or protein transcribed in adrenal gland tissue involved in miRNA-mediated gene silencing and linked to acute intermittent porphyria pathogenesis.
    recall@20=0.00 hit@1=0 mrr=0.00  gold ids: [61541]
    MISSED gold 61541: - name: LINC00461
- type: gene/protein
- source: NCBI
- details:
  - query: LINC00461
  - notfound: True

    retrieved top-20 ids: [46666, 112770, 46368, 48083, 46369, 47093, 108571, 51555, 110776, 105331, 112521, 27843, 48683, 47095, 43313, 46364, 112369, 106339, 43312, 47094]
[5] query: What are the common gene targets for 1-Naphthylamine-5-sulfonic acid and 2',6'-Difluorobiphenyl-4-carboxylic acid?
    recall@20=0.00 hit@1=0 mrr=0.00  gold ids: [111]
    MISSED gold 111: - name: TTR
- type: gene/protein
- source: NCBI
- details:
  - query: TTR
  - alias (other gene names): ['ATTR', 'CTS', 'CTS1', 'HEL111', 'HsT2651', 'PALB', 'TBPA', 'TTN']
  - genomic_pos (genomic position): {'chr': '18', 'end': 31598833, 'ensemblgene': 'ENSG00000118271', 'start': 31557009, 'strand'
    retrieved top-20 ids: [14086, 119235, 17488, 14093, 119298, 119261, 118492, 116383, 119150, 18943, 121680, 14084, 119890, 6073, 116069, 121679, 118698, 16777, 116149, 116145]
[6] query: Which gene or protein is not expressed in either the small intestine or the biceps brachii muscle?
    recall@20=0.00 hit@1=0 mrr=0.00  gold ids: [34739]
    MISSED gold 34739: - name: DRAXIN
- type: gene/protein
- source: NCBI
- details:
  - query: DRAXIN
  - alias (other gene names): ['AGPA3119', 'C1orf187', 'UNQ3119', 'neucrin']
  - genomic_pos (genomic position): {'chr': '1', 'end': 11725857, 'ensemblgene': 'ENSG00000162490', 'start': 11691710, 'strand': 1}
  - name (g
    retrieved top-20 ids: [70251, 63952, 66355, 70883, 66337, 70248, 68775, 66577, 70895, 66336, 63821, 70894, 70893, 70892, 68428, 71404, 64008, 64026, 75406, 68427]
[7] query: What are the common gene targets associated with the binding or transport of both Dibenzofuran-4,6-Dicarboxylic Acid and 2',6'-Difluorobiphenyl-4-carboxylic Acid?
    recall@20=0.00 hit@1=0 mrr=0.00  gold ids: [111]
    MISSED gold 111: - name: TTR
- type: gene/protein
- source: NCBI
- details:
  - query: TTR
  - alias (other gene names): ['ATTR', 'CTS', 'CTS1', 'HEL111', 'HsT2651', 'PALB', 'TBPA', 'TTN']
  - genomic_pos (genomic position): {'chr': '18', 'end': 31598833, 'ensemblgene': 'ENSG00000118271', 'start': 31557009, 'strand'
    retrieved top-20 ids: [14076, 42158, 53613, 120090, 121680, 120395, 119235, 12323, 14084, 53552, 117540, 11121, 5748, 506, 54600, 54547, 115141, 119306, 56755, 121679]
[8] query: I'm experiencing weakness and tingling in my hands and feet, and there's a history of breast secretory carcinoma in my family. What could be my possible illness considering these symptoms and family medical background?
    recall@20=0.00 hit@1=0 mrr=0.00  gold ids: [96661]
    MISSED gold 96661: - name: breast cystic hypersecretory carcinoma
- type: disease
- source: MONDO
- details:
  - mondo_name: breast cystic hypersecretory carcinoma

    retrieved top-20 ids: [95814, 95731, 95813, 37775, 96976, 95712, 98056, 94766, 38164, 95523, 95332, 37774, 36676, 99527, 36285, 38318, 95249, 96923, 38267, 36860]

## Previously rejected edits (these did NOT improve val recall@20 -- do not repeat them)
(none yet)

## Your task
1. Diagnose in a few sentences WHY the incumbent misses these gold nodes
   (look at what the missed nodes' texts share, and which primitives could
   reach them).
2. Propose ONE improved program. Edit budget: at most 4 changed
   regions (difflib hunks) vs the incumbent -- prefer small, targeted edits
   over rewrites. For each edit give a one-line rationale.
3. End your reply with the COMPLETE new module in a single ```python fenced
   block (it replaces the whole file; it must compile under the contract).


---

# RESPONSE (attempt 1)

## 1. Diagnosis

The incumbent ranks purely by **text embedding similarity to the raw query**, so it can only surface nodes whose own document is lexically/semantically close to the question. But almost every missed gold is reached by **graph structure, not text**:

- **Intersection queries** ([5],[7] → TTR; [1]): the answer is a *common neighbor* of two anchor entities ("common gene targets of drug A **and** drug B", "parent-child of TCR **and** interacts CD101"). The gold node's text (e.g. `TTR / ATTR / TBPA…`) shares nothing with the chemical names in the query, so it's invisible to vector search — but it's a `target`/`parent_child`/`interacts_with` neighbor of seeds that *are* retrievable.
- **Typed-relation / family queries** ([3] RAB33A, [2], [6], [8]): the gold is a 1–2 hop neighbor (ppi, parent_child, expression edges) of textually-findable anchors, with low direct text sim.

So the fix is to **expand the neighborhood of the top vector seeds and reward nodes that are common neighbors of multiple anchors** (the intersection signal), while re-ranking the expansion by text to keep it focused.

## 2. Proposed edits (1 contiguous hunk replacing the body; 4 logical changes)

1. **Keep vector seeds as the scored base** — preserves the baseline's recall floor.
2. **Expand `get_neighbors(direction='both')` over the top-12 seeds and count hit frequency** — `freq>=2` captures the "common target / intersection" gold the baseline can't reach.
3. **Re-rank the new (graph-only) candidates by `rank_by_text`** and score them as `0.5*textsim + graph-frequency bonus` — text keeps relevant neighbors, frequency lifts intersection answers that have ~0 text sim.
4. **Give seeds a small frequency bonus** — a seed that is also a graph hub for the query's anchors rises toward the top.

## 3. Complete module

```python
"""Vector seeds + anchor-neighborhood expansion with common-neighbor (intersection) boosting."""


def search(q, G):
    seeds = G.vector_search(q, k=100)
    scores = {nid: float(sim) for nid, sim in seeds}
    seen = set(scores)

    # Top seeds act as anchors; expand their 1-hop neighborhood both directions.
    anchors = [nid for nid, sim in seeds[:12]]
    freq = {}
    for src, rel, dst in G.get_neighbors(anchors, direction='both', limit=50):
        for x in (src, dst):
            freq[x] = freq.get(x, 0) + 1

    # Graph-only candidates: score by text relevance plus an intersection bonus
    # (freq>=2 == reached from multiple anchors -> "common target / parent-child" answers).
    new = [x for x in freq if x not in seen]
    text = {}
    if new:
        for nid, sim in G.rank_by_text(new, q, top=60):
            text[nid] = float(sim)
    for x in new:
        scores[x] = text.get(x, 0.0) * 0.5 + 0.1 * min(max(freq[x] - 1, 0), 3)

    # Seeds that are also graph hubs for the query's anchors get a small lift.
    for x in seen:
        if x in freq:
            scores[x] = scores[x] + 0.04 * min(freq[x], 4)

    return scores
```
