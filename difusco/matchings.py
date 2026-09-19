import random
import networkx as nx


def greedy_balanced_matching_family(
    graph,
    num_matchings=32,
    seed=0,
):
    """
    Fast approximate balanced family of maximal matchings.

    Edges that have been used fewer times are considered first.
    Ties are randomized.

    Much cheaper than repeated max_weight_matching calls.
    """
    rng = np.random.default_rng(seed)

    edges = [
        tuple(sorted((int(u), int(v))))
        for u, v in graph.edges()
    ]

    num_edges = len(edges)

    if num_edges == 0:
        return [[] for _ in range(num_matchings)]

    usage = np.zeros(num_edges, dtype=np.int64)

    family = []

    for _ in range(num_matchings):
        # Randomize tie-breaking first.
        tie_break = rng.random(num_edges)

        # Least-used edges first.
        order = np.lexsort(
            (
                tie_break,
                usage,
            )
        )

        used_vertices = set()
        matching = []

        for edge_idx in order:
            u, v = edges[edge_idx]

            if u in used_vertices or v in used_vertices:
                continue

            matching.append((u, v))

            used_vertices.add(u)
            used_vertices.add(v)

            usage[edge_idx] += 1

        family.append(matching)

    return family

def greedy_balanced_matching_family(
    graph,
    num_matchings=32,
    seed=0,
    vertex_balance=0.05,
):
    rng = np.random.default_rng(seed)

    edges = [
        tuple(sorted((int(u), int(v))))
        for u, v in graph.edges()
    ]

    num_edges = len(edges)

    if num_edges == 0:
        return [[] for _ in range(num_matchings)]

    edge_usage = np.zeros(num_edges, dtype=np.int64)

    node_list = list(graph.nodes())
    max_node = max(node_list) if node_list else -1

    vertex_usage = np.zeros(
        max_node + 1,
        dtype=np.int64,
    )

    family = []

    for _ in range(num_matchings):
        jitter = rng.random(num_edges) * 1e-6

        scores = np.empty(
            num_edges,
            dtype=np.float64,
        )

        for i, (u, v) in enumerate(edges):
            scores[i] = (
                edge_usage[i]
                + vertex_balance
                * (
                    vertex_usage[u]
                    + vertex_usage[v]
                )
                + jitter[i]
            )

        order = np.argsort(scores)

        used_vertices = set()
        matching = []

        for edge_idx in order:
            u, v = edges[edge_idx]

            if u in used_vertices or v in used_vertices:
                continue

            matching.append((u, v))

            used_vertices.add(u)
            used_vertices.add(v)

            edge_usage[edge_idx] += 1
            vertex_usage[u] += 1
            vertex_usage[v] += 1

        family.append(matching)

    return family

def balanced_matching_family(
    graph,
    num_matchings=32,
    seed=0,
    vertex_balance=0.05,
):
    rng = random.Random(seed)

    edges = [tuple(sorted(e)) for e in graph.edges()]
    edge_count = {e: 0 for e in edges}
    vertex_count = {v: 0 for v in graph.nodes()}

    family = []

    for _ in range(num_matchings):
        weighted_graph = nx.Graph()
        weighted_graph.add_nodes_from(graph.nodes())

        for u, v in edges:
            # Large constant is irrelevant with maxcardinality=True,
            # but keeping weight positive makes debugging easier.
            balance_penalty = (
                edge_count[(u, v)]
                + vertex_balance * (vertex_count[u] + vertex_count[v])
            )

            jitter = rng.random() * 1e-4
            weight = 1000.0 - balance_penalty + jitter

            weighted_graph.add_edge(u, v, weight=weight)

        matching = nx.algorithms.matching.max_weight_matching(
            weighted_graph,
            maxcardinality=True,
            weight="weight",
        )

        matching = [
            tuple(sorted((u, v)))
            for u, v in matching
        ]
        matching.sort()

        family.append(matching)

        for u, v in matching:
            edge_count[(u, v)] += 1
            vertex_count[u] += 1
            vertex_count[v] += 1

    return family
