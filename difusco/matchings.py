import random
import networkx as nx


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
