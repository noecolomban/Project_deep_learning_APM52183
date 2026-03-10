import networkx as nx

def build_shrikhande():
    shrikhande = nx.Graph()
    vertices = [(i, j) for i in range(4) for j in range(4)]
    shrikhande.add_nodes_from(vertices)
    generators = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    for x, y in vertices:
        for dx, dy in generators:
            shrikhande.add_edge((x, y), ((x + dx) % 4, (y + dy) % 4))
    return nx.to_numpy_array(nx.convert_node_labels_to_integers(shrikhande))

def build_rooks():
    k4 = nx.complete_graph(4)
    rooks = nx.cartesian_product(k4, k4)
    return nx.to_numpy_array(nx.convert_node_labels_to_integers(rooks))