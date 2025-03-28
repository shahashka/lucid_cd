from causallearn.graph.GeneralGraph import GeneralGraph, Endpoint, Edge
import networkx as nx
import time
import numpy as np
import itertools
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Dag import Dag
N=200
G = nx.gnm_random_graph(n=N, m=N, seed=3, directed=True)
edges = list(G.edges())
nodes = list(G.nodes())
nodes_cl = [GraphNode(x) for x in nodes]
node_names = [("X%d" % (i + 1)) for i in range(N)]
nodes_cl = []
for name in node_names:
    node = GraphNode(name)
    nodes_cl.append(node)
G_cl = Dag(nodes_cl)

# Graph creation 
start = time.time()
G_nx = nx.DiGraph(edges)
time_nx = time.time()
for (i,j) in edges:
    G_cl.add_edge(Edge(G_cl.nodes[i], G_cl.nodes[j], Endpoint.TAIL, Endpoint.ARROW))
time_cl = time.time()
print(f"Graph generation time nx {time_nx - start}, cl {time_cl - time_nx}")

# Get number of edges
start = time.time()
num_edges = G_nx.number_of_edges
time_nx = time.time()
num_edges = G_cl.get_num_edges()
time_cl = time.time()
print(f"Get number of edges time nx {time_nx - start}, cl {time_cl - time_nx}")

# Topological sort
start = time.time()
top_sort = nx.topological_sort(G_nx)
time_nx = time.time()
top_sort = G_cl.get_causal_ordering()
time_cl = time.time()
print(f"Topological sort time nx {time_nx - start}, cl {time_cl - time_nx}")

