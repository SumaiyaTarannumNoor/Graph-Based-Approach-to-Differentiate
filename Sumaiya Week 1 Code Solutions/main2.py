import networkx as nx
import re
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np


# Load the datasets
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().splitlines()
    return text


# Tokenization of Bangla text (simple whitespace tokenization)
def tokenize(text):
    return re.findall(r'\b\w+\b', text)


# Construct a directed graph from the text
def construct_graph(text):
    graph = nx.DiGraph()
    words = tokenize(text)

    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        if graph.has_edge(word1, word2):
            graph[word1][word2]['weight'] += 1
        else:
            graph.add_edge(word1, word2, weight=1)

    # Label nodes with English numbers
    mapping = {node: str(i + 1) for i, node in enumerate(graph.nodes())}
    nx.relabel_nodes(graph, mapping, copy=False)

    return graph


# Extract graph features
def extract_features(graph):
    degrees = dict(graph.degree())
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    clustering_coefficients = nx.clustering(graph)

    # Degree distribution
    degree_distribution = Counter(degrees.values())

    # Average clustering coefficient
    avg_clustering = np.mean(list(clustering_coefficients.values()))

    # Handle the average path length calculation
    try:
        if nx.is_strongly_connected(graph):
            avg_path_length = nx.average_shortest_path_length(graph)
        else:
            # Calculate the average path length for the largest strongly connected component
            largest_scc = max(nx.strongly_connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_scc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
    except nx.NetworkXError:
        avg_path_length = float('inf')

    # Graph entropy (based on degree distribution)
    total_degrees = sum(degree_distribution.values())
    entropy = -sum((count / total_degrees) * np.log2(count / total_degrees) for count in degree_distribution.values() if count != 0)

    return {
        'degree_distribution': degree_distribution,
        'avg_clustering': avg_clustering,
        'avg_path_length': avg_path_length,
        'entropy': entropy
    }


# Heuristic Coloring based on clustering coefficient and entropy
def heuristic_coloring(graph, avg_clustering, entropy):
    color_map = []
    degree_values = [d for _, d in graph.degree()]  # Get degree values as a list
    mean_degree = np.mean(degree_values)  # Calculate mean of degree values

    for node in graph:
        node_clustering = nx.clustering(graph, node)
        node_degree = graph.degree[node]

        # Simple heuristic: color by clustering coefficient and degree
        if node_clustering > avg_clustering and node_degree > mean_degree:
            color_map.append('red')  # AI-like characteristics
        else:
            color_map.append('blue')  # Human-like characteristics
    return color_map


# Load datasets
ai_texts = load_dataset('ai_written.txt')
human_texts = load_dataset('human_written.txt')

# Create graphs and extract features for AI and Human written texts
ai_graph = nx.compose_all([construct_graph(text) for text in ai_texts])
human_graph = nx.compose_all([construct_graph(text) for text in human_texts])

ai_features = extract_features(ai_graph)
human_features = extract_features(human_graph)

# Generate heuristic color maps
ai_color_map = heuristic_coloring(ai_graph, ai_features['avg_clustering'], ai_features['entropy'])
human_color_map = heuristic_coloring(human_graph, human_features['avg_clustering'], human_features['entropy'])

# Visualize the graphs side by side
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# AI-written graph
axs[0].set_title("AI-Written Text Graph with Heuristic Coloring")
nx.draw(ai_graph, ax=axs[0], with_labels=True, node_color=ai_color_map, edge_color='gray', node_size=500, font_size=10,
        font_color='black')

# Human-written graph
axs[1].set_title("Human-Written Text Graph with Heuristic Coloring")
nx.draw(human_graph, ax=axs[1], with_labels=True, node_color=human_color_map, edge_color='gray', node_size=500, font_size=10,
        font_color='black')

plt.show()
