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

    return graph


# Extract graph features
def extract_features(graph):
    if len(graph) == 0:  # Check if the graph is empty
        return {
            'degree_distribution': Counter(),
            'avg_clustering': 0,
            'avg_path_length': float('inf'),
            'entropy': 0
        }

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


# Simple heuristic for classification
def classify_text(graph_features, ai_features, human_features):
    # Compare the graph's features to the average AI and human features
    ai_avg_clustering = np.mean([features['avg_clustering'] for features in ai_features])
    human_avg_clustering = np.mean([features['avg_clustering'] for features in human_features])

    ai_avg_entropy = np.mean([features['entropy'] for features in ai_features])
    human_avg_entropy = np.mean([features['entropy'] for features in human_features])

    # Compare clustering coefficients and entropy to determine classification
    if (abs(graph_features['avg_clustering'] - ai_avg_clustering) < abs(
            graph_features['avg_clustering'] - human_avg_clustering) and
            abs(graph_features['entropy'] - ai_avg_entropy) < abs(graph_features['entropy'] - human_avg_entropy)):
        return "AI-Written"
    else:
        return "Human-Written"


# Load datasets
ai_texts = load_dataset('ai_written.txt')
human_texts = load_dataset('human_written.txt')

# Create graphs and extract features
ai_graphs = [construct_graph(text) for text in ai_texts]
human_graphs = [construct_graph(text) for text in human_texts]

ai_features = [extract_features(graph) for graph in ai_graphs]
human_features = [extract_features(graph) for graph in human_graphs]

# Input text for classification
user_input = input("Enter the text to classify: ")

# Construct graph and extract features for the input text
new_graph = construct_graph(user_input)
new_features = extract_features(new_graph)

# Classify the input text
classification = classify_text(new_features, ai_features, human_features)
print(f"The text is classified as: {classification}")

# Add numeric labels to nodes
node_labels = {node: str(i + 1) for i, node in enumerate(new_graph.nodes())}

# Visualize the graph with numeric labels
plt.figure(figsize=(10, 8))
nx.draw(new_graph, labels=node_labels, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, font_color='black')
plt.title("Graph Representation of the Text with Number Labels")
plt.show()
