import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re


def visualize_transfer_network(graph, clubs_to_visualize, include_loan=True):
    """
    Visualize the transfer network for a specific season.
    Args:
        graph (nx.DiGraph): Directed transfer network.
        clubs_to_visualize (list): List of club names to visualize.
    """
    sub_graph = graph.subgraph(clubs_to_visualize)

    if include_loan:
        edge_colors = [
            "red" if d["is_loan"] == 1 else "gray"
            for _, _, d in sub_graph.edges(data=True)
        ]
    else:
        sub_graph = nx.DiGraph(
            [(u, v, d) for u, v, d in sub_graph.edges(data=True) if d["is_loan"] == 0]
        )
        edge_colors = ["gray" for _, _, d in sub_graph.edges(data=True)]

    wrapped_labels = {
        node: wrap_label(node, max_words_per_line=1) for node in sub_graph.nodes()
    }

    edge_weights = [
        max(d["total_fee"] / 1e6, 0.5) for _, _, d in sub_graph.edges(data=True)
    ]

    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(sub_graph, seed=42, k=4.0, iterations=200)

    nx.draw_networkx_nodes(
        sub_graph, pos, node_color="lightblue", node_size=2000, edgecolors="black"
    )
    nx.draw_networkx_edges(
        sub_graph,
        pos,
        edge_color=edge_colors,
        width=edge_weights,
        alpha=0.7,
        arrows=True,
        arrowsize=30,
        connectionstyle="arc3,rad=0.1",  # Adds curvature for clarity
    )
    nx.draw_networkx_labels(
        sub_graph, pos, labels=wrapped_labels, font_size=6, font_weight="bold"
    )

    normal_patch = mpatches.Patch(color="gray", label="Normal Transfer")
    loan_patch = mpatches.Patch(color="red", label="Loan")
    plt.legend(handles=[normal_patch, loan_patch], loc="upper right")

    plt.title(
        "Club Transfer Network (Loans in Red, Bigger Transfers Thicker)", fontsize=14
    )
    plt.axis("off")
    plt.show()


def wrap_label(name, max_words_per_line=1):
    words = re.split(r"\W+", name)
    lines = [
        " ".join(words[i : i + max_words_per_line])
        for i in range(0, len(words), max_words_per_line)
    ]
    return "\n".join(lines)
