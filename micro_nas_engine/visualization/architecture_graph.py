import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any

from utils.logger import logger

def plot_architecture_graph(arch: Dict[str, Any], save_path: str = "architecture_diagram.png"):
    """
    Generates a node-link diagram of the given architecture using NetworkX and saves it.
    """
    try:
        G = nx.DiGraph()

        nodes = []
        edges = []

        # Add Input Node
        G.add_node("Input", layer_type="input")
        nodes.append("Input")
        prev_node = "Input"

        # Build graph based on type
        if arch["type"] == "mlp":
            for i, hidden_units in enumerate(arch["hidden_units"]):
                node_name = f"FC_{i+1}\nUnits: {hidden_units}"
                G.add_node(node_name, layer_type="fc")
                nodes.append(node_name)
                edges.append((prev_node, node_name))

                # Add Activation
                act_name = f"Act_{i+1}\n{arch['activation'].upper()}"
                G.add_node(act_name, layer_type="act")
                nodes.append(act_name)
                edges.append((node_name, act_name))
                prev_node = act_name

        elif arch["type"] == "cnn":
            # Conv Layers
            for i, (filters, kernel) in enumerate(zip(arch["filters"], arch["kernel_sizes"])):
                node_name = f"Conv_{i+1}\nFilters: {filters}\nKernel: {kernel}x{kernel}"
                G.add_node(node_name, layer_type="conv")
                nodes.append(node_name)
                edges.append((prev_node, node_name))

                # Activation and Pooling
                act_name = f"Act_{i+1}\n{arch['activation'].upper()}"
                G.add_node(act_name, layer_type="act")
                nodes.append(act_name)
                edges.append((node_name, act_name))

                pool_name = f"MaxPool_{i+1}\n2x2"
                G.add_node(pool_name, layer_type="pool")
                nodes.append(pool_name)
                edges.append((act_name, pool_name))
                prev_node = pool_name

            # Flatten Node
            G.add_node("Flatten", layer_type="flatten")
            nodes.append("Flatten")
            edges.append((prev_node, "Flatten"))
            prev_node = "Flatten"

            # FC Layers
            for i, hidden_units in enumerate(arch["hidden_units"]):
                node_name = f"FC_{i+1}\nUnits: {hidden_units}"
                G.add_node(node_name, layer_type="fc")
                nodes.append(node_name)
                edges.append((prev_node, node_name))

                # Add Activation
                act_name = f"Act_FC_{i+1}\n{arch['activation'].upper()}"
                G.add_node(act_name, layer_type="act")
                nodes.append(act_name)
                edges.append((node_name, act_name))
                prev_node = act_name

        # Add Output Node
        G.add_node("Output", layer_type="output")
        nodes.append("Output")
        edges.append((prev_node, "Output"))

        # Add edges to graph
        G.add_edges_from(edges)

        # Draw
        plt.figure(figsize=(max(4, len(nodes) * 1.5), 6))

        # Spring layout works, but for deep nets a simple vertical/horizontal layout is better
        # We'll calculate a simple linear layout
        pos = {node: (i, 0) for i, node in enumerate(nodes)}

        # Color nodes by layer type
        color_map = []
        for node in G:
            l_type = G.nodes[node].get("layer_type", "other")
            if l_type == "input": color_map.append("lightgreen")
            elif l_type == "output": color_map.append("lightcoral")
            elif l_type == "conv": color_map.append("skyblue")
            elif l_type == "fc": color_map.append("khaki")
            elif l_type == "act": color_map.append("thistle")
            elif l_type == "pool": color_map.append("lightgrey")
            else: color_map.append("white")

        nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=3000,
                font_size=9, font_weight="bold", arrows=True, arrowsize=20, edge_color="gray")

        plt.title(f"Best Discovered Architecture Diagram\nType: {arch['type'].upper()} | Depth: {arch['depth']} | Dropout: {arch['dropout']}")
        plt.axis("off")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Architecture graph plot to {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate architecture graph: {e}")
