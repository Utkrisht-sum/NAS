import matplotlib.pyplot as plt
import io
import base64
import networkx as nx

def plot_pareto_frontier(history):
    if not history:
        return None

    accs = [h['accuracy'] for h in history]
    costs = [h['cost'] for h in history]

    plt.figure(figsize=(6, 4))
    plt.scatter(costs, accs, c='blue', alpha=0.5, label='Evaluated Models')

    sorted_idx = sorted(range(len(costs)), key=lambda k: costs[k])
    pareto_costs = []
    pareto_accs = []

    max_acc = -1
    for i in sorted_idx:
        if accs[i] > max_acc:
            pareto_costs.append(costs[i])
            pareto_accs.append(accs[i])
            max_acc = accs[i]

    plt.plot(pareto_costs, pareto_accs, c='red', marker='o', linestyle='-', label='Pareto Frontier')
    plt.xlabel('Compute Cost (Approximate)')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('NAS Pareto Frontier')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64

def plot_nas_timeline(history):
    if not history:
        return None

    # Assume history is chronologically appended
    accs = [h['accuracy'] for h in history]
    evals = list(range(1, len(history) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(evals, accs, c='green', marker='x', linestyle='--', alpha=0.7)
    plt.xlabel('Evaluation Number (Timeline)')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Architecture Evolution Timeline')
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64

def plot_architecture_dag(arch_config, task_type):
    # Very simple DAG visualization using networkx for CNNs
    if task_type != "Image" or "cells" not in arch_config:
        return None

    G = nx.DiGraph()
    cells = arch_config["cells"]

    G.add_node("Input")
    prev_node = "Input"

    for c_idx, cell in enumerate(cells):
        cell_name = f"Cell_{c_idx}"
        G.add_node(cell_name)
        G.add_edge(prev_node, cell_name, label="stride")

        # Add internal nodes (simplified for visualization)
        for n_idx, node in enumerate(cell):
            node_name = f"C{c_idx}_N{n_idx}"
            G.add_node(node_name)

            op1 = node[0][0] if isinstance(node[0], tuple) else str(node[0])
            G.add_edge(cell_name, node_name, label=op1)

        prev_node = cell_name

    G.add_node("Output")
    G.add_edge(prev_node, "Output")

    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=8, font_weight='bold', arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title('Best Architecture DAG (Simplified)')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64
