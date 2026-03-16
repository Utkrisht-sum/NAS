import matplotlib.pyplot as plt
import io
import base64

def plot_pareto_frontier(history):
    """
    Returns a base64 encoded png of the Pareto frontier plot.
    """
    if not history:
        return None

    accs = [h['accuracy'] for h in history]
    costs = [h['cost'] for h in history]

    plt.figure(figsize=(6, 4))
    plt.scatter(costs, accs, c='blue', alpha=0.5, label='Evaluated Models')

    # Simple pareto frontier calculation for plotting
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
    plt.title('NAS Search Results')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return image_base64
