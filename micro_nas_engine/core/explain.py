def generate_explainability_report(best_arch, history, failures):
    """
    Generates a textual explanation for why the best architecture was chosen
    and summarizes the search process.
    """
    if not best_arch:
        return "No architecture found."

    total_evaluated = len(history)
    total_failures = len(failures)

    best_acc = best_arch.get('accuracy', 0)
    best_cost = best_arch.get('cost', 0)
    best_mem = best_arch.get('memory', 0)
    dna = best_arch.get('config', {}).get('dna', 'Unknown')

    # Analyze alternatives
    better_acc_but_higher_cost = [h for h in history if h['accuracy'] > best_acc and h['cost'] > best_cost]
    lower_cost_but_worse_acc = [h for h in history if h['cost'] < best_cost and h['accuracy'] < best_acc]

    report = f"### Architecture Explainability Report\n\n"
    report += f"**Selected Architecture DNA:** `{dna}`\n"
    report += f"- **Validation Accuracy:** {best_acc:.2f}%\n"
    report += f"- **Compute Cost (Proxy):** {best_cost:.0f}\n"
    report += f"- **Estimated Memory:** {best_mem:.1f} MB\n\n"

    report += f"**Why was this chosen?**\n"
    report += f"This architecture provided the optimal trade-off in our hardware-aware fitness function `F(A) = α*acc - β*cost - γ*lat - δ*mem`.\n"

    if better_acc_but_higher_cost:
        report += f"- There were {len(better_acc_but_higher_cost)} architectures with higher accuracy, but they were penalized for excessive compute cost or memory usage.\n"
    if lower_cost_but_worse_acc:
        report += f"- There were {len(lower_cost_but_worse_acc)} cheaper architectures, but they sacrificed too much accuracy.\n\n"

    report += f"**Search Summary:**\n"
    report += f"- Total Evaluated: {total_evaluated}\n"
    report += f"- Total Failures Avoided/Repaired: {total_failures}\n"

    if total_failures > 0:
        report += "\n**Common Failure Patterns Avoided:**\n"
        # Just grab unique reasons
        reasons = list(set([f['reason'] for f in failures]))
        for r in reasons[:3]:
            report += f"- {r}\n"

    return report
