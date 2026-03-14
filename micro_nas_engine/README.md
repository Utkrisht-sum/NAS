# MICRONAS ENGINE

MICRONAS ENGINE is a fully functional local software application designed for automatic model discovery, training, and multi-objective optimization. It uses a custom Evolutionary Neural Architecture Search (NAS) approach to find neural networks that maximize accuracy while minimizing compute cost (parameters & FLOPs).

## Features

- **Neural Architecture Search (NAS)**: Automatic discovery of MLP and CNN architectures.
- **Auto Dataset Analysis**: Determines task type (classification/regression) and inputs automatically.
- **Multi-Objective Optimization**: Computes a Pareto frontier balancing Accuracy and Compute Cost.
- **Evolutionary Search**: Uses crossover, mutation, and elitism to evolve architectures.
- **Natural Language Parsing**: Analyzes user prompts for keyword heuristics (e.g., "efficient", "accurate") to adjust fitness weights.
- **Self-Learning Memory**: Saves successful architectures and uses them to seed future populations.
- **Local Execution**: Runs entirely on local hardware with automatic CPU/CUDA selection.
- **Visualizations**: Automatically generates Architecture Diagrams (NetworkX) and Pareto Plots (Matplotlib).
- **Dark Mode GUI**: Easy-to-use Qt interface powered by PySide6.

## Installation Requirements

- Ubuntu Linux 22.04+ (or equivalent modern OS)
- Python 3.9+
- Local compute hardware (CUDA GPU recommended but not required)

## Installation

1. Clone or download this repository.
2. Open a terminal and navigate to the project root `micro_nas_engine/`.
3. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution

To launch the GUI and start the MICRONAS ENGINE:

```bash
python main.py
```

### Initial Pre-Training
Upon first launch, the software will automatically download standard small datasets (MNIST, CIFAR-10, Iris) in the background to build its initial base knowledge.

## How to Use

1. **Select Dataset**: Use the top-left panel to pick a predefined dataset, provide a CSV URL, or browse your local file system for a CSV or Image Directory.
2. **Provide Prompt**: Enter a natural language goal. For example: *"Train the best model for this dataset while keeping it small."* The engine will parse words like "small" to penalize heavy architectures more aggressively.
3. **Configure NAS**: Adjust population size, total generations, and compute budget (max parameters).
4. **Start Search**: Click Start. The engine will evaluate architectures, display logs in the bottom-right panel, and ultimately output the Pareto frontier and best overall architecture diagram.

## Project Structure

- `core/`: NAS engine loop, evolutionary algorithms, and search space definitions.
- `datasets/`: Dataset downloading, loading, and automatic analysis.
- `gui/`: PySide6 graphical user interface modules.
- `knowledge/`: Self-learning architecture memory persistence.
- `metrics/`: Compute cost estimation and fitness functions.
- `models/`: Architecture encoding and dynamic PyTorch model building.
- `training/`: PyTorch training and evaluation loops.
- `visualization/`: Matplotlib and NetworkX graph plotting logic.
- `utils/`: Configuration, logging, and seed utilities.

---
*Created as an expert AI systems architecture solution.*
