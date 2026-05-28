# Non-Linearity

A small collection of **Jupyter notebooks** exploring nonlinear dynamical systems and PDEs, with a focus on modern operator-learning / physics-informed approaches (e.g., **PINNs** and **DeepONet**) implemented in Python.

## Repository contents

### Notebooks

- `L01_Simple System.ipynb` — Intro / simple nonlinear system experiments.
- `Simple_Harmonic_Motion_PINN.ipynb` — Physics-Informed Neural Network (PINN) for simple harmonic motion.
- `Heat_Equation.ipynb` — Heat equation experiments.
- `Diffusion_Equation_Pinn.ipynb` — PINN for the diffusion equation.
- `Burger_Equation_PINN.ipynb` — PINN for Burgers’ equation.
- `Burger_equation_DeepOnet.ipynb` — DeepONet approach for Burgers’ equation.
- `test.ipynb` — Scratch / quick tests.

### Python utilities

- `pp.py` — Helper/utility code used by one or more notebooks.

## Getting started

### 1) Clone the repository

```bash
git clone https://github.com/sadi2003q/Non-Linearity.git
cd Non-Linearity
```

### 2) Create an environment (recommended)

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

### 3) Install dependencies

This repo doesn’t currently include a pinned `requirements.txt`. Start with:

```bash
pip install jupyter numpy matplotlib scipy
```

If a notebook imports TensorFlow or PyTorch, install the one it expects (the import cells in each notebook are the source of truth):

```bash
# one of the following, depending on the notebook
pip install tensorflow
# or
pip install torch
```

### 4) Run Jupyter

```bash
jupyter notebook
```

Open any `.ipynb` from the Jupyter UI.

## Notes

- If you’d like, I can also add a `requirements.txt` (or `environment.yml`) by scanning the notebooks for imports.
- If you have preferred citation/style for the math background (PINNs/DeepONet), tell me and I’ll add a References section.
