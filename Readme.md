# GNN From Scratch — Minimal Autograd + Graph Neural Network  
*(See Readme.txt inside each folder for deeper explanations)*

A small, readable framework that shows how a Graph Neural Network works from
the ground up. No PyTorch. No TensorFlow. Everything here is built manually,
including a tiny autograd system that performs forward and backward passes.

The goal is simple:  
**Make GNNs understandable, line by line, with no black boxes.**

This project is meant for learning, experiments, and quick prototyping.  
The code is intentionally minimal so you can read and understand every part
without getting lost in abstraction.

---

## 📁 Repository Structure


Main
  ├─ Value/
  │   ├─ Readme.txt
  │   └─ Value.py
  ├─ GNN/
  │   ├─ Readme.txt
  │   └─ GNN.py
  ├─ Samples/
  │   ├─ dataset_loading/
  │   ├─ visualization_code/
  │   └─ Execution_samples/
  └─ overall.ipynb


---

## Quick start

### **1. Clone the repo**
```bash
git clone https://github.com/Samanvith1404/MicroGNN.git
cd MicroGNN
```

## 2) Run a sample execution

Navigate to:

Samples/Execution_samples/


Then run:

python Samples/Execution_samples/run_gnn_test.py


You should see:

# adjacency matrix printed

* GNN layer operations 
* tanh + softmax transformations
* the loss value decreasing
* final updated weights

# This verifies:
* Autograd works
* Matrix multiplications work
* Message passing works
* Linear layers work

Backprop flows correctly

## What this framework includes
# ✔ Minimal Autograd Engine (Value.py)

A tiny autograd engine that supports:
* addition
* multiplication
* matrix multiply
* tanh
* exp
* softmax

backward pass for every operation

Each number is wrapped in a Value object that:
* stores data
* stores gradient
* stores the backward function
* stores the graph of dependencies

```bash
from Value.Value import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + a
c.backward()

print(c.data)
print(a.grad)
print(b.grad)

```

## ✔ Graph Neural Network Module (GNN.py)

Implements message passing using:
```bash
(A @ X @ W).tanh()
```

Supports:
* adjacency matrix construction
* node feature transformations
* softmax-based propagation
* multi-layer message passing

simple feed-forward NN on top of the GNN

```bash
H = (A @ X @ W).tanh()
H = (A @ H @ W2).softmax()
out = H.feed_LNN()
```

## ✔ End-to-End Training Loop

# Inside Samples/Execution_samples/, you will find a working test script where:
* forward pass is computed
* loss = MSELoss(output, target)
* backward pass computes full gradients
* manual gradient descent updates weights

Everything is printed out so you can trace each step.

## Why this project exists

Most GNN tutorials immediately jump to PyTorch or TensorFlow.
This hides the real mechanics behind:

* message passing
* gradient flow
* weight updates
* intermediate representations

Here, everything is visible, simple, and readable.

# Perfect for:
* students
* beginners in ML
* people preparing for interviews
* anyone who wants to understand GNNs deeply
* anyone who loves building things from scratch

## Planned upgrades

* Multi-layer GNN stacks
* More activation functions
* Adam and SGD optimizers
* GraphSAGE, GAT, GCN implementations
* Cleaner dataset utilities

Config-driven training system

## Contributing
This framework is intentionally small and educational.
If you want to suggest improvements, feel free to open a pull request or issue.

## Credits

Built entirely from scratch using Python
by @Samanvith1404.


Email:samanvith1404@gmail.com
