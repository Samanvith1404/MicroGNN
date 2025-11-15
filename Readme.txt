# Graph Neural Network Framework (Custom Autograd + GNN)

This project is a small, self-contained framework that lets you build and train
Graph Neural Networks from scratch. Everything here runs without PyTorch,
TensorFlow, or any external deep-learning library. The idea was to learn how a
GNN actually works under the hood by writing every part manually.

There are two main building blocks:

---

## 1. `Value.py` — tiny autograd engine

This file handles all the math and gradient tracking. Every number in the
network is wrapped in a `Value` object, which remembers how it was created.  
When `.backward()` is called, it walks back through all operations and fills
in gradients. Its heavily inspired by the simplicity of micrograd.

It supports:
- addition, subtraction, multiplication
- exponentials and tanh
- power operations
- gradient propagation through a dynamic graph

This is what makes training possible.

---

## 2. `GNN.py` — basic Graph Neural Network

This file contains the actual model logic. It can:
- build adjacency matrices
- do matrix multiplications using the autograd engine
- apply tanh and softmax
- set up simple linear layers
- run a full forward pass
- compute loss using MSE

The whole GNN is built on top of `Value`, so every part of the model is fully
differentiable.

---

## What this framework is for

The goal isn’t to compete with big libraries. This is more of a learning and
exploration tool — a way to understand:
- how autograd works internally  
- how message passing in GNNs actually happens  
- how gradients flow through graph operations  
- how weight updates work without any magic happening behind the scenes  

Everything here is intentionally small and easy to read.  
If you want to extend the model or experiment with new ideas, this setup makes
it straightforward.

---

## Example: running a simple experiment

There is a small script (`run_gnn_test.py`) that trains the GNN on one graph
sample. It just checks that:
- forward pass works  
- backprop works  
- weights update correctly  
- loss decreases  

It’s not meant to be a polished training loop — just a sanity check that the
entire pipeline behaves as expected.

---

## Dataset

The code assumes that each graph provides:
- `edge_index` — list of edges  
- `x` — node features  
- `y` — target labels  

You can plug in your own dataset as long as it follows this format.

---

## Why build everything manually?

Mostly to understand things better. Writing your own autograd engine and GNN
forces you to appreciate what libraries like PyTorch do behind the scenes.  
Once you see gradients flowing through your own code, everything else starts to
make a lot more sense.

---

## Future ideas (optional)

A few directions this framework could grow into:
- adding more activation functions  
- supporting batching  
- experimenting with different GNN variants  
- adding training utilities (optimizers, schedulers, etc.)  

For now, it’s a compact and transparent implementation that’s easy to play
with and modify.

---

If you want to dig into the code, start with `Value.py` — understanding that
file makes the rest of the project very easy to follow.
