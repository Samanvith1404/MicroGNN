What this file is?

GNN.py contains the main Graph Neural Network code.
It handles things like building adjacency matrices, doing matrix multiplications, applying activations, running a small linear neural network, and calculating loss.

Everything inside this file uses the Value class under the hood, so gradients flow naturally through the whole graph.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Why I wrote it?

I wanted a minimal GNN implementation that doesn’t rely on huge frameworks.
This file lets me experiment with graph-based learning using simple, readable code.

The idea was to keep everything small enough to tweak easily but still complete enough to run a forward and backward pass like a real GNN model.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

What’s inside??


1. Adjacency builder

adjacency() creates an adjacency matrix (with self-loops) from edge lists.
This gives a clean starting point for message passing.

2. Matrix multiplication

The @ operator is implemented so that two GNN objects can multiply like regular matrices.
This is how information flows across nodes in the graph.

3. Activations

There are two basic activations:

tanh() — used inside linear layers

softmax() — done row-wise

Both of these work with the gradient engine.

4. Linear layers

linearNN() sets up fully connected layers with random weights and zero biases.
These are used after the graph message passing step.

5. Forward pass

feed_LNN() runs each linear layer, adds biases, and applies tanh.
This is the main “neural network” part of the model.

6. Loss

There’s a simple MSE loss function for training.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

How it all fits together??

1)You create the adjacency matrix.
2)You pass node features.
3)The GNN multiplies them together → message passing.
4)The linear layers refine the representations.
5)You compare the output with the target.
6)Value.backward() computes all the gradients.

That’s the entire workflow of the model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
