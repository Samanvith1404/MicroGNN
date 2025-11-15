# This is a small test script I wrote mainly to check whether all parts of the
# GNN pipeline (adjacency building, message passing, autograd, linear layers, etc.)
# work together without errors. It’s not meant to be a perfect model — just a
# sanity check that everything runs end-to-end.

import random
from GNN import GNN, MSELoss
from Value import Value

def run_gnn(dataset):
    # Build the adjacency matrix for the sample graph.
    # I'm using num_nodes=5 here since the dataset sample has 5 nodes.
    A = GNN.adjacency(dataset.edge_index.tolist(), num_nodes=5)

    # Pull out node features. I'm just picking two columns to keep things simple.
    x = dataset.x
    X_data = [[i, j] for i, j in zip(x[:, 5], x[:, 7])]

    # Convert node features into Value objects (needed for autograd).
    X = GNN([[Value(v) for v in row] for row in X_data])

    # Random initialization for four different GNN weight matrices.
    # The exact values don't matter — the goal is just to test gradient flow.
    W_data = [[0.5, 0.7], [0.9, 0.1]]
    W = GNN([[Value(random.random() * random.choice([1, -1])) for v in row] for row in W_data])
    W2 = GNN([[Value(random.random() * random.choice([1, -1])) for v in row] for row in W_data])
    W3 = GNN([[Value(random.random() * random.choice([1, -1])) for v in row] for row in W_data])
    W4 = GNN([[Value(random.random() * random.choice([1, -1])) for v in row] for row in W_data])

    # Print the initial weights so I can see if random initialization looks okay.
    for i in W.matrix:
        for j in i:
            print(j.data, end=' ')
        print()

    # Prepare target values from the dataset. Again, converting everything to Value().
    target_data = dataset.y
    target = GNN([[Value(v) for v in row] for row in target_data])

    # First pass just to set up the linear layers.
    H = (A @ X @ W).tanh()
    H.linearNN([5, 1])  # Simple 5 → 1 network just to test something trainable.

    # Standard small training loop — 100 iterations of forward + backward + SGD.
    for _ in range(100):

        # 4-layer message-passing stack.
        # This isn't meant to mimic a real architecture; it's mainly a stress-test
        # to make sure gradients flow through multiple GNN layers.
        H = (A @ X @ W).tanh()
        H = (A @ H @ W2).softmax()
        H = (A @ H @ W3).tanh()
        H = (A @ H @ W4).softmax()

        # Run the linear NN layers we created earlier.
        out = H.feed_LNN()

        # Compute loss (MSE). If everything is correct, this should go down slowly.
        loss = MSELoss(out, target)
        print("Loss:", loss.data)

        # Backprop through the whole graph. This relies on our custom autograd engine.
        loss.backward()

        # Learning rate for weight updates.
        lr = 0.1

        # Manual SGD update for W and W2.
        for row in W.matrix:
            for val in row:
                val.data -= lr * val.grad
                val.grad = 0.0

        for row in W2.matrix:
            for val in row:
                val.data -= lr * val.grad
                val.grad = 0.0

        # Update weights in the linear layers.
        for lw in H.linear_weight:
            for row in lw.matrix:
                for val in row:
                    val.data -= lr * val.grad
                    val.grad = 0.0

        # Update biases too.
        for bias in H.linear_bias:
            for val in bias:
                val.data -= lr * val.grad
                val.grad = 0.0

    # After finishing training, print out the updated W to see how much it changed.
    print("\nAfter update - W:")
    for i in W.matrix:
        for j in i:
            print(round(j.data, 4), end=' ')
        print()


# Running this on one graph example from the dataset.
# This helps confirm that everything works as expected.
# (No batching — just a simple functional test.)
# Example: run_gnn(dataset[0])
