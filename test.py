import GNN,Value,random

A = GNN.adjacency([[0,1,2],[1,2,3]], num_nodes=5)
X_data = [[6,0],[1,0],[1,0],[1,0],[1,0]]
X = GNN([[Value(v) for v in row] for row in X_data])
W_data = [[0.5,0.7],[0.9,0.1]]
W = GNN([[Value(random.random() * random.choice([1, -1])) for v in row] for row in W_data])
W2 = GNN([[Value(random.random() * random.choice([1, -1])) for v in row] for row in W_data])
target_data = [[1],[0],[0],[1],[0]]
target = GNN([[Value(v) for v in row] for row in target_data])
H = (A @ X @ W).tanh()
H.linearNN([5,1])
for _ in range(500):
    H = (A @ X @ W).tanh()
    H = (A @ H @ W2).softmax()
    out = H.feed_LNN()
    loss = GNN.MSELoss(out, target)
    print("Loss:", loss.data)
    loss.backward()
    lr = 0.01
    for row in W.matrix:
        for val in row:
            val.data -= lr * val.grad
            val.grad = 0.0
    for row in W2.matrix:
        for val in row:
            val.data -= lr * val.grad
            val.grad = 0.0
    for lw in H.linear_weight:
        for row in lw.matrix:
            for val in row:
                val.data -= lr * val.grad
                val.grad = 0.0
    for bias in H.linear_bias:
        for val in bias:
            val.data -= lr * val.grad
            val.grad = 0.0
print("\nAfter update - W:")
for i in W.matrix:
    for j in i:
        print(round(j.data, 4), end=' ')
    print()