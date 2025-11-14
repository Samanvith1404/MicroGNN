import Value,random

class GNN:
    """A class for Graph Neural Network operations with matrices of Value objects."""
    def __init__(self, data):
        self.matrix = data
        self.shape = (len(data), len(data[0]))
        self.linear_weight = []
        self.linear_bias = []

    def __repr__(self):
        return f"GNN(shape={self.shape})"

    @staticmethod
    def adjacency(edges, num_nodes=None):
        """Create adjacency matrix from edges."""
        x0, x1 = edges
        nodes = set(x0 + x1)
        if num_nodes is None:
            num_nodes = max(nodes) + 1 if nodes else 0
        A = [[Value(0) for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            A[i][i] = Value(1)
        for i in range(len(x0)):
            u, v = x0[i], x1[i]
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                A[u][v] = Value(1)
                A[v][u] = Value(1)
        return GNN(A)

    def __matmul__(self, other):
        """Matrix multiplication."""
        result = []
        for i in range(len(self.matrix)):
            row = []
            for j in range(len(other.matrix[0])):
                s = Value(0)
                for k in range(len(self.matrix[0])):
                    s = s + (self.matrix[i][k] * other.matrix[k][j])
                row.append(s)
            result.append(row)
        return GNN(result)

    def tanh(self):
        """Element-wise tanh."""
        temp = []
        for row in self.matrix:
            temp.append([val.tanh() for val in row])
        return GNN(temp)

    def softmax(self):
        """Row-wise softmax with numerical stability."""
        temp = []
        for row in self.matrix:
            max_val = max(val.data for val in row)
            exps = [(val - Value(max_val)).exp() for val in row]
            total = exps[0]
            for e in exps[1:]:
                total = total + e
            soft_row = [e / total for e in exps]
            temp.append(soft_row)
        return GNN(temp)

    def linearNN(self, inps):
        """Initialize linear layers with given architecture."""
        inps = [len(self.matrix[0])] + inps
        if not self.linear_weight:
            for i in range(len(inps)-1):
                a = [[Value(random.uniform(-0.5, 0.5)) for _ in range(inps[i+1])] for _ in range(inps[i])]
                b = [Value(0) for _ in range(inps[i+1])]
                self.linear_weight.append(GNN(a))
                self.linear_bias.append(b)

    def feed_LNN(self):
        """Forward pass through linear layers with bias and tanh."""
        H = GNN([[val for val in row] for row in self.matrix])
        for W, B in zip(self.linear_weight, self.linear_bias):
            H = H @ W
            for i in range(len(H.matrix)):
                for j in range(len(H.matrix[i])):
                    H.matrix[i][j] = H.matrix[i][j] + B[j]
            for i in range(len(H.matrix)):
                for j in range(len(H.matrix[i])):
                    H.matrix[i][j] = H.matrix[i][j].tanh()
        return H

    def show(self):
        """Display matrix values and shape."""
        for row in self.matrix:
            print([round(v.data, 2) for v in row])
        print(f"Shape: {self.shape}")

    @staticmethod
    def MSELoss(pred, target):
        """Mean Squared Error loss."""
        s = Value(0)
        for row_pred, row_target in zip(pred.matrix, target.matrix):
            for p, t in zip(row_pred, row_target):
                diff = p - t
                s = s + diff * diff
        return s