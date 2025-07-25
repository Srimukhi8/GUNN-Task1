import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# 1. Read the graph_data.txt file
filename = 'graph_data.txt'
with open(filename, 'r') as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# 2. Parse header
num_nodes, num_features = map(int, lines[0].split())

# 3. Parse node features
node_features = []
for i in range(1, 1 + num_nodes):
    parts = lines[i].split()
    # parts[0] is node_id, rest are features
    feats = list(map(float, parts[1:]))
    node_features.append(feats)
node_features = torch.tensor(node_features, dtype=torch.float32)

# 4. Parse edges
src = []
dst = []
unique_edges = []
for i in range(1 + num_nodes, len(lines)):
    u, v = map(int, lines[i].split())
    src.append(u)
    dst.append(v)
    src.append(v)  
    dst.append(u)  
    unique_edges.append((u, v))

# 5. Build DGL graph
g = dgl.graph((src, dst), num_nodes=num_nodes)
g.ndata['feat'] = node_features

# 6. Define a simple GCN layer
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
    def forward(self, g, features):
        with g.local_scope():
            g.ndata['h'] = features
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h'))
            h = g.ndata['h']
            return self.linear(h)

# 7. Prompt for output dimension
out_dim = int(input('Enter output feature dimension: '))
gcn = GCNLayer(num_features, out_dim)

# Manually set weights and biases
with torch.no_grad():
    gcn.linear.weight.fill_(1.0)  # Set all weights to 1.0
    gcn.linear.bias.fill_(0.0)    # Set all biases to 0.0

# 8. Forward pass
with torch.no_grad():
    node_out = gcn(g, node_features)

print('=== Node-Level ===')
for node in node_out:
    print(' '.join(f'{v:.2f}' for v in node.tolist()))
print()

# 9. Node scores (sum of features)
node_scores = node_out.sum(dim=1)

# 10. Edge-level outputs
def to_edge_scores(node_scores, g):
    src, dst = g.edges()
    return (node_scores[src] * node_scores[dst]).tolist()

def to_edge_binary(node_scores, g, threshold=0.5):
    scores = to_edge_scores(node_scores, g)
    return [int(s > threshold) for s in scores]

def to_graph_score(node_scores):
    return node_scores.mean().item()

def to_graph_binary(node_scores, threshold=0.5):
    return int(to_graph_score(node_scores) > threshold)

edge_scores = to_edge_scores(node_scores, g)
edge_truth = to_edge_binary(node_scores, g, 0.5)
graph_score = to_graph_score(node_scores)
graph_truth = to_graph_binary(node_scores, 0.5)

print('=== Graph-Level ===')
print(f'Score = {graph_score:.2f}\n')

print('=== Edge-Level ===')
for i, (u, v) in enumerate(unique_edges):
    score = node_scores[u] * node_scores[v]
    print(f'Edge {i} ({u}, {v}) | score = {score:.2f}')

"""
Enter output feature dimension: 4
=== Node-Level ===
1.45 1.45 1.45 1.45
2.10 2.10 2.10 2.10
0.75 0.75 0.75 0.75
2.15 2.15 2.15 2.15
0.85 0.85 0.85 0.85

=== Graph-Level ===
Score = 5.84

=== Edge-Level ===
Edge 0 | score = 48.72
Edge 1 | score = 25.20
Edge 2 | score = 25.80
Edge 3 | score = 29.24
Edge 4 | score = 19.72
"""

"""
Custom Implementation Output:
Enter output feature dimension: 4
=== Node Features (post-GCN) ===
Node 0: 1.45 1.45 1.45 1.45
Node 1: 2.1 2.1 2.1 2.1
Node 2: 0.75 0.75 0.75 0.75
Node 3: 2.15 2.15 2.15 2.15 
Node 4: 0.85 0.85 0.85 0.85

=== GraphΓÇÉLevel ===
Score = 5.84

=== EdgeΓÇÉLevel ===
Edge 0 | score = 48.72
Edge 1 | score = 19.72
Edge 2 | score = 25.2
Edge 3 | score = 25.8
Edge 4 | score = 29.24
"""