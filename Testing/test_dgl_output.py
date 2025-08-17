import torch
import dgl
import pytest
from dgl_output_main import GCNLayer

# Helper functions to mimic the script's logic for testing

def read_graph_data(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    num_nodes, num_features = map(int, lines[0].split())
    node_features = []
    for i in range(1, 1 + num_nodes):
        parts = lines[i].split()
        feats = list(map(float, parts[1:]))
        node_features.append(feats)
    node_features = torch.tensor(node_features, dtype=torch.float32)
    src = []
    dst = []
    for i in range(1 + num_nodes, len(lines)):
        u, v = map(int, lines[i].split())
        src.append(u)
        dst.append(v)
    return node_features, (src, dst), num_nodes, num_features

def build_dgl_graph(node_features, edges, num_nodes):
    src, dst = edges
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['feat'] = node_features
    return g

def run_gcn_layer(g, node_features, in_dim, out_dim):
    gcn = GCNLayer(in_dim, out_dim)
    with torch.no_grad():
        out = gcn(g, node_features)
    return out

def test_graph_loading():
    node_features, edges, num_nodes, num_features = read_graph_data('graph_data.txt')
    assert node_features.shape[0] == num_nodes
    assert node_features.shape[1] == num_features
    assert len(edges[0]) == 5  # 5 edges in the sample file
    assert len(edges[1]) == 5

def test_gcn_output_shape():
    node_features, edges, num_nodes, num_features = read_graph_data('graph_data.txt')
    g = build_dgl_graph(node_features, edges, num_nodes)
    out_dim = 3
    out = run_gcn_layer(g, node_features, num_features, out_dim)
    assert out.shape == (num_nodes, out_dim) 
