// BaseLayer.h
#pragma once
#include <vector>
using namespace std;

// BaseLayer provides a standard interface for all GNN layers (GAT, GCN, GraphSAGE, etc.)
class BaseLayer {
public:
    virtual ~BaseLayer() {}

    // Forward pass interface to be overridden by all derived GNN layers
    virtual vector<vector<float>> forward(
        const vector<vector<float>>& node_features,
        const vector<vector<int>>& adjacency_list
    ) = 0;
};
