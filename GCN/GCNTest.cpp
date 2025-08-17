// GCNL.cpp

#include "GCNTest.h"
#include <random>
#include <algorithm>
#include <cmath>

// Xavier Initialization
GCNTestLayer::GCNTestLayer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {
    weight_matrix.resize(input_dim, vector<float>(output_dim,1.0f));
}

// ReLU activation
float GCNTestLayer::relu(float x) {
    return max(0.0f, x);
}

// Aggregates normalized neighbor features for a node
vector<float> GCNTestLayer::aggregate_neighbors(
    int node,
    const vector<vector<float>>& node_features,
    const vector<vector<int>>& adjacency_list,
    const vector<int>& degrees
) {
    vector<float> aggregated(input_dim, 0.0f);
    for (int neighbor : adjacency_list[node]) {
        float normalization = sqrt(degrees[node] * degrees[neighbor]);
        if (normalization != 0.0f) {
            for (int d = 0; d < input_dim; d++) {
                aggregated[d] += node_features[neighbor][d] / normalization;
            }
        }
    }
    return aggregated;
}

// Applies weight matrix for a given output dimension
float GCNTestLayer::linear_transform(
    const vector<float>& aggregated_features,
    int output_index
) {
    float val = 0.0f;
    for (int d = 0; d < input_dim; d++) {
        val += aggregated_features[d] * weight_matrix[d][output_index];
    }
    return val;
}

// Forward pass for GCN Layer
vector<vector<float>> GCNTestLayer::forward(
    const vector<vector<float>>& node_features,
    const vector<vector<int>>& adjacency_list
) {
    int n_nodes = node_features.size();
    vector<vector<float>> updated_features(n_nodes, vector<float>(output_dim, 0.0f));

    // Precompute degrees
    vector<int> degrees(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        degrees[i] = adjacency_list[i].size();
    }

    for (int i = 0; i < n_nodes; i++) {
        vector<float> aggregated = aggregate_neighbors(i, node_features, adjacency_list, degrees);
        for (int o = 0; o < output_dim; o++) {
            float val = linear_transform(aggregated, o);
            updated_features[i][o] = relu(val);
        }
    }

    return updated_features;
}
