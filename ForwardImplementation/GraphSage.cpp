// GraphSage.cpp

#include "GraphSage.h"
#include <random>
#include <cmath>
#include <algorithm>

// Xavier Initialization
GraphSAGELayer::GraphSAGELayer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {
    weight_matrix.resize(2 * input_dim, vector<float>(output_dim));
    float limit = sqrt(6.0f / (2 * input_dim + output_dim));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, limit);

    for (int i = 0; i < 2 * input_dim; i++)
        for (int j = 0; j < output_dim; j++)
            weight_matrix[i][j] = dis(gen);
}

// ReLU activation
float GraphSAGELayer::relu(float x) {
    return max(0.0f, x);
}

// Mean aggregation of neighbor features
vector<float> GraphSAGELayer::aggregate_neighbors_mean(
    int node,
    const vector<vector<float>>& node_features,
    const vector<vector<int>>& adjacency_list
) {
    vector<float> neighbor_agg(input_dim, 0.0f);
    int neighbor_count = adjacency_list[node].size();

    if (neighbor_count > 0) {
        for (int neighbor : adjacency_list[node]) {
            for (int d = 0; d < input_dim; d++) {
                neighbor_agg[d] += node_features[neighbor][d];
            }
        }
        for (int d = 0; d < input_dim; d++) {
            neighbor_agg[d] /= neighbor_count; // mean aggregation
        }
    }
    return neighbor_agg;
}

// Concatenate own features with neighbor aggregation
vector<float> GraphSAGELayer::concatenate_self_and_neighbors(
    const vector<float>& self_features,
    const vector<float>& neighbor_features
) {
    vector<float> concat_features(2 * input_dim);
    for (int d = 0; d < input_dim; d++) {
        concat_features[d] = self_features[d];
        concat_features[d + input_dim] = neighbor_features[d];
    }
    return concat_features;
}

// Linear transformation for a given output index
float GraphSAGELayer::linear_transform(
    const vector<float>& concat_features,
    int output_index
) {
    float val = 0.0f;
    for (int d = 0; d < 2 * input_dim; d++) {
        val += concat_features[d] * weight_matrix[d][output_index];
    }
    return val;
}

// Forward pass for GraphSAGE layer
vector<vector<float>> GraphSAGELayer::forward(
    const vector<vector<float>>& node_features,
    const vector<vector<int>>& adjacency_list
) {
    int n_nodes = node_features.size();
    vector<vector<float>> updated_features(n_nodes, vector<float>(output_dim, 0.0f));

    for (int i = 0; i < n_nodes; i++) {
        vector<float> neighbor_agg = aggregate_neighbors_mean(i, node_features, adjacency_list);
        vector<float> concat_features = concatenate_self_and_neighbors(node_features[i], neighbor_agg);

        for (int o = 0; o < output_dim; o++) {
            float val = linear_transform(concat_features, o);
            updated_features[i][o] = relu(val);
        }
    }

    return updated_features;
}
