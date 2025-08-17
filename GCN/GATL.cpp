// GATLayer.cpp

#include "GATL.h"
#include <random>
#include <cmath>
#include <algorithm>

// Constructor with Xavier initialization
GATLayer::GATLayer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {
    W.resize(input_dim, vector<float>(output_dim));
    a.resize(2 * output_dim);

    float limit = sqrt(6.0f / (input_dim + output_dim));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, limit);

    for (int i = 0; i < input_dim; i++)
        for (int j = 0; j < output_dim; j++)
            W[i][j] = dis(gen);

    for (int i = 0; i < 2 * output_dim; i++)
        a[i] = dis(gen);
}

// ReLU activation
float GATLayer::relu(float x) {
    return max(0.0f, x);
}

// Leaky ReLU activation
float GATLayer::leaky_relu(float x, float alpha) {
    return (x > 0) ? x : alpha * x;
}

// Linear transformation for a single node
vector<float> GATLayer::linear_transform(const vector<float>& features) {
    vector<float> z(output_dim, 0.0f);
    for (int o = 0; o < output_dim; o++)
        for (int d = 0; d < input_dim; d++)
            z[o] += features[d] * W[d][o];
    return z;
}

// Compute attention score e_ij using attention vector 'a'
float GATLayer::compute_attention_score(const vector<float>& z_i, const vector<float>& z_j) {
    float score = 0.0f;
    for (int o = 0; o < output_dim; o++) {
        score += a[o] * z_i[o] + a[o + output_dim] * z_j[o];
    }
    return leaky_relu(score);
}

// Stable softmax computation
vector<float> GATLayer::softmax(const vector<float>& scores) {
    float max_val = *max_element(scores.begin(), scores.end());
    float sum_exp = 0.0f;
    vector<float> exp_scores(scores.size());
    for (size_t i = 0; i < scores.size(); i++) {
        exp_scores[i] = exp(scores[i] - max_val);
        sum_exp += exp_scores[i];
    }
    if(sum_exp!=0.0f) {
        for (float &val : exp_scores) {
            val /= sum_exp;
        }
    }
    return exp_scores;
}

// Forward pass
vector<vector<float>> GATLayer::forward(
    const vector<vector<float>>& node_features,
    const vector<vector<int>>& adjacency_list
) {
    int n_nodes = node_features.size();
    vector<vector<float>> updated_features(n_nodes, vector<float>(output_dim, 0.0f));
    vector<vector<float>> z(n_nodes);

    // Step 1: Linear transform each node's features
    for (int i = 0; i < n_nodes; i++) {
        z[i] = linear_transform(node_features[i]);
    }

    // Step 2: Compute attention and aggregate
    for (int i = 0; i < n_nodes; i++) {
        vector<int> neighbors = adjacency_list[i];
        neighbors.push_back(i); // self-loop

        vector<float> e_ij(neighbors.size());
        for (size_t idx = 0; idx < neighbors.size(); idx++) {
            int j = neighbors[idx];
            e_ij[idx] = compute_attention_score(z[i], z[j]);
        }

        vector<float> alpha_ij = softmax(e_ij);

        for (int o = 0; o < output_dim; o++) {
            float agg = 0.0f;
            for (size_t idx = 0; idx < neighbors.size(); idx++) {
                int j = neighbors[idx];
                agg += alpha_ij[idx] * z[j][o];
            }
            updated_features[i][o] = relu(agg); // ReLU activation
        }
    }

    return updated_features;
}
