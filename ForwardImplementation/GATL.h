// GATL.h
#pragma once
#include "BaseLayer.h"
#include <vector>
using namespace std;

// Implements a layer of Graph Attention Network(GAT)
// It takes into account the importance of each neighbour also in aggregation.
// It uses self-attention mechanism on graphs to compute this importance
class GATLayer : public BaseLayer {
public:
    int input_dim, output_dim;  // Input and output dimension
    vector<vector<float>> W;    // Weight matrix for linear transformation
    vector<float> a;            // Attention vector used for computing attention coefficients

    // Constructor initializes the GAT layer with input and output dimensions
    // and performs Xavier initialization for weights and attention parameters.
    GATLayer(int input_dim, int output_dim);

    // Forward pass computes the updated node features based on attention mechanism.
    // It projects input features, computes attention scores with neighbours, applies softmax,
    // aggregates neighbour features weighted by attention, and applies ReLU.
    vector<vector<float>> forward(
        const vector<vector<float>>& node_features, // node-feature matrix:[number of nodes][input_dim]
        const vector<vector<int>>& adjacency_list   // represents the graph
    ) override;

private:
    // Applies ReLU activation to a single float value
    float relu(float x);

    // Applies LeakyReLU activation with a configurable alpha slope for negative inputs.
    float leaky_relu(float x, float alpha = 0.2f);

    // Applies weight matrix to a single node's feature vector to transform feature vector of size output_dim.
    vector<float> linear_transform(
        const vector<float>& features // Input feature vector of a node
    );

    // Computes attention score (unnormalised) for node i and node j
    // using attention vector applied to concatenation of projected features of node i and node j
    float compute_attention_score(
        const vector<float>& z_i, // feature vector of node i
        const vector<float>& z_j  // feature vector of node j (neighbour)
    );

    // Applies softmax function to a vector of unnormalised attention scores
    // returns a vector of normalised attention coefficients
    vector<float> softmax(
        const vector<float>& scores  // Unnormalised attention scores for a node and its neighbours
    );
};
