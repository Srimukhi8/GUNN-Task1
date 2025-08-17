// GCNL.h

#pragma once
#include "BaseLayer.h"
#include <vector>
using namespace std;

// GCNTestLayer implements Graph Convolution Neural Network
// performs feature aggregation ONLY from neighbours and then does linear transformation.
class GCNTestLayer : public BaseLayer {
public:
    // This constructor initialises the Layer with input and output dimensions
    // performs Xavier initialisation of weight matrix.
    GCNTestLayer(int input_dim, int output_dim);

    // forward pass computes the updated node features 
    // using the input features and the adjacency list
    vector<vector<float>> forward(
        const vector<vector<float>>& node_features, // feature matrix-[number of nodes][input_dim]
        const vector<vector<int>>& adjacency_list // represents graph structure
    ) override;

private:
    int input_dim;              // dimension of input features
    int output_dim;             // dimension of output features
    vector<vector<float>> weight_matrix; // weight matrix of shape [input_dim][output_dim]
    
    float relu(float x); // Applies ReLU function to a single value (Activation function)

    // Aggregates normalised neighbour features for a given node.
    // Each neighbour's features are first scaled down by the inverse of
    // the product of degrees of node and the neighbour, ensuring normalisation.
    vector<float> aggregate_neighbors(
        int node,                                   // the centre node
        const vector<vector<float>>& node_features, // Feature matrix of nodes
        const vector<vector<int>>& adjacency_list,  // graph adjacency list
        const vector<int>& degrees                  // pre-computed degrees of each node
    );

    // Applies weight matrix to the aggregated neighbour features to compute
    // the output for a single output dimension
    float linear_transform(
        const vector<float>& aggregated_features, // Aggregated and normalised neighbour features
        int output_index                          // index of output dimension being computed
    );
};
