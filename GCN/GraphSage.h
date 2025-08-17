// GraphSage.h

#pragma once
#include "BaseLayer.h"
#include <vector>
using namespace std;

// Implements GraphSage Layer which aggregates the neighbouring features
// using mean aggregation and CONCATENATES them with node's own features
// and then performs linear transformation.
class GraphSAGELayer : public BaseLayer {
public:
    // This constructor initialises the Layer with input and output dimensions
    // performs Xavier initialisation of weight matrix.
    GraphSAGELayer(int input_dim, int output_dim);

    // forward pass computes updated node features by aggregating neighbour features,
    // concatenating with self-features and multiplying with weight matrix followed by ReLU.
    vector<vector<float>> forward(
        const vector<vector<float>>& node_features, // node-feature matrix:[number of nodes][input_dim]
        const vector<vector<int>>& adjacency_list   // represents the graph
    ) override;

private:
    int input_dim;              // dimension of input features
    int output_dim;             // dimension of output features
    vector<vector<float>> weight_matrix; // weight matrix of shape [input_dim][output_dim]
    
    // Applies ReLU activation to single float value
    float relu(float x);

    // Aggregates features of the neighbours of this node using mean aggregation.
    vector<float> aggregate_neighbors_mean(
        int node,                                   // index of the central node
        const vector<vector<float>>& node_features, // input node feature matrix
        const vector<vector<int>>& adjacency_list   // graph representation
    );

    // CONCATENATES node's own features to the aggregated neighbour features
    // resulting feature vector is of size 2*input_dim.
    vector<float> concatenate_self_and_neighbors(
        const vector<float>& self_features,     // node's own feature vector
        const vector<float>& neighbor_features  // aggregated neighbour-feature vector
    );

    // Applies weight matrix to the aggregated feature vector to compute
    // output for a single output dimension
    float linear_transform(
        const vector<float>& concat_features, // the concatenated feature vector
        int output_index                      // index of output dimension being computed
    );
};
