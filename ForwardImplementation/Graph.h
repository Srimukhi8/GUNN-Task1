#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
using namespace std;

// Produces a clear abstraction for representing a graph structure
// with node features, adjacency list, optionally edge and global features
// to facilitate GNN architecture
class Graph {
public:
    int num_nodes;         // Number of nodes in the graph
    int num_node_features; // Number of features per node

    vector<vector<float>> node_features;   // Feature matrix : [n_nodes][n_node_features]
    vector<vector<int>> adjacency_list;    // graph_representation : [n_nodes][variable number of neighbors]

    // Optional features :
    vector<vector<float>> edge_features;   // For edge-conditioned GNNs
    vector<float> global_features;         // For graph-level attributes

    // New addition: stores (src, dst) for every edge added
    vector<pair<int, int>> edge_list;

    // Constructs a graph with the specified number of nodes and node feature dimensions.
    // Initializes empty adjacency list and zero-initialized feature matrices.
    Graph(int num_nodes, int num_node_features);

    // Move constructor for efficient transfers without deep copying
    Graph(Graph&& other) noexcept;

    // Move assignment operator for efficient transfers without deep copying
    Graph& operator=(Graph&& other) noexcept;

    // Adds an undirected edge between source node and destination node
    void add_edge(int src, int dst);

    // Sets the feature vector of a specific node
    void set_node_feature(int node_id, const vector<float>& features);

    // New accessor: returns the src and dst for a given edge ID
    pair<int, int> edge(size_t edge_id) const;
};

#endif
