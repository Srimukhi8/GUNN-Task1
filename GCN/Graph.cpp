#include "Graph.h"

Graph::Graph(int num_nodes, int num_node_features)
    : num_nodes(num_nodes), num_node_features(num_node_features)
{
    node_features.resize(num_nodes, vector<float>(num_node_features, 0.0f));
    adjacency_list.resize(num_nodes);
}

Graph::Graph(Graph&& other) noexcept
    : num_nodes(other.num_nodes),
      num_node_features(other.num_node_features),
      node_features(std::move(other.node_features)),
      adjacency_list(std::move(other.adjacency_list)),
      edge_features(std::move(other.edge_features)),
      global_features(std::move(other.global_features)),
      edge_list(std::move(other.edge_list))  // Added
{}

Graph& Graph::operator=(Graph&& other) noexcept {
    if (this != &other) {
        num_nodes = other.num_nodes;
        num_node_features = other.num_node_features;
        node_features = std::move(other.node_features);
        adjacency_list = std::move(other.adjacency_list);
        edge_features = std::move(other.edge_features);
        global_features = std::move(other.global_features);
        edge_list = std::move(other.edge_list);  // Added
    }
    return *this;
}

void Graph::add_edge(int src, int dst) {
    adjacency_list[src].push_back(dst);
    // If undirected:
    adjacency_list[dst].push_back(src);
    edge_list.emplace_back(src, dst);  // Added
}

void Graph::set_node_feature(int node_id, const vector<float>& features) {
    if (features.size() == num_node_features) {
        node_features[node_id] = features;
    }
    // Otherwise, throw or handle mismatch
}

// Added: Returns (src, dst) for given edge index
pair<int, int> Graph::edge(size_t edge_id) const {
    return edge_list[edge_id];
}
