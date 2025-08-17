#include <iostream>
#include "GraphReader.h"
#include "GCNL.h" // or GraphSAGELayer, GATLayer

int main() {
    Graph g = read_graph_from_file("graph_data.txt");
    int out;
    std::cin >> out;
    GCNLayer gcn(g.num_node_features, out);
    auto updated_features = gcn.forward(g.node_features, g.adjacency_list);

    for (const auto& node : updated_features) {
        for (float val : node) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}
