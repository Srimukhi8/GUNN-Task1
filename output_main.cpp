// main.cpp

#include "Graph.h"              // your Graph class
#include "GraphReader.h"         // gives read_graph_from_file(...)
#include "GCNL.h"               // your existing GCNLayer
#include "output.h"    // for edge/graph conversions
#include <iostream>
#include <vector>
#include <numeric>

int main(int argc, char** argv) {
    // 1) Grab the input filename from the command line
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_input_file>\n";
        return 1;
    }
    string filename = argv[1];

    // 2) Read the Graph (num_nodes, num_features, features, edges)
    //    from the user-provided file
    Graph g = read_graph_from_file(filename);

    // 3) Run one GCN layer to update each node's feature vector
    //    (input_dim = g.num_node_features, output_dim prompted from the user)
    int out;
    cin >> out;
    GCNLayer gcn(g.num_node_features, out);
    auto features = gcn.forward(g.node_features, g.adjacency_list);
    cout << "=== Node‐Level ===\n";
    for (const auto& node : features) {
        for (float val : node) {
            cout << val << " ";
        }
        cout << "\n";
    }
    cout << "\n";
    //    features: vector of size [num_nodes][out]

    // 4) Turn each 'out'-entry feature vector → one node score (sum)
    vector<float> nodeScores;
    nodeScores.reserve(features.size());
    for (auto &feat : features) {
        float sum = accumulate(feat.begin(), feat.end(), 0.0f);
        nodeScores.push_back(sum);
    }

    // 5) Use OutputConverter to get edge-level and graph-level outputs
    auto  edgeScores = OutputConverter::toEdgeScores(nodeScores, g);
    auto   edgeTruth = OutputConverter::toEdgeBinary(nodeScores, g, 0.5f);

    float graphScore = OutputConverter::toGraphScore(nodeScores);
    bool  graphTruth = OutputConverter::toGraphBinary(nodeScores, 0.5f);

    // 6) Print everything out in simple terms
    cout << "=== Graph‐Level ===\n";
    cout << "Score = " << graphScore
              << "\n\n";

    cout << "=== Edge‐Level ===\n";
    for (size_t i = 0; i < edgeScores.size(); ++i) {
        cout << "Edge " << i
                  << " | score = " << edgeScores[i]
                  << "\n";
    }

    return 0;
}