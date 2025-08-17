// main.cpp

#include "Graph.h"              // your Graph class
#include "GraphReader.h"        // read_graph_from_file(...)
#include "GCNL.h"               // your existing GCNLayer
#include "output.h"             // OutputConverter API
#include <iostream>
#include <vector>
#include <numeric>
#include <functional>           // for function

int main(int argc, char** argv) {
    // 1) Grab the input filename
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_input_file>\n";
        return 1;
    }
    string filename = argv[1];

    // 2) Read the graph
    Graph g = read_graph_from_file(filename);

    // 3) Run one GCN layer
    cout << "Enter output feature dimension: ";
    int out_dim;
    cin >> out_dim;

    GCNLayer gcn(g.num_node_features, out_dim);
    auto features = gcn.forward(g.node_features, g.adjacency_list);

    cout << "=== Node Features (post-GCN) ===\n";
    for (size_t i = 0; i < features.size(); ++i) {
        cout << "Node " << i << ": ";
        for (float val : features[i]) {
            cout << val << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    // 4) Compute node‐level scores (sum of features)
    vector<float> nodeScores;
    nodeScores.reserve(features.size());
    for (auto &feat : features) {
        float sum = accumulate(feat.begin(), feat.end(), 0.0f);
        nodeScores.push_back(sum);
    }

    // 5) Define CUSTOM edge‐combiner and graph‐aggregator
    // Example edge combiner: squared difference of endpoint scores
    OutputConverter::EdgeCombiner customEdgeCombiner = 
        [](float a, float b) {
            float diff = a - b;
            return diff * diff;
        };

    // Example graph aggregator: maximum node score
    OutputConverter::GraphAggregator customGraphAgg = 
        [](const OutputConverter::NodeScores& v) {
            if (v.empty()) return 0.0f;
            return *max_element(v.begin(), v.end());
        };

    // 6) Use OutputConverter with CUSTOM functions
    auto edgeScores = OutputConverter::toEdgeScores(
        nodeScores, 
        g,   
        OutputConverter::DefaultAgg::prodCombiner,// use squared‐difference combiner
        true                   // undirected graph
    );

    auto edgeTruth = OutputConverter::toEdgeBinary(
        nodeScores,
        g,
        0.5f,                  // threshold  
        OutputConverter::DefaultAgg::prodCombiner,// same combiner
        true
    );

    float graphScore = OutputConverter::toGraphScore(
        nodeScores, 
        OutputConverter::DefaultAgg::meanGraph        // use max aggregator
    );

    bool graphTruth = OutputConverter::toGraphBinary(
        nodeScores,
        0.5f,                  // threshold        
        OutputConverter::DefaultAgg::meanGraph  // same aggregator
    );

    // 7) Print results
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
