#include "GraphReader.h"
#include <fstream>
#include <sstream>
#include <iostream>

Graph read_graph_from_file(const string& filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error opening file " << filename << "\n";
        exit(1);
    }

    string line;
    int num_nodes, num_features;

    // Read header
    getline(infile, line);
    istringstream header_stream(line);
    header_stream >> num_nodes >> num_features;

    Graph g(num_nodes, num_features);

    // Read node features
    for (int i = 0; i < num_nodes; i++) {
        getline(infile, line);
        istringstream iss(line);
        int node_id;
        iss >> node_id;
        vector<float> features(num_features);
        for (int j = 0; j < num_features; j++) {
            iss >> features[j];
        }
        g.set_node_feature(node_id, features);
    }

    // Read edges
    while (getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int src, dst;
        iss >> src >> dst;
        g.add_edge(src, dst);
    }

    infile.close();
    return g;
}
