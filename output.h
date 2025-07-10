#ifndef OUTPUT_CONVERTER_H
#define OUTPUT_CONVERTER_H

#include <vector>
#include <functional>
#include "Graph.h"   // your Graph class
using namespace std;

// Output converter namespace provides utility function to convert
// node level scores to edge level and graph level scores, binary edge predictions,
// graph level binaries.
namespace OutputConverter {

  using NodeScores   = vector<float>; // stores one score per node
  using EdgeScores   = vector<float>; // stores one score per edge
  using BinaryVector = vector<bool>;  // stores binary labels for edges on graphs

  // Default combiner: multiply the two endpoint scores to compute edge score
  inline float defaultCombiner(float a, float b) {
    return a * b;
  }

  // Default Aggregator: mean of all node scores for Graph level score
  inline float defaultAggregator(const NodeScores& v) {
    if (v.empty()) return 0.0f;
    float sum = 0.0f;
    for (auto x : v) sum += x;
    return sum / v.size();
  }

  // Convert one score per node → one score per edge using combiner function
  // If undirected, computes score for each edge only once.
  EdgeScores toEdgeScores(
    const NodeScores&                   nodeScores, //Input node scores
    const Graph&                        graph, // Graph Structure
    function<float(float,float)>   combiner  = defaultCombiner, // Function to combine two node scores
    bool                                undirected = true // treat edges as undirected if true
  );

  // Threshold edge‐level scores into true/false using a given threshold
  BinaryVector toEdgeBinary(
    const NodeScores&                   nodeScores,  //Input node scores
    const Graph&                        graph,  // Graph Structure
    float                               threshold,  // threshold for binary decision
    function<float(float,float)>   combiner  = defaultCombiner, // Function to combine two node scores
    bool                                undirected = true // treat edges as undirected if true
  );

  // Convert all node scores → one graph‐level score using aggregator function
  float toGraphScore(
    const NodeScores&                    nodeScores,  //Input node scores
    function<float(const NodeScores&)> aggregator = defaultAggregator // aggregator function
  );

  // Threshold graph‐level score into a boolean
  bool toGraphBinary(
    const NodeScores&                    nodeScores,  //Input node scores
    float                                threshold, // threshold for binary decision
    function<float(const NodeScores&)> aggregator = defaultAggregator // aggregator function
  );

} // namespace OutputConverter

#endif // OUTPUT_CONVERTER_H