#ifndef OUTPUT_CONVERTER_H
#define OUTPUT_CONVERTER_H

#include <vector>
#include <functional>
#include "Graph.h"       // your Graph class
#include "Aggregator.h"  // brings in NodeScores, EdgeCombiner, GraphAggregator, DefaultAgg

using namespace std;

namespace OutputConverter {

  using NodeScores   = vector<float>;    // one score per node  
  using EdgeScores   = vector<float>;    // one score per edge  
  using BinaryVector = vector<bool>;     // binary labels  

  // pull in our aliases
  using EdgeCombiner    = ::OutputConverter::EdgeCombiner;
  using GraphAggregator = ::OutputConverter::GraphAggregator;

  // If the user omits a combiner/aggregator, we default to these:
  //   DefaultAgg::prodCombiner   → a * b   (matches your old default)
  //   DefaultAgg::meanGraph      → mean(v) (matches your old default)
  
  EdgeScores toEdgeScores(
    const NodeScores& nodeScores,
    const Graph&      graph,
    EdgeCombiner      combiner   = DefaultAgg::prodCombiner,
    bool              undirected = true
  );

  BinaryVector toEdgeBinary(
    const NodeScores& nodeScores,
    const Graph&      graph,
    float             threshold,
    EdgeCombiner      combiner   = DefaultAgg::prodCombiner,
    bool              undirected = true
  );

  float toGraphScore(
    const NodeScores&   nodeScores,
    GraphAggregator     aggregator = DefaultAgg::meanGraph
  );

  bool toGraphBinary(
    const NodeScores&   nodeScores,
    float               threshold,
    GraphAggregator     aggregator = DefaultAgg::meanGraph
  );

} // namespace OutputConverter

#endif // OUTPUT_CONVERTER_H
