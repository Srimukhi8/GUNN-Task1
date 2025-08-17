#include "output.h"
#include <unordered_set>
#include <cmath>

namespace OutputConverter {

  EdgeScores toEdgeScores(
    const NodeScores& nodeScores,
    const Graph&      graph,
    EdgeCombiner      combiner,
    bool              undirected
  ) {
    EdgeScores out;
    out.reserve(graph.num_nodes);

    unordered_set<long long> seen;
    auto key = [&](int u, int v) {
      return (static_cast<long long>(u) << 32)
           | static_cast<unsigned int>(v);
    };

    for (int u = 0; u < graph.num_nodes; ++u) {
      for (int v : graph.adjacency_list[u]) {
        if (undirected) {
          if (u >= v) continue;
          if (!seen.insert(key(u, v)).second) continue;
        }
        out.push_back(combiner(nodeScores[u], nodeScores[v]));
      }
    }
    return out;
  }

  BinaryVector toEdgeBinary(
    const NodeScores& nodeScores,
    const Graph&      graph,
    float             threshold,
    EdgeCombiner      combiner,
    bool              undirected
  ) {
    auto scores = toEdgeScores(nodeScores, graph, combiner, undirected);
    BinaryVector b; 
    b.reserve(scores.size());
    for (auto s : scores) 
      b.push_back(s > threshold);
    return b;
  }

  float toGraphScore(
    const NodeScores& nodeScores,
    GraphAggregator   aggregator
  ) {
    return aggregator(nodeScores);
  }

  bool toGraphBinary(
    const NodeScores& nodeScores,
    float             threshold,
    GraphAggregator   aggregator
  ) {
    return aggregator(nodeScores) > threshold;
  }

} // namespace OutputConverter
