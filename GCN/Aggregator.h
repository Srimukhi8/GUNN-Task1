// Aggregator.h
#pragma once

#include "Graph.h"
#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace OutputConverter {

  //──────────────────────────────────────────────────────────────────────────
  // Type aliases matching OutputConverter signatures
  //──────────────────────────────────────────────────────────────────────────

  // A vector of per‐node scalar scores
  using NodeScores   = std::vector<float>;

  // A vector of per‐edge scalar scores
  using EdgeScores   = std::vector<float>;

  // A vector of boolean flags
  using BinaryVector = std::vector<bool>;

  // Combines two node‐scores into one edge‐score
  using EdgeCombiner    = std::function<float(float, float)>;

  // Aggregates all node‐scores into one graph‐score
  using GraphAggregator = std::function<float(const NodeScores&)>;


  //──────────────────────────────────────────────────────────────────────────
  // Default implementations (used when the user omits their own)
  //──────────────────────────────────────────────────────────────────────────
  namespace DefaultAgg {

    // sum of endpoint scores
    inline float sumCombiner(float a, float b) {
      return a + b;
    }

    // product of endpoint scores
    inline float prodCombiner(float a, float b) {
      return a * b;
    }

    // maximum of endpoint scores
    inline float maxCombiner(float a, float b) {
      return std::max(a, b);
    }

    // minimum of endpoint scores
    inline float minCombiner(float a, float b) {
      return std::min(a, b);
    }

    // absolute difference of endpoint scores
    inline float absDiffCombiner(float a, float b) {
      return std::fabs(a - b);
    }

    // sum of all node scores
    inline float sumGraph(const NodeScores& scores) {
      return std::accumulate(scores.begin(), scores.end(), 0.0f);
    }

    // mean of all node scores
    inline float meanGraph(const NodeScores& scores) {
      if (scores.empty()) return 0.0f;
      return std::accumulate(scores.begin(), scores.end(), 0.0f)
           / static_cast<float>(scores.size());
    }

    // maximum of all node scores
    inline float maxGraph(const NodeScores& scores) {
      if (scores.empty()) return 0.0f;
      return *std::max_element(scores.begin(), scores.end());
    }

    // minimum of all node scores
    inline float minGraph(const NodeScores& scores) {
      if (scores.empty()) return 0.0f;
      return *std::min_element(scores.begin(), scores.end());
    }

  }  // namespace DefaultAgg

}  // namespace OutputConverter
