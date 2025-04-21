//
// Created by xyyang on 23-11-14.
//
//#include <onnxoptimizer/model_util.h>

#include <string>
#include <vector>
#include <iostream>
#include "onnxoptimizer/optimize_c_api/optimize_c_api.h"

int main(int argc, char* argv[]) {
  std::string flights = "/volumn/Retree_exp/duckdb/third_party/onnx_optimizer/examples/model4test/flights_rf.onnx";
  std::string nasa = "/volumn/Retree_exp/duckdb/third_party/onnx_optimizer/examples/model4test/nasa_rf.onnx";

  optimize_on_decision_tree_predicate(flights, 0, 1, 16);
  optimize_on_decision_tree_predicate(nasa, 0, 1, 16);
  optimize_on_decision_tree_predicate_convert(flights);
  optimize_on_decision_tree_predicate_convert(nasa);
  
  return 0;
}
