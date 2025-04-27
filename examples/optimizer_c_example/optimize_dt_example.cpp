//
// Created by xyyang on 23-11-14.
//
//#include <onnxoptimizer/model_util.h>

#include <string>
#include <vector>
#include <iostream>
#include "onnxoptimizer/optimize_c_api/optimize_c_api.h"
#include <chrono>
#include <fstream>
#include <stdlib.h>

/* 
./optimize_dt_example workload model_name comparison_operator threshold nthreads 
*/
int main(int argc, char* argv[]) {
  std::string workload = argv[1];
  std::string model_name = argv[2];
  int comparison_operator = atoi(argv[3]);
  char* endptr;
  float threshold = strtof(argv[4], &endptr);
  if (endptr == argv[4]) {
      printf("参数不是有效的浮点数\n");
      return 1;
  }
  int nthread = atoi(argv[5]);
  int times = 20;

  std::string root_path = "/volumn/Retree_exp/workloads/";
  std::string model_path = root_path + workload + "/model/" + model_name;

  std::ofstream outputfile;
	outputfile.open("/volumn/Retree_exp/micro/optimization/optimization_cost.csv", std::ios::app);

  double total_merge_cost = 0.0;
  for (size_t i = 0; i < times; i++)
  {
    auto start = std::chrono::high_resolution_clock::now();
    optimize_on_decision_tree_predicate(model_path, comparison_operator, threshold, nthread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    total_merge_cost += duration.count();
  }

  double total_prune_cost = 0.0;
  for (size_t i = 0; i < times; i++)
  {
    auto start = std::chrono::high_resolution_clock::now();
    optimize_on_decision_tree_predicate_opt_level_0(model_path, comparison_operator, threshold, nthread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    total_prune_cost += duration.count();
  }

  outputfile << workload << "," << model_name << "," << threshold << ","
  << nthread << "," << total_prune_cost/times << "," << total_merge_cost/times << "\n";
  std::cout << workload << "," << model_name << "," << threshold << ","
  << nthread << "," << total_prune_cost/times << "," << total_merge_cost/times << "\n";
  
  return 0;
}
