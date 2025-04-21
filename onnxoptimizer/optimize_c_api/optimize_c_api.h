//
// Created by xyyang on 23-12-1.
//

#ifndef ONNX_OPTIMIZER_OPTIMIZE_C_API_H
#define ONNX_OPTIMIZER_OPTIMIZE_C_API_H
#include <vector>
void optimize_with_model_path(std::string& mp_in_path1,
                              std::string& mp_in_path2, std::string& mp_name1,
                              std::string& mp_name2, std::string& mp_out_path);

void merge_single_model_with_predicate(std::string& onnx_model_path,
                                       std::string& predicate,
                                       std::string& value_type,
                                       std::string prefix, int count);

void merge_double_models_with_predicate(std::string& onnx_model_path,
                                        std::string& predicate,
                                        std::string prefix_l,
                                        std::string prefix_r);

void merge_with_model_path(std::string& mp_in_path1, std::string& mp_in_path2,
                           std::string& mp_name1, std::string& mp_name2,
                           std::string& mp_out_path);

void optimize_on_merged_model(std::string& mp_in_path,
                              std::string& mp_out_path);

std::vector<std::string> check_redundant(std::string& changed_model_path,
                                         std::string& compared_model_path);

void change_models(std::string& changed_model_path,std::string& output_model_path,
                std::string& changed_input_model_path,
                const std::vector<std::string>& output_name);
void add_prefix_on_model(std::string& changed_model_path, std::string& output_model_path, std::string& prefix);

/**
 * @brief retree optimization 
 *        1. convert
 *        2. prune
 *        3. merge
 * @param input_model_path
 * @param comparison_operator 0: ==, 1: <, 2: <=, 3: >, 4: >= 
 * @param threshold 
 * @param threads_count default=1 
 * @return optimized model path
 */
std::string optimize_on_decision_tree_predicate(std::string& input_model_path, uint8_t comparison_operator, float threshold, int nthreads = 1);
std::string optimize_on_decision_tree_predicate_opt_level_0(std::string& input_model_path, uint8_t comparison_operator, float threshold, int nthreads = 1);
std::string optimize_on_decision_tree_predicate_convert(std::string& input_model_path);
std::string optimize_on_decision_tree_predicate_prune(std::string& input_model_path, uint8_t comparison_operator, float threshold, int threads_count = 1);
std::string optimize_on_decision_tree_predicate_merge(std::string& input_model_path, int threads_count = 1);
std::string optimize_on_decision_tree_predicate_naive_merge(std::string& input_model_path, int threads_count = 1);

int get_decision_tree_labels_size(std::string& input_model_path);


#endif  // ONNX_OPTIMIZER_OPTIMIZE_C_API_H

