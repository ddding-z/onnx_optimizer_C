#pragma once

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "onnx/common/ir.h"
#include "onnx/proto_utils.h"

namespace onnx::optimization {

struct NodeID {
  int id;
  std::string node;
};

class TreeNode {
 public:
  int id;
  int feature_id;
  std::string mode;
  double value;
  std::optional<int> target_id;
  std::optional<double> target_weight;
  int samples;

  std::weak_ptr<TreeNode> parent;
  std::shared_ptr<TreeNode> left;
  std::shared_ptr<TreeNode> right;

  TreeNode(int _id, int _feature_id, const std::string& _mode, double _value,
           std::optional<int> _target_id, std::optional<double> _target_weight,
           int _samples);
};

class TreeEnsembleRegressor {
 public:
  TreeEnsembleRegressor();

  static TreeEnsembleRegressor from_trees(
      const std::vector<std::shared_ptr<TreeNode>>& roots);

  static TreeEnsembleRegressor from_tree(std::shared_ptr<TreeNode> root,
                                         int tree_no = 0);

  int n_targets;
  std::vector<int64_t> nodes_falsenodeids;
  std::vector<int64_t> nodes_featureids;
  std::vector<double> nodes_hitrates;
  std::vector<int64_t> nodes_missing_value_tracks_true;
  std::vector<std::string> nodes_modes;
  std::vector<int64_t> nodes_nodeids;
  std::vector<int64_t> nodes_treeids;
  std::vector<int64_t> nodes_truenodeids;
  std::vector<double> nodes_values;
  std::string post_transform;
  std::vector<int64_t> target_ids;
  std::vector<int64_t> target_nodeids;
  std::vector<int64_t> target_treeids;
  std::vector<double> target_weights;

 private:
  static void from_tree_internal(TreeEnsembleRegressor& regressor,
                                 std::shared_ptr<TreeNode> node, int tree_no);
};

/**
 * @brief all retree rules
 */
class ReTreeRule {
 private:
  using ComparisonFunc = bool (*)(float, float);
  static ComparisonFunc comparison_funcs[];

 public:
  static void processNode(Node* node);
  static void processNode(
      Node* node, std::vector<std::vector<std::string>>& result_nodes_list,
      std::vector<std::tuple<int, int>>& tree_intervals);
  static void convertModelProto(ModelProto& mp_in, Node* node,
                                std::string& model_path);
  static std::vector<std::tuple<int, int>> getTreeIntervals(Node* node);
  static std::vector<std::tuple<int, int>> get_target_tree_intervals(
      Node* node);
  static std::shared_ptr<TreeNode> model2tree(
      Node* treeNode, int64_t root_node_id, std::tuple<int, int>& tree_interval,
      std::tuple<int, int>& target_tree_interval);

  static std::vector<std::shared_ptr<TreeNode>> model2trees(Node* treeNode,
                                                            int nthreads);
  static void toTrees(Node* treeNode, TreeEnsembleRegressor& regressor);
  static void pruning(size_t tree_no, const std::tuple<int, int>& tree_interval,
                      std::vector<std::string>& result_nodes, Node* treeNode,
                      uint8_t comparison_operator, float threshold);
  static void update_result(
      std::shared_ptr<TreeNode> node, int path_length,
      std::shared_ptr<TreeNode> root,
      std::vector<std::tuple<std::shared_ptr<TreeNode>, int>>& result);
  static int find_merge_nodes(
      std::shared_ptr<TreeNode> node, int path_length,
      std::shared_ptr<TreeNode> root, bool left_branch,
      std::vector<std::tuple<std::shared_ptr<TreeNode>, int>>& result);
  static void merge(std::shared_ptr<TreeNode> root,
                    std::vector<std::shared_ptr<TreeNode>> nodes,
                    bool left_branch);
  static void dfs(std::shared_ptr<TreeNode> node);
  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph, Node* treeNode,
                           std::string& model_path, uint8_t comparison_operator,
                           float threshold, int nthreads, int opt_level);
  static std::string match(std::string& model_path, uint8_t comparison_operator,
                           float threshold, int nthreads, int opt_level);
};

/**
 * @brief convert treeclassifier to treeregressor
 */
class DTConvertRule {
 public:
  static void processNode(Node* node);

  static std::string convertModelProto(ModelProto& mp_in, Node* node,
                                       std::string& model_path);

  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                           Node* node, std::string& model_path);

  static std::string match(std::string& model_path);

  static int getLabelsSize(std::string& model_path);
};

/**
 * @brief random forest prune
 */
class DTPruneRule {
 private:
  using ComparisonFunc = bool (*)(float, float);
  static ComparisonFunc comparison_funcs[];

 public:
  static std::vector<std::tuple<int, int>> getTreeIntervals(Node* node);
  static int pruning_recursive(size_t tree_no,
                               const std::tuple<int, int>& tree_interval,
                               size_t node_id,
                               std::vector<std::string>& result_nodes,
                               Node* treeNode, uint8_t comparison_operator,
                               float threshold);
  static void pruning_loop(size_t tree_no,
                           const std::tuple<int, int>& tree_interval,
                           std::vector<std::string>& result_nodes,
                           Node* treeNode, uint8_t comparison_operator,
                           float threshold);
  static void pruning_loop_optimized(size_t tree_no,
                                     const std::tuple<int, int>& tree_interval,
                                     std::vector<std::string>& result_nodes,
                                     Node* treeNode,
                                     uint8_t comparison_operator,
                                     float threshold);
  static bool processNode(
      Node* node, std::vector<std::vector<std::string>>& result_nodes_list,
      std::vector<std::tuple<int, int>>& tree_intervals);
  static std::string apply(int threads_count, ModelProto& mp_in,
                           std::shared_ptr<Graph>& graph,
                           std::string& model_path, Node* treeNode,
                           uint8_t comparison_operator, float threshold);
  static std::string match(std::string& model_path, uint8_t comparison_operator,
                           float threshold, int threads_count);
};

/**
 * @brief random forest merge
 */
class DTMergeRule {
 public:
  static std::shared_ptr<TreeNode> model2tree_recursive(
      Node* treeNode, int64_t node_id, std::shared_ptr<TreeNode> parent,
      std::tuple<int, int>& tree_interval,
      std::tuple<int, int>& target_tree_interval);
  static void dfs(std::shared_ptr<TreeNode> node);
  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                           std::string& model_path, Node* treeNode,
                           int nthreads);
  static std::string match(std::string& model_path, int nthreads);
};

class DTNaiveMergeRule {
 public:
  static void dfs(std::shared_ptr<TreeNode> node);
  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                           std::string& model_path, Node* treeNode,
                           int nthreads);
  static std::string match(std::string& model_path, int nthreads);
};

}  // namespace onnx::optimization