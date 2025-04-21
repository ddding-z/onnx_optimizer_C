#include "retree.h"

#include "retree_util.cpp"

#define M_FALSE 0
#define M_TRUE 1
#define M_NO 2

#define CONSERVATIVE 0

namespace onnx::optimization {

/**
 * @brief convert treeclassifier to treeregressor
 */
void DTConvertRule::processNode(Node* node) {
  auto classlabels_int64s = node->is(Symbol("classlabels_int64s"));
  auto class_treeids = node->is(Symbol("class_treeids"));
  auto class_ids = node->is(Symbol("class_ids"));
  auto class_nodeids = node->is(Symbol("class_nodeids"));
  auto class_weights = node->fs(Symbol("class_weights"));

  std::unordered_set<int> unique_treeids(class_treeids.begin(),
                                         class_treeids.end());
  int n_trees = unique_treeids.size();

  // ** convert clf attributes 2 reg attributes
  int64_t stride =
      classlabels_int64s.size() == 2 ? 1 : classlabels_int64s.size();
  int nleaf = static_cast<int>(class_weights.size() / stride);

  // input_class_treeids 2 target_treeids
  std::vector<int64_t> target_treeids;
  target_treeids.reserve(nleaf);
  for (int64_t i = 0; i < nleaf; ++i) {
    target_treeids.push_back(class_treeids[i * stride]);
  }

  // input_class_ids 2 target_ids
  std::vector<int64_t> target_ids;
  target_ids.reserve(nleaf);
  for (int64_t i = 0; i < nleaf; ++i) {
    target_ids.push_back(class_ids[i * stride]);
  }

  // input_class_nodeids 2 target_nodeids
  std::vector<int64_t> target_nodeids;
  target_nodeids.reserve(nleaf);
  for (int64_t i = 0; i < nleaf; ++i) {
    target_nodeids.push_back(class_nodeids[i * stride]);
  }

  // input_class_weights 2 target_weights
  std::vector<double> target_weights;
  target_weights.reserve(nleaf);
  if (stride == 1) {
    for (auto w : class_weights) {
      target_weights.push_back((w > 0.5 / n_trees) ? 1.0 : 0.0);
    }
  } else {
    for (int i = 0; i < nleaf; ++i) {
      auto start_it = class_weights.begin() + (i * stride);
      auto end_it = class_weights.begin() + ((i + 1) * stride);

      auto max_it = std::max_element(start_it, end_it);
      int index = std::distance(start_it, max_it);
      // ** case: wine_quality classlabels_int64s [3,4,5,6,7,8]
      target_weights.push_back(static_cast<double>(classlabels_int64s[index]));
    }
  }

  node->removeAttribute(Symbol("classlabels_int64s"));
  node->removeAttribute(Symbol("class_treeids"));
  node->removeAttribute(Symbol("class_ids"));
  node->removeAttribute(Symbol("class_nodeids"));
  node->removeAttribute(Symbol("class_weights"));

  node->i_(Symbol("n_targets"), 1);
  node->is_(Symbol("target_treeids"), std::move(target_treeids));
  node->is_(Symbol("target_ids"), std::move(target_ids));
  node->is_(Symbol("target_nodeids"), std::move(target_nodeids));
  node->fs_(Symbol("target_weights"), std::move(target_weights));
}

std::string DTConvertRule::convertModelProto(ModelProto& mp_in, Node* node,
                                             std::string& model_path) {
  auto g_m = mp_in.mutable_graph();
  auto input = g_m->input();

  // construct output
  onnx::ValueInfoProto output;
  TensorShapeProto_Dimension output_dim1, output_dim2;
  output_dim2.set_dim_value(1);
  output.set_name("regression");
  output.mutable_type()->mutable_tensor_type()->set_elem_type(1);
  *output.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim() =
      output_dim1;
  *output.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim() =
      output_dim2;

  // construct input
  onnx::NodeProto* p_n_clf;
  for (int i = 0; i < g_m->node_size(); ++i) {
    auto node = g_m->mutable_node(i);
    if (node->op_type() == "TreeEnsembleClassifier") {
      p_n_clf = node;
    }
  }

  // construct TreeEnsembleRegressor
  onnx::NodeProto p_n;
  p_n.set_op_type("TreeEnsembleRegressor");
  p_n.set_name(p_n_clf->name());
  p_n.set_domain(p_n_clf->domain());
  auto inputs = p_n_clf->input();
  for (auto input : inputs) {
    p_n.add_input(input);
  }
  p_n.add_output(output.name());
  for (auto attr_name : node->attributeNames()) {
    addAttribute(&p_n, node, attr_name);
  }

  // Remove TreeEnsembleClassifier and subsequent Nodes
  for (int i = g_m->node_size() - 1; i >= 0; i--) {
    auto node = g_m->node(i);
    if (node.op_type() == "TreeEnsembleClassifier") {
      g_m->mutable_node()->DeleteSubrange(i, 1);
    } else if (node.op_type() == "ZipMap") {
      if (node.input()[0] == "probabilities") {
        g_m->mutable_node()->DeleteSubrange(i, 1);
      }
    } else if (node.op_type() == "Cast") {
      if (node.input()[0] == "label") {
        g_m->mutable_node()->DeleteSubrange(i, 1);
      }
    }
  }
  // replace TreeEnsembleClassifier with TreeEnsembleRegressor
  g_m->clear_output();
  *g_m->add_node() = p_n;
  *g_m->add_output() = output;

  onnx::checker::check_model(mp_in);
  auto new_model_path = getNewmodelname(model_path, "reg");
  saveModel(&mp_in, new_model_path);
  return new_model_path;
}

std::string DTConvertRule::apply(ModelProto& mp_in,
                                 std::shared_ptr<Graph>& graph, Node* node,
                                 std::string& model_path) {
  if (node->hasAttribute(Symbol("classlabels_int64s"))) {
    processNode(node);
    return convertModelProto(mp_in, node, model_path);
  } else if (node->hasAttribute(Symbol("classlabels_strings"))) {
    // todo: convert classlabels_strings
    // processNode(node, 1);
    // return convertModelProto(mp_in, node, model_path, 1);
  }
  return model_path;
}

std::string DTConvertRule::match(std::string& model_path) {
  std::string output_model_path = model_path;
  ModelProto mp_in;
  loadModel(&mp_in, model_path, true);
  std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

  bool found = false;
  Node* treeNode;
  graph->forEachNode([&found, &treeNode](Node* node) {
    if (node->hasAttribute(Symbol("class_treeids"))) {
      if (node->s(Symbol("post_transform")) == "NONE") {
        found = true;
        treeNode = node;
      }
    }
  });

  if (found)
    output_model_path = apply(mp_in, graph, treeNode, model_path);

  return output_model_path;
}

int DTConvertRule::getLabelsSize(std::string& model_path) {
  ModelProto mp_in;
  loadModel(&mp_in, model_path, true);
  std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

  int count = 2;
  graph->forEachNode([&count](Node* node) {
    if (node->hasAttribute(Symbol("classlabels_int64s"))) {
      auto classlabels_int64s = node->is(Symbol("classlabels_int64s"));
      count = classlabels_int64s.size();
    }
  });
  return count;
}

/**
 * @brief random forest prune
 */
DTPruneRule::ComparisonFunc DTPruneRule::comparison_funcs[] = {
    [](float x, float y) { return x == y; },
    [](float x, float y) { return x < y; },
    [](float x, float y) { return x <= y; },
    [](float x, float y) { return x > y; },
    [](float x, float y) { return x >= y; }};

std::vector<std::tuple<int, int>> DTPruneRule::getTreeIntervals(Node* node) {
  std::vector<int> tree_roots;
  std::vector<int64_t> nodes_treeids = node->is(Symbol("nodes_treeids"));
  int next_tree_id = 0;
  for (size_t i = 0; i < nodes_treeids.size(); ++i) {
    if (nodes_treeids[i] == next_tree_id) {
      next_tree_id++;
      tree_roots.push_back(i);
    }
  }

  std::vector<std::tuple<int, int>> tree_intervals;
  for (size_t i = 0; i < tree_roots.size(); ++i) {
    int end;
    if (i == tree_roots.size() - 1) {
      end = nodes_treeids.size();
    } else {
      end = tree_roots[i + 1];
    }
    tree_intervals.push_back(std::make_tuple(tree_roots[i], end));
  }

  return tree_intervals;
}

int DTPruneRule::pruning_recursive(size_t tree_no,
                                   const std::tuple<int, int>& tree_interval,
                                   size_t node_id,
                                   std::vector<std::string>& result_nodes,
                                   Node* treeNode, uint8_t comparison_operator,
                                   float threshold) {
  int tree_start = std::get<0>(tree_interval);
  int tree_end = std::get<1>(tree_interval);

  const auto& left_nodess = treeNode->is(Symbol("nodes_truenodeids"));
  const auto& right_nodess = treeNode->is(Symbol("nodes_falsenodeids"));
  const auto& node_typess = treeNode->ss(Symbol("nodes_modes"));

  std::vector<int64_t> left_nodes(left_nodess.begin() + tree_start,
                                  left_nodess.begin() + tree_end);
  std::vector<int64_t> right_nodes(right_nodess.begin() + tree_start,
                                   right_nodess.begin() + tree_end);
  std::vector<std::string> node_types(node_typess.begin() + tree_start,
                                      node_typess.begin() + tree_end);

  const auto& target_treeids = treeNode->is(Symbol("target_treeids"));
  const auto& target_nodeids = treeNode->is(Symbol("target_nodeids"));
  const auto& target_weights = treeNode->fs(Symbol("target_weights"));

  result_nodes[node_id] = node_types[node_id];
  bool is_leaf = (node_types[node_id] == "LEAF");

  if (is_leaf) {
    int target_idx = -1;
    for (size_t ti = 0; ti < target_nodeids.size(); ++ti) {
      int ni = target_nodeids[ti];
      if (ni == node_id && target_treeids[ti] == tree_no) {
        target_idx = ti;
        break;
      }
    }

    int result = comparison_funcs[comparison_operator](
        target_weights[target_idx], threshold);
    result_nodes[node_id] = (result == 1) ? "LEAF_TRUE" : "LEAF_FALSE";

    return result;

  } else {
    size_t left_node_id = left_nodes[node_id];
    int left_result =
        pruning_recursive(tree_no, tree_interval, left_node_id, result_nodes,
                          treeNode, comparison_operator, threshold);

    size_t right_node_id = right_nodes[node_id];
    int right_result =
        pruning_recursive(tree_no, tree_interval, right_node_id, result_nodes,
                          treeNode, comparison_operator, threshold);

    if (left_result == 0 && right_result == 0) {
      result_nodes[node_id] = "LEAF_FALSE";
      result_nodes[left_node_id] = "REMOVED";
      result_nodes[right_node_id] = "REMOVED";
      return 0;
    }

    if (left_result == 1 && right_result == 1) {
      result_nodes[node_id] = "LEAF_TRUE";
      result_nodes[left_node_id] = "REMOVED";
      result_nodes[right_node_id] = "REMOVED";
      return 1;
    }

    return 2;
  }
}

void DTPruneRule::pruning_loop(size_t tree_no,
                               const std::tuple<int, int>& tree_interval,
                               std::vector<std::string>& result_nodes,
                               Node* treeNode, uint8_t comparison_operator,
                               float threshold) {
  int tree_start = std::get<0>(tree_interval);
  int tree_end = std::get<1>(tree_interval);

  const auto& left_nodess = treeNode->is(Symbol("nodes_truenodeids"));
  const auto& right_nodess = treeNode->is(Symbol("nodes_falsenodeids"));
  const auto& node_typess = treeNode->ss(Symbol("nodes_modes"));
  std::vector<int64_t> left_nodes(left_nodess.begin() + tree_start,
                                  left_nodess.begin() + tree_end);
  std::vector<int64_t> right_nodes(right_nodess.begin() + tree_start,
                                   right_nodess.begin() + tree_end);
  std::vector<std::string> node_types(node_typess.begin() + tree_start,
                                      node_typess.begin() + tree_end);

  const auto& target_treeids = treeNode->is(Symbol("target_treeids"));
  const auto& target_nodeids = treeNode->is(Symbol("target_nodeids"));
  const auto& target_weights = treeNode->fs(Symbol("target_weights"));

  std::vector<int> computed_result(node_types.size(), -1);

  struct StackFrame {
    int64_t node_id;
    bool visited;
  };
  std::stack<StackFrame> s;
  s.push(StackFrame{0, false});

  while (!s.empty()) {
    auto& frame = s.top();
    size_t curr_id = frame.node_id;
    if (!frame.visited) {
      result_nodes[curr_id] = node_types[curr_id];
      frame.visited = true;

      if (node_types[curr_id] == "LEAF") {
        int target_idx = -1;
        for (size_t ti = 0; ti < target_nodeids.size(); ++ti) {
          if (target_nodeids[ti] == curr_id && target_treeids[ti] == tree_no) {
            target_idx = ti;
            break;
          }
        }
        int result = comparison_funcs[comparison_operator](
            target_weights[target_idx], threshold);
        result_nodes[curr_id] = (result == 1) ? "LEAF_TRUE" : "LEAF_FALSE";
        computed_result[curr_id] = result;
        s.pop();
      } else {
        s.push(StackFrame{right_nodes[curr_id], false});
        s.push(StackFrame{left_nodes[curr_id], false});
      }
    } else {
      size_t left_id = left_nodes[curr_id];
      size_t right_id = right_nodes[curr_id];
      int left_res = computed_result[left_id];
      int right_res = computed_result[right_id];

      if (left_res == 0 && right_res == 0) {
        result_nodes[curr_id] = "LEAF_FALSE";
        result_nodes[left_id] = "REMOVED";
        result_nodes[right_id] = "REMOVED";
        computed_result[curr_id] = 0;
      } else if (left_res == 1 && right_res == 1) {
        result_nodes[curr_id] = "LEAF_TRUE";
        result_nodes[left_id] = "REMOVED";
        result_nodes[right_id] = "REMOVED";
        computed_result[curr_id] = 1;
      } else {
        computed_result[curr_id] = 2;
      }
      s.pop();
    }
  }
}

void DTPruneRule::pruning_loop_optimized(
    size_t tree_no, const std::tuple<int, int>& tree_interval,
    std::vector<std::string>& result_nodes, Node* treeNode,
    uint8_t comparison_operator, float threshold) {
  int tree_start = std::get<0>(tree_interval);
  int tree_end = std::get<1>(tree_interval);
  size_t n_nodes = tree_end - tree_start;

  const auto& left_nodess = treeNode->is(Symbol("nodes_truenodeids"));
  const auto& right_nodess = treeNode->is(Symbol("nodes_falsenodeids"));
  const auto& node_typess = treeNode->ss(Symbol("nodes_modes"));
  const int64_t* left_nodes_ptr = left_nodess.data() + tree_start;
  const int64_t* right_nodes_ptr = right_nodess.data() + tree_start;
  const std::string* node_types_ptr = node_typess.data() + tree_start;

  const auto& target_treeids = treeNode->is(Symbol("target_treeids"));
  const auto& target_nodeids = treeNode->is(Symbol("target_nodeids"));
  const auto& target_weights = treeNode->fs(Symbol("target_weights"));

  std::unordered_map<int, size_t> target_index;
  target_index.reserve(target_nodeids.size());
  for (size_t ti = 0; ti < target_nodeids.size(); ++ti) {
    if (target_treeids[ti] == tree_no) {
      target_index[target_nodeids[ti]] = ti;
    }
  }

  std::vector<int> computed_result(n_nodes, -1);

  struct StackFrame {
    int64_t node_id;
    bool visited;
  };
  std::vector<StackFrame> stack;
  stack.reserve(n_nodes);
  stack.push_back({0, false});

  while (!stack.empty()) {
    auto& frame = stack.back();
    size_t curr_id = static_cast<size_t>(frame.node_id);

    if (!frame.visited) {
      if (node_types_ptr[curr_id] == "LEAF") {
        auto it = target_index.find(static_cast<int>(curr_id));
        int target_idx = (it != target_index.end() ? it->second : -1);
        int result = comparison_funcs[comparison_operator](
            target_weights[target_idx], threshold);
        result_nodes[curr_id] = (result == 1) ? "LEAF_TRUE" : "LEAF_FALSE";
        computed_result[curr_id] = result;
        stack.pop_back();
      } else {
        result_nodes[curr_id] = node_types_ptr[curr_id];
        frame.visited = true;
        stack.push_back({right_nodes_ptr[curr_id], false});
        stack.push_back({left_nodes_ptr[curr_id], false});
      }
    } else {
      size_t left_id = static_cast<size_t>(left_nodes_ptr[curr_id]);
      size_t right_id = static_cast<size_t>(right_nodes_ptr[curr_id]);
      int left_res = computed_result[left_id];
      int right_res = computed_result[right_id];

      if (left_res == 0 && right_res == 0) {
        result_nodes[curr_id] = "LEAF_FALSE";
        result_nodes[left_id] = "REMOVED";
        result_nodes[right_id] = "REMOVED";
        computed_result[curr_id] = 0;
      } else if (left_res == 1 && right_res == 1) {
        result_nodes[curr_id] = "LEAF_TRUE";
        result_nodes[left_id] = "REMOVED";
        result_nodes[right_id] = "REMOVED";
        computed_result[curr_id] = 1;
      } else {
        computed_result[curr_id] = 2;
      }
      stack.pop_back();
    }
  }
}

bool DTPruneRule::processNode(
    Node* node, std::vector<std::vector<std::string>>& result_nodes_list,
    std::vector<std::tuple<int, int>>& tree_intervals) {
  const int tree_count = tree_intervals.size();
  std::vector<int> tree_leaf_counts;
  for (const auto& removed_nodes : result_nodes_list) {
    int leaf_false_count =
        std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_FALSE");
    int leaf_true_count =
        std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_TRUE");
    tree_leaf_counts.push_back(leaf_false_count + leaf_true_count);
  }

  int64_t input_n_targets = node->i(Symbol("n_targets"));
  std::vector<int64_t> input_nodes_falsenodeids =
      node->is(Symbol("nodes_falsenodeids"));
  std::vector<int64_t> input_nodes_featureids =
      node->is(Symbol("nodes_featureids"));
  std::vector<double> input_nodes_hitrates = node->fs(Symbol("nodes_hitrates"));
  std::vector<int64_t> input_nodes_missing_value_tracks_true =
      node->is(Symbol("nodes_missing_value_tracks_true"));
  std::vector<std::string> input_nodes_modes = node->ss(Symbol("nodes_modes"));
  std::vector<int64_t> input_nodes_nodeids = node->is(Symbol("nodes_nodeids"));
  std::vector<int64_t> input_nodes_treeids = node->is(Symbol("nodes_treeids"));
  std::vector<int64_t> input_nodes_truenodeids =
      node->is(Symbol("nodes_truenodeids"));
  std::vector<double> input_nodes_values = node->fs(Symbol("nodes_values"));
  std::vector<int64_t> input_target_ids = node->is(Symbol("target_ids"));
  std::vector<int64_t> input_target_nodeids =
      node->is(Symbol("target_nodeids"));
  std::vector<int64_t> input_target_treeids =
      node->is(Symbol("target_treeids"));
  std::vector<double> input_target_weights = node->fs(Symbol("target_weights"));

  std::vector<std::vector<NodeID>> new_ids_list;
  for (const auto& removed_nodes : result_nodes_list) {
    std::vector<NodeID> new_ids;
    int id_ = 0;
    for (const auto& node : removed_nodes) {
      if (node == "LEAF_FALSE" || node == "LEAF_TRUE" || node == "BRANCH_LEQ") {
        new_ids.push_back({id_, node});
        id_++;
      } else {
        new_ids.push_back({-1, node});
      }
    }
    new_ids_list.push_back(new_ids);
  }

  // 构建 nodes_falsenodeids
  std::vector<int64_t> nodes_falsenodeids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].node != "REMOVED") {
        int64_t falsenodeid = new_ids[input_nodes_falsenodeids[ii]].id;
        nodes_falsenodeids.push_back(falsenodeid != -1 ? falsenodeid : 0);
      }
      ++i;
    }
  }

  // 构建 nodes_featureids
  std::vector<int64_t> nodes_featureids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_featureids.push_back(
            new_ids[i].node == "BRANCH_LEQ" ? input_nodes_featureids[ii] : 0);
      }
      ++i;
    }
  }

  // 构建 nodes_hitrates
  std::vector<double> nodes_hitrates;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_hitrates.push_back(input_nodes_hitrates[ii]);
      }
      ++i;
    }
  }

  // 构建 nodes_missing_value_tracks_true
  std::vector<int64_t> nodes_missing_value_tracks_true;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_missing_value_tracks_true.push_back(
            input_nodes_missing_value_tracks_true[ii]);
      }
      ++i;
    }
  }

  // 构建 nodes_modes
  std::vector<std::string> nodes_modes;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& new_ids = new_ids_list[tree_no];

    for (const auto& new_id : new_ids) {
      if (new_id.id != -1) {
        nodes_modes.push_back(new_id.node == "BRANCH_LEQ" ? "BRANCH_LEQ"
                                                          : "LEAF");
      }
    }
  }

  // 构建 nodes_nodeids
  std::vector<int64_t> nodes_nodeids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_nodeids.push_back(new_ids[i].id);
      }
      ++i;
    }
  }

  // 构建 nodes_treeids
  std::vector<int64_t> nodes_treeids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_treeids.push_back(input_nodes_treeids[ii]);
      }
      ++i;
    }
  }
  // 构建 nodes_truenodeids
  std::vector<int64_t> nodes_truenodeids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].node != "REMOVED") {
        int truenodeid = new_ids[input_nodes_truenodeids[ii]].id;
        nodes_truenodeids.push_back(truenodeid != -1 ? truenodeid : 0);
      }
      ++i;
    }
  }

  // 构建 nodes_values
  std::vector<double> nodes_values;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_values.push_back(
            new_ids[i].node == "BRANCH_LEQ" ? input_nodes_values[ii] : 0);
      }
      ++i;
    }
  }

  // 构建 target_ids
  std::vector<int64_t> target_ids(
      std::accumulate(tree_leaf_counts.begin(), tree_leaf_counts.end(), 0), 0);

  // 构建 target_nodeids
  std::vector<int64_t> target_nodeids;
  for (const auto& new_ids : new_ids_list) {
    for (const auto& new_id : new_ids) {
      if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
        target_nodeids.push_back(new_id.id);
      }
    }
  }

  // 构建 target_treeids
  std::vector<int64_t> target_treeids;
  for (size_t tree_no = 0; tree_no < tree_leaf_counts.size(); ++tree_no) {
    target_treeids.insert(target_treeids.end(), tree_leaf_counts[tree_no],
                          tree_no);
  }

  // 构建 target_weights
  std::vector<double> target_weights;
  double tw = 1.0 / tree_count;
  for (const auto& new_ids : new_ids_list) {
    for (const auto& new_id : new_ids) {
      if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
        target_weights.push_back(new_id.node == "LEAF_TRUE" ? tw : 0.0f);
      }
    }
  }

  node->is_(Symbol("nodes_falsenodeids"), std::move(nodes_falsenodeids));
  node->is_(Symbol("nodes_featureids"), std::move(nodes_featureids));
  node->fs_(Symbol("nodes_hitrates"), std::move(nodes_hitrates));
  node->is_(Symbol("nodes_missing_value_tracks_true"),
            std::move(nodes_missing_value_tracks_true));
  node->ss_(Symbol("nodes_modes"), std::move(nodes_modes));
  node->is_(Symbol("nodes_nodeids"), std::move(nodes_nodeids));
  node->is_(Symbol("nodes_treeids"), std::move(nodes_treeids));
  node->is_(Symbol("nodes_truenodeids"), std::move(nodes_truenodeids));
  node->fs_(Symbol("nodes_values"), std::move(nodes_values));
  node->is_(Symbol("target_ids"), std::move(target_ids));
  node->is_(Symbol("target_nodeids"), std::move(target_nodeids));
  node->is_(Symbol("target_treeids"), std::move(target_treeids));
  node->fs_(Symbol("target_weights"), std::move(target_weights));

  return true;
}

std::string DTPruneRule::apply(int threads_count, ModelProto& mp_in,
                               std::shared_ptr<Graph>& graph,
                               std::string& model_path, Node* treeNode,
                               uint8_t comparison_operator, float threshold) {
  auto tree_intervals = getTreeIntervals(treeNode);
  // only original regressor need
  if (model_path.find("reg") == std::string::npos) {
    threshold /= tree_intervals.size();
  }

  std::vector<std::vector<std::string>> result_nodes_list;
  for (const auto& interval : tree_intervals) {
    int start = std::get<0>(interval);
    int end = std::get<1>(interval);
    result_nodes_list.push_back(std::vector<std::string>(end - start, ""));
  }
  // std::ofstream outputfile(
  //   "/volumn/Retree_exp/DTPruneRule.csv",
  //   std::ios::app);
  // auto start = std::chrono::high_resolution_clock::now();
  if (threads_count == 1) {
    for (size_t tree_no = 0; tree_no < result_nodes_list.size(); ++tree_no) {
      auto& result_nodes = result_nodes_list[tree_no];
      const auto& tree_interval = tree_intervals[tree_no];
      pruning_loop(tree_no, tree_interval, result_nodes, treeNode,
                   comparison_operator, threshold);
    }
  } else {
    ThreadPool pool(threads_count);
    std::vector<std::future<void>> futures;
    futures.reserve(result_nodes_list.size());
    for (size_t tree_no = 0; tree_no < result_nodes_list.size(); ++tree_no) {
      futures.push_back(pool.enqueue([&, tree_no]() {
        auto& result_nodes = result_nodes_list[tree_no];
        const auto& tree_interval = tree_intervals[tree_no];
        // pruning_loop_optimized(tree_no, tree_interval, result_nodes,
        // treeNode,
        //              comparison_operator, threshold);
        pruning_loop(tree_no, tree_interval, result_nodes, treeNode,
                     comparison_operator, threshold);
      }));
    }
    for (auto& fut : futures) {
      fut.get();
    }
  }
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> prune_duration =
  //     end - start;
  // outputfile << model_path << "," << std::to_string(threshold) << "," <<
  // prune_duration.count() << "\n";
  if (processNode(treeNode, result_nodes_list, tree_intervals)) {
    return saveModelWithNewName(mp_in, graph, model_path,
                                "pruned" + std::to_string(threshold));
  } else {
    std::string new_model_path = model_path;
    return new_model_path;
  }
}

std::string DTPruneRule::match(std::string& model_path,
                               uint8_t comparison_operator, float threshold,
                               int threads_count) {
  std::string output_model_path = model_path;
  ModelProto mp_in;
  loadModel(&mp_in, model_path, true);
  std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

  bool found = false;
  Node* treeNode;
  graph->forEachNode([&found, &treeNode](Node* node) {
    if (node->hasAttribute(Symbol("target_treeids"))) {
      if (node->s(Symbol("post_transform")) == "NONE") {
        found = true;
        treeNode = node;
      }
    }
  });

  if (found)
    output_model_path = apply(threads_count, mp_in, graph, model_path, treeNode,
                              comparison_operator, threshold);

  return output_model_path;
}

/**
 * @brief random forest merge
 */
// Retree rules
ReTreeRule::ComparisonFunc ReTreeRule::comparison_funcs[] = {
    [](float x, float y) { return x == y; },
    [](float x, float y) { return x < y; },
    [](float x, float y) { return x <= y; },
    [](float x, float y) { return x > y; },
    [](float x, float y) { return x >= y; }};

// use in convert
void ReTreeRule::processNode(Node* node) {
  auto classlabels_int64s = node->is(Symbol("classlabels_int64s"));
  auto class_treeids = node->is(Symbol("class_treeids"));
  auto class_ids = node->is(Symbol("class_ids"));
  auto class_nodeids = node->is(Symbol("class_nodeids"));
  auto class_weights = node->fs(Symbol("class_weights"));

  std::unordered_set<int> unique_treeids(class_treeids.begin(),
                                         class_treeids.end());
  int n_trees = unique_treeids.size();

  // ** convert clf attributes 2 reg attributes
  int64_t stride =
      classlabels_int64s.size() == 2 ? 1 : classlabels_int64s.size();
  int nleaf = static_cast<int>(class_weights.size() / stride);

  // input_class_treeids 2 target_treeids
  std::vector<int64_t> target_treeids;
  target_treeids.reserve(nleaf);
  for (int64_t i = 0; i < nleaf; ++i) {
    target_treeids.push_back(class_treeids[i * stride]);
  }

  // input_class_ids 2 target_ids
  std::vector<int64_t> target_ids;
  target_ids.reserve(nleaf);
  for (int64_t i = 0; i < nleaf; ++i) {
    target_ids.push_back(class_ids[i * stride]);
  }

  // input_class_nodeids 2 target_nodeids
  std::vector<int64_t> target_nodeids;
  target_nodeids.reserve(nleaf);
  for (int64_t i = 0; i < nleaf; ++i) {
    target_nodeids.push_back(class_nodeids[i * stride]);
  }

  // input_class_weights 2 target_weights
  std::vector<double> target_weights;
  target_weights.reserve(nleaf);
  if (stride == 1) {
    for (auto w : class_weights) {
      target_weights.push_back((w > 0.5 / n_trees) ? 1.0 : 0.0);
    }
  } else {
    for (int i = 0; i < nleaf; ++i) {
      auto start_it = class_weights.begin() + (i * stride);
      auto end_it = class_weights.begin() + ((i + 1) * stride);

      auto max_it = std::max_element(start_it, end_it);
      int index = std::distance(start_it, max_it);
      // ** case: wine_quality classlabels_int64s [3,4,5,6,7,8]
      target_weights.push_back(static_cast<double>(classlabels_int64s[index]));
    }
  }

  node->removeAttribute(Symbol("classlabels_int64s"));
  node->removeAttribute(Symbol("class_treeids"));
  node->removeAttribute(Symbol("class_ids"));
  node->removeAttribute(Symbol("class_nodeids"));
  node->removeAttribute(Symbol("class_weights"));

  node->i_(Symbol("n_targets"), 1);
  node->is_(Symbol("target_treeids"), std::move(target_treeids));
  node->is_(Symbol("target_ids"), std::move(target_ids));
  node->is_(Symbol("target_nodeids"), std::move(target_nodeids));
  node->fs_(Symbol("target_weights"), std::move(target_weights));
}

// use in prune
void ReTreeRule::processNode(
    Node* node, std::vector<std::vector<std::string>>& result_nodes_list,
    std::vector<std::tuple<int, int>>& tree_intervals) {
  const int tree_count = tree_intervals.size();
  std::vector<int> tree_leaf_counts;
  for (const auto& removed_nodes : result_nodes_list) {
    int leaf_false_count =
        std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_FALSE");
    int leaf_true_count =
        std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_TRUE");
    tree_leaf_counts.push_back(leaf_false_count + leaf_true_count);
  }

  int64_t input_n_targets = node->i(Symbol("n_targets"));
  std::vector<int64_t> input_nodes_falsenodeids =
      node->is(Symbol("nodes_falsenodeids"));
  std::vector<int64_t> input_nodes_featureids =
      node->is(Symbol("nodes_featureids"));
  std::vector<double> input_nodes_hitrates = node->fs(Symbol("nodes_hitrates"));
  std::vector<int64_t> input_nodes_missing_value_tracks_true =
      node->is(Symbol("nodes_missing_value_tracks_true"));
  std::vector<std::string> input_nodes_modes = node->ss(Symbol("nodes_modes"));
  std::vector<int64_t> input_nodes_nodeids = node->is(Symbol("nodes_nodeids"));
  std::vector<int64_t> input_nodes_treeids = node->is(Symbol("nodes_treeids"));
  std::vector<int64_t> input_nodes_truenodeids =
      node->is(Symbol("nodes_truenodeids"));
  std::vector<double> input_nodes_values = node->fs(Symbol("nodes_values"));
  std::vector<int64_t> input_target_ids = node->is(Symbol("target_ids"));
  std::vector<int64_t> input_target_nodeids =
      node->is(Symbol("target_nodeids"));
  std::vector<int64_t> input_target_treeids =
      node->is(Symbol("target_treeids"));
  std::vector<double> input_target_weights = node->fs(Symbol("target_weights"));

  std::vector<std::vector<NodeID>> new_ids_list;
  for (const auto& removed_nodes : result_nodes_list) {
    std::vector<NodeID> new_ids;
    int id_ = 0;
    for (const auto& node : removed_nodes) {
      if (node == "LEAF_FALSE" || node == "LEAF_TRUE" || node == "BRANCH_LEQ") {
        new_ids.push_back({id_, node});
        id_++;
      } else {
        new_ids.push_back({-1, node});
      }
    }
    new_ids_list.push_back(new_ids);
  }

  // 构建 nodes_falsenodeids
  std::vector<int64_t> nodes_falsenodeids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].node != "REMOVED") {
        int64_t falsenodeid = new_ids[input_nodes_falsenodeids[ii]].id;
        nodes_falsenodeids.push_back(falsenodeid != -1 ? falsenodeid : 0);
      }
      ++i;
    }
  }

  // 构建 nodes_featureids
  std::vector<int64_t> nodes_featureids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_featureids.push_back(
            new_ids[i].node == "BRANCH_LEQ" ? input_nodes_featureids[ii] : 0);
      }
      ++i;
    }
  }

  // 构建 nodes_hitrates
  std::vector<double> nodes_hitrates;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_hitrates.push_back(input_nodes_hitrates[ii]);
      }
      ++i;
    }
  }

  // 构建 nodes_missing_value_tracks_true
  std::vector<int64_t> nodes_missing_value_tracks_true;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_missing_value_tracks_true.push_back(
            input_nodes_missing_value_tracks_true[ii]);
      }
      ++i;
    }
  }

  // 构建 nodes_modes
  std::vector<std::string> nodes_modes;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& new_ids = new_ids_list[tree_no];

    for (const auto& new_id : new_ids) {
      if (new_id.id != -1) {
        nodes_modes.push_back(new_id.node == "BRANCH_LEQ" ? "BRANCH_LEQ"
                                                          : "LEAF");
      }
    }
  }

  // 构建 nodes_nodeids
  std::vector<int64_t> nodes_nodeids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_nodeids.push_back(new_ids[i].id);
      }
      ++i;
    }
  }

  // 构建 nodes_treeids
  std::vector<int64_t> nodes_treeids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_treeids.push_back(input_nodes_treeids[ii]);
      }
      ++i;
    }
  }
  // 构建 nodes_truenodeids
  std::vector<int64_t> nodes_truenodeids;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].node != "REMOVED") {
        int truenodeid = new_ids[input_nodes_truenodeids[ii]].id;
        nodes_truenodeids.push_back(truenodeid != -1 ? truenodeid : 0);
      }
      ++i;
    }
  }

  // 构建 nodes_values
  std::vector<double> nodes_values;
  for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
    const auto& [tree_start, tree_end] = tree_intervals[tree_no];
    const auto& new_ids = new_ids_list[tree_no];
    int i = 0;
    for (size_t ii = tree_start; ii < tree_end; ++ii) {
      if (new_ids[i].id != -1) {
        nodes_values.push_back(
            new_ids[i].node == "BRANCH_LEQ" ? input_nodes_values[ii] : 0);
      }
      ++i;
    }
  }

  // 构建 target_ids
  std::vector<int64_t> target_ids(
      std::accumulate(tree_leaf_counts.begin(), tree_leaf_counts.end(), 0), 0);

  // 构建 target_nodeids
  std::vector<int64_t> target_nodeids;
  for (const auto& new_ids : new_ids_list) {
    for (const auto& new_id : new_ids) {
      if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
        target_nodeids.push_back(new_id.id);
      }
    }
  }

  // 构建 target_treeids
  std::vector<int64_t> target_treeids;
  for (size_t tree_no = 0; tree_no < tree_leaf_counts.size(); ++tree_no) {
    target_treeids.insert(target_treeids.end(), tree_leaf_counts[tree_no],
                          tree_no);
  }

  // 构建 target_weights
  std::vector<double> target_weights;
  double tw = 1.0 / tree_count;
  for (const auto& new_ids : new_ids_list) {
    for (const auto& new_id : new_ids) {
      if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
        target_weights.push_back(new_id.node == "LEAF_TRUE" ? tw : 0.0f);
      }
    }
  }

  node->is_(Symbol("nodes_falsenodeids"), std::move(nodes_falsenodeids));
  node->is_(Symbol("nodes_featureids"), std::move(nodes_featureids));
  node->fs_(Symbol("nodes_hitrates"), std::move(nodes_hitrates));
  node->is_(Symbol("nodes_missing_value_tracks_true"),
            std::move(nodes_missing_value_tracks_true));
  node->ss_(Symbol("nodes_modes"), std::move(nodes_modes));
  node->is_(Symbol("nodes_nodeids"), std::move(nodes_nodeids));
  node->is_(Symbol("nodes_treeids"), std::move(nodes_treeids));
  node->is_(Symbol("nodes_truenodeids"), std::move(nodes_truenodeids));
  node->fs_(Symbol("nodes_values"), std::move(nodes_values));
  node->is_(Symbol("target_ids"), std::move(target_ids));
  node->is_(Symbol("target_nodeids"), std::move(target_nodeids));
  node->is_(Symbol("target_treeids"), std::move(target_treeids));
  node->fs_(Symbol("target_weights"), std::move(target_weights));
}

// change graph & node (proto layer)
void ReTreeRule::convertModelProto(ModelProto& mp_in, Node* node,
                                   std::string& model_path) {
  auto g_m = mp_in.mutable_graph();
  auto input = g_m->input();

  // construct output
  onnx::ValueInfoProto output;
  TensorShapeProto_Dimension output_dim1, output_dim2;
  output_dim2.set_dim_value(1);
  output.set_name("regression");
  output.mutable_type()->mutable_tensor_type()->set_elem_type(1);
  *output.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim() =
      output_dim1;
  *output.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim() =
      output_dim2;

  // construct input
  onnx::NodeProto* p_n_clf;
  for (int i = 0; i < g_m->node_size(); ++i) {
    auto node = g_m->mutable_node(i);
    if (node->op_type() == "TreeEnsembleClassifier") {
      p_n_clf = node;
    }
  }

  // construct TreeEnsembleRegressor
  onnx::NodeProto p_n;
  p_n.set_op_type("TreeEnsembleRegressor");
  p_n.set_name(p_n_clf->name());
  p_n.set_domain(p_n_clf->domain());
  auto inputs = p_n_clf->input();
  for (auto input : inputs) {
    p_n.add_input(input);
  }
  p_n.add_output(output.name());
  for (auto attr_name : node->attributeNames()) {
    addAttribute(&p_n, node, attr_name);
  }

  // Remove TreeEnsembleClassifier and subsequent Nodes
  for (int i = g_m->node_size() - 1; i >= 0; i--) {
    auto node = g_m->node(i);
    if (node.op_type() == "TreeEnsembleClassifier") {
      g_m->mutable_node()->DeleteSubrange(i, 1);
    } else if (node.op_type() == "ZipMap") {
      if (node.input()[0] == "probabilities") {
        g_m->mutable_node()->DeleteSubrange(i, 1);
      }
    } else if (node.op_type() == "Cast") {
      if (node.input()[0] == "label") {
        g_m->mutable_node()->DeleteSubrange(i, 1);
      }
    }
  }
  // replace TreeEnsembleClassifier with TreeEnsembleRegressor
  g_m->clear_output();
  *g_m->add_node() = p_n;
  *g_m->add_output() = output;

  onnx::checker::check_model(mp_in);

  // auto new_model_path = getNewmodelname(model_path, "reg");
  // saveModel(&mp_in, new_model_path);
  // return new_model_path;
}

std::vector<std::tuple<int, int>> ReTreeRule::getTreeIntervals(Node* node) {
  std::vector<int> tree_roots;
  std::vector<int64_t> nodes_treeids = node->is(Symbol("nodes_treeids"));
  int next_tree_id = 0;
  for (size_t i = 0; i < nodes_treeids.size(); ++i) {
    if (nodes_treeids[i] == next_tree_id) {
      next_tree_id++;
      tree_roots.push_back(i);
    }
  }

  std::vector<std::tuple<int, int>> tree_intervals;
  for (size_t i = 0; i < tree_roots.size(); ++i) {
    int end;
    if (i == tree_roots.size() - 1) {
      end = nodes_treeids.size();
    } else {
      end = tree_roots[i + 1];
    }
    tree_intervals.push_back(std::make_tuple(tree_roots[i], end));
  }

  return tree_intervals;
}

std::vector<std::tuple<int, int>> ReTreeRule::get_target_tree_intervals(
    Node* node) {
  std::vector<int> target_tree_roots;
  std::vector<int64_t> target_treeids = node->is(Symbol("target_treeids"));
  int next_tree_id = 0;
  for (int i = 0; i < target_treeids.size(); ++i) {
    if (target_treeids[i] == next_tree_id) {
      next_tree_id++;
      target_tree_roots.push_back(i);
    }
  }

  std::vector<std::tuple<int, int>> target_tree_intervals;
  for (int i = 0; i < target_tree_roots.size(); ++i) {
    int end;
    if (i == target_tree_roots.size() - 1) {
      end = target_treeids.size();
    } else {
      end = target_tree_roots[i + 1];
    }
    target_tree_intervals.push_back(std::make_tuple(target_tree_roots[i], end));
  }
  return target_tree_intervals;
}

std::shared_ptr<TreeNode> ReTreeRule::model2tree(
    Node* treeNode, int64_t root_node_id, std::tuple<int, int>& tree_interval,
    std::tuple<int, int>& target_tree_interval) {
  const auto& input_nodes_falsenodeids_ =
      treeNode->is(Symbol("nodes_falsenodeids"));
  const auto& input_nodes_featureids_ =
      treeNode->is(Symbol("nodes_featureids"));
  const auto& input_nodes_hitrates_ = treeNode->fs(Symbol("nodes_hitrates"));
  const auto& input_nodes_modes_ = treeNode->ss(Symbol("nodes_modes"));
  const auto& input_nodes_truenodeids_ =
      treeNode->is(Symbol("nodes_truenodeids"));
  const auto& input_nodes_values_ = treeNode->fs(Symbol("nodes_values"));
  const auto& input_target_nodeids_ = treeNode->is(Symbol("target_nodeids"));
  const auto& input_target_weights_ = treeNode->fs(Symbol("target_weights"));

  int tree_start = std::get<0>(tree_interval);
  int tree_end = std::get<1>(tree_interval);
  int target_tree_start = std::get<0>(target_tree_interval);
  int target_tree_end = std::get<1>(target_tree_interval);

  std::vector<int64_t> input_nodes_falsenodeids(
      input_nodes_falsenodeids_.begin() + tree_start,
      input_nodes_falsenodeids_.begin() + tree_end);
  std::vector<int64_t> input_nodes_featureids(
      input_nodes_featureids_.begin() + tree_start,
      input_nodes_featureids_.begin() + tree_end);
  std::vector<double> input_nodes_hitrates(
      input_nodes_hitrates_.begin() + tree_start,
      input_nodes_hitrates_.begin() + tree_end);
  std::vector<std::string> input_nodes_modes(
      input_nodes_modes_.begin() + tree_start,
      input_nodes_modes_.begin() + tree_end);
  std::vector<int64_t> input_nodes_truenodeids(
      input_nodes_truenodeids_.begin() + tree_start,
      input_nodes_truenodeids_.begin() + tree_end);
  std::vector<double> input_nodes_values(
      input_nodes_values_.begin() + tree_start,
      input_nodes_values_.begin() + tree_end);
  std::vector<int64_t> input_target_nodeids(
      input_target_nodeids_.begin() + target_tree_start,
      input_target_nodeids_.begin() + target_tree_end);
  std::vector<double> input_target_weights(
      input_target_weights_.begin() + target_tree_start,
      input_target_weights_.begin() + target_tree_end);

  std::unordered_map<int, int> input_target_nodeid_map;
  for (size_t i = 0; i < input_target_nodeids.size(); ++i) {
    input_target_nodeid_map[input_target_nodeids[i]] = static_cast<int>(i);
  }

  struct StackElement {
    int64_t node_id;
    std::weak_ptr<TreeNode> parent;
    bool is_left;
  };
  std::stack<StackElement> stack;
  std::shared_ptr<TreeNode> root = nullptr;

  stack.push({root_node_id, std::weak_ptr<TreeNode>(), false});

  while (!stack.empty()) {
    auto [current_id, parent_weak, is_left] = stack.top();
    stack.pop();

    int64_t feature_id = input_nodes_featureids[current_id];
    std::string mode = input_nodes_modes[current_id];
    double value = input_nodes_values[current_id];
    int samples = static_cast<int>(input_nodes_hitrates[current_id]);

    std::optional<int64_t> target_id;
    auto it = input_target_nodeid_map.find(current_id);
    if (it != input_target_nodeid_map.end()) {
      target_id = it->second;
    }
    std::optional<double> target_weight;
    if (target_id.has_value()) {
      target_weight = input_target_weights[target_id.value()];
    } else {
      target_weight = std::nullopt;
    }

    auto node = std::make_shared<TreeNode>(current_id, feature_id, mode, value,
                                           target_id, target_weight, samples);
    node->parent = parent_weak;

    if (auto parent = parent_weak.lock()) {
      if (is_left) {
        parent->left = node;
      } else {
        parent->right = node;
      }
    } else {
      root = node;
    }

    if (mode != "LEAF") {
      int64_t left_id = input_nodes_truenodeids[current_id];
      int64_t right_id = input_nodes_falsenodeids[current_id];
      stack.push({right_id, node, false});
      stack.push({left_id, node, true});
    }
  }

  return root;
}

std::vector<std::shared_ptr<TreeNode>> ReTreeRule::model2trees(Node* treeNode,
                                                               int nthreads) {
  auto tree_intervals = getTreeIntervals(treeNode);
  auto target_tree_intervals = get_target_tree_intervals(treeNode);
  std::vector<std::shared_ptr<TreeNode>> trees;
  trees.resize(tree_intervals.size());
  if (nthreads == 1) {
    for (size_t i = 0; i < tree_intervals.size(); i++) {
      std::shared_ptr<TreeNode> root =
          model2tree(treeNode, 0, tree_intervals[i], target_tree_intervals[i]);
      trees[i] = root;
    }
  } else {
    ThreadPool pool(nthreads);
    std::vector<std::future<void>> futures;
    futures.reserve(tree_intervals.size());
    for (size_t i = 0; i < tree_intervals.size(); i++) {
      futures.push_back(pool.enqueue([&, i]() {
        std::shared_ptr<TreeNode> root = model2tree(
            treeNode, 0, tree_intervals[i], target_tree_intervals[i]);
        trees[i] = root;
      }));
    }
    for (auto& fut : futures) {
      fut.get();
    }
  }
  return trees;
}

void ReTreeRule::toTrees(Node* treeNode, TreeEnsembleRegressor& regressor) {
  treeNode->is_(Symbol("nodes_falsenodeids"),
                std::move(regressor.nodes_falsenodeids));
  treeNode->is_(Symbol("nodes_featureids"),
                std::move(regressor.nodes_featureids));
  treeNode->fs_(Symbol("nodes_hitrates"), std::move(regressor.nodes_hitrates));
  treeNode->is_(Symbol("nodes_missing_value_tracks_true"),
                std::move(regressor.nodes_missing_value_tracks_true));
  treeNode->ss_(Symbol("nodes_modes"), std::move(regressor.nodes_modes));
  treeNode->is_(Symbol("nodes_nodeids"), std::move(regressor.nodes_nodeids));
  treeNode->is_(Symbol("nodes_treeids"), std::move(regressor.nodes_treeids));
  treeNode->is_(Symbol("nodes_truenodeids"),
                std::move(regressor.nodes_truenodeids));
  treeNode->fs_(Symbol("nodes_values"), std::move(regressor.nodes_values));
  treeNode->is_(Symbol("target_ids"), std::move(regressor.target_ids));
  treeNode->is_(Symbol("target_nodeids"), std::move(regressor.target_nodeids));
  treeNode->is_(Symbol("target_treeids"), std::move(regressor.target_treeids));
  treeNode->fs_(Symbol("target_weights"), std::move(regressor.target_weights));
}

// optimized loop version
void ReTreeRule::pruning(size_t tree_no,
                         const std::tuple<int, int>& tree_interval,
                         std::vector<std::string>& result_nodes, Node* treeNode,
                         uint8_t comparison_operator, float threshold) {
  int tree_start = std::get<0>(tree_interval);
  int tree_end = std::get<1>(tree_interval);
  size_t n_nodes = tree_end - tree_start;

  const auto& left_nodess = treeNode->is(Symbol("nodes_truenodeids"));
  const auto& right_nodess = treeNode->is(Symbol("nodes_falsenodeids"));
  const auto& node_typess = treeNode->ss(Symbol("nodes_modes"));
  const int64_t* left_nodes_ptr = left_nodess.data() + tree_start;
  const int64_t* right_nodes_ptr = right_nodess.data() + tree_start;
  const std::string* node_types_ptr = node_typess.data() + tree_start;

  const auto& target_treeids = treeNode->is(Symbol("target_treeids"));
  const auto& target_nodeids = treeNode->is(Symbol("target_nodeids"));
  const auto& target_weights = treeNode->fs(Symbol("target_weights"));

  std::unordered_map<int, size_t> target_index;
  target_index.reserve(target_nodeids.size());
  for (size_t ti = 0; ti < target_nodeids.size(); ++ti) {
    if (target_treeids[ti] == tree_no) {
      target_index[target_nodeids[ti]] = ti;
    }
  }

  std::vector<int> computed_result(n_nodes, -1);

  struct StackFrame {
    int64_t node_id;
    bool visited;
  };
  std::vector<StackFrame> stack;
  stack.reserve(n_nodes);
  stack.push_back({0, false});

  while (!stack.empty()) {
    auto& frame = stack.back();
    size_t curr_id = static_cast<size_t>(frame.node_id);

    if (!frame.visited) {
      if (node_types_ptr[curr_id] == "LEAF") {
        auto it = target_index.find(static_cast<int>(curr_id));
        int target_idx = (it != target_index.end() ? it->second : -1);
        int result = comparison_funcs[comparison_operator](
            target_weights[target_idx], threshold);
        result_nodes[curr_id] = (result == 1) ? "LEAF_TRUE" : "LEAF_FALSE";
        computed_result[curr_id] = result;
        stack.pop_back();
      } else {
        result_nodes[curr_id] = node_types_ptr[curr_id];
        frame.visited = true;
        stack.push_back({right_nodes_ptr[curr_id], false});
        stack.push_back({left_nodes_ptr[curr_id], false});
      }
    } else {
      size_t left_id = static_cast<size_t>(left_nodes_ptr[curr_id]);
      size_t right_id = static_cast<size_t>(right_nodes_ptr[curr_id]);
      int left_res = computed_result[left_id];
      int right_res = computed_result[right_id];

      if (left_res == 0 && right_res == 0) {
        result_nodes[curr_id] = "LEAF_FALSE";
        result_nodes[left_id] = "REMOVED";
        result_nodes[right_id] = "REMOVED";
        computed_result[curr_id] = 0;
      } else if (left_res == 1 && right_res == 1) {
        result_nodes[curr_id] = "LEAF_TRUE";
        result_nodes[left_id] = "REMOVED";
        result_nodes[right_id] = "REMOVED";
        computed_result[curr_id] = 1;
      } else {
        computed_result[curr_id] = 2;
      }
      stack.pop_back();
    }
  }
}

void ReTreeRule::update_result(
    std::shared_ptr<TreeNode> node, int path_length,
    std::shared_ptr<TreeNode> root,
    std::vector<std::tuple<std::shared_ptr<TreeNode>, int>>& result) {
  if (node->mode == "LEAF")
    return;
  assert(node->feature_id == root->feature_id);

  if (result.empty()) {
    result.push_back(std::make_tuple(node, path_length));
    return;
  }

  auto last = std::get<0>(result.back());
  assert(node->feature_id == last->feature_id);
  float new_delta = std::abs(round(round(node->value) - round(root->value)));
  float old_delta = std::abs(round(round(last->value) - round(root->value)));

  if (new_delta == old_delta) {
    result.push_back(std::make_tuple(node, path_length));
    return;
  }
  if (new_delta < old_delta) {
    result.clear();
    result.push_back(std::make_tuple(node, path_length));
    return;
  }
  return;
}

int ReTreeRule::find_merge_nodes(
    std::shared_ptr<TreeNode> node, int path_length,
    std::shared_ptr<TreeNode> root, bool left_branch,
    std::vector<std::tuple<std::shared_ptr<TreeNode>, int>>& result) {
  if (node->mode == "LEAF") {
    return (node->target_weight.value() == 0) ? M_FALSE : M_TRUE;
  }

  bool same_feature = (node->feature_id == root->feature_id);
  int left_merge_stats = M_NO, right_merge_stats = M_NO;

  if (!same_feature || (same_feature && !left_branch)) {
    left_merge_stats = find_merge_nodes(node->left, path_length + 1, root,
                                        left_branch, result);
    if (left_merge_stats == M_NO)
      return M_NO;
  }

  if (!same_feature || (same_feature && left_branch)) {
    right_merge_stats = find_merge_nodes(node->right, path_length + 1, root,
                                         left_branch, result);
    if (right_merge_stats == M_NO)
      return M_NO;
  }

  if (!same_feature) {
    if (left_merge_stats != right_merge_stats)
      return M_NO;
    else {
      // update_result(node, path_length, root, result);
      return left_merge_stats;
    }
  }

  if (!left_branch) {
    if (left_merge_stats != M_NO)
      update_result(node, path_length, root, result);
    return left_merge_stats;
  }

  if (left_branch) {
    if (right_merge_stats != M_NO)
      update_result(node, path_length, root, result);
    return right_merge_stats;
  }
  return M_NO;
}

void ReTreeRule::merge(std::shared_ptr<TreeNode> root,
                       std::vector<std::shared_ptr<TreeNode>> nodes,
                       bool left_branch) {
  assert(root->mode == "BRANCH_LEQ");
  assert(!nodes.empty());
  for (auto node : nodes) {
    assert(node->mode == "BRANCH_LEQ");
    assert(node->feature_id == root->feature_id);
  }

  root->value = nodes[0]->value;

  for (auto node : nodes) {
    if (left_branch) {
      auto parent = node->parent.lock();
      if (!parent) {
        throw std::runtime_error("Parent node is no longer available.");
      }

      std::shared_ptr<TreeNode> left = node->left;

      if (node == parent->left)
        parent->left = left;
      else
        parent->right = left;
      left->parent = parent;

    } else {
      auto parent = node->parent.lock();
      if (!parent) {
        throw std::runtime_error("Parent node is no longer available.");
      }
      std::shared_ptr<TreeNode> right = node->right;

      if (node == parent->left)
        parent->left = right;
      else
        parent->right = right;
      right->parent = parent;
    }

    node->parent.reset();
    node->left.reset();
    node->right.reset();
  }
}

void ReTreeRule::dfs(std::shared_ptr<TreeNode> node) {
  if (node->mode == "LEAF") {
    return;
  }
  dfs(node->left);
  dfs(node->right);
  std::vector<std::tuple<std::shared_ptr<TreeNode>, int>> left_merge_nodes,
      right_merge_nodes;
  int left_merge_stats =
      find_merge_nodes(node->left, 1, node, true, left_merge_nodes);
  if (left_merge_stats == M_NO) {
    return;
  }

  int right_merge_stats =
      find_merge_nodes(node->right, 1, node, false, right_merge_nodes);
  if (right_merge_stats == M_NO) {
    return;
  }

  if (left_merge_stats != right_merge_stats) {
    return;
  }

  int max_left_path_length = 0, max_right_path_length = 0;
  for (auto& entry : left_merge_nodes) {
    max_left_path_length = std::max(max_left_path_length, std::get<1>(entry));
  }
  for (auto& entry : right_merge_nodes) {
    max_right_path_length = std::max(max_right_path_length, std::get<1>(entry));
  }

  std::vector<std::shared_ptr<TreeNode>> left_merge_nodes_only,
      right_merge_nodes_only;
  for (auto& entry : left_merge_nodes) {
    left_merge_nodes_only.push_back(std::get<0>(entry));
  }
  for (auto& entry : right_merge_nodes) {
    right_merge_nodes_only.push_back(std::get<0>(entry));
  }

  if (!left_merge_nodes_only.empty() && !right_merge_nodes_only.empty()) {
    if (CONSERVATIVE) {
      return;
    }
    if (max_left_path_length > max_right_path_length) {
      merge(node, left_merge_nodes_only, true);
    } else {
      merge(node, right_merge_nodes_only, false);
    }
    return;
  }

  if (!left_merge_nodes_only.empty()) {
    merge(node, left_merge_nodes_only, true);
    return;
  }

  merge(node, right_merge_nodes_only, false);
}

std::string ReTreeRule::apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                              Node* treeNode, std::string& model_path,
                              uint8_t comparison_operator, float threshold,
                              int nthreads, int opt_level) {
  // step 1: convert clf model to reg
  std::stringstream threshold_stream;
  threshold_stream << std::fixed << std::setprecision(3) << threshold;
  std::string prune_suffix = "prune_" + threshold_stream.str();
  std::string merge_suffix = "merge_" + threshold_stream.str();

  // todo: convert classlabels_strings
  if (treeNode->hasAttribute(Symbol("classlabels_strings"))) {
    return model_path;
  }
  bool clf = false;
  if (treeNode->hasAttribute(Symbol("classlabels_int64s"))) {
    processNode(treeNode);
    convertModelProto(mp_in, treeNode, model_path);
    clf = true;
    // update graph & treeNode
    graph = std::move(ImportModelProto(mp_in));
    for (Node* node : graph->nodes()) {
      if (node->hasAttribute(Symbol("target_treeids"))) {
        treeNode = node;
      }
    }
  }

  auto tree_intervals = getTreeIntervals(treeNode);
  if (!clf)
    threshold /= tree_intervals.size();

  // step 2: prune
  std::vector<std::vector<std::string>> result_nodes_list;
  for (const auto& interval : tree_intervals) {
    int start = std::get<0>(interval);
    int end = std::get<1>(interval);
    result_nodes_list.push_back(std::vector<std::string>(end - start, ""));
  }

  if (nthreads == 1) {
    for (size_t tree_no = 0; tree_no < result_nodes_list.size(); ++tree_no) {
      auto& result_nodes = result_nodes_list[tree_no];
      const auto& tree_interval = tree_intervals[tree_no];
      pruning(tree_no, tree_interval, result_nodes, treeNode,
              comparison_operator, threshold);
    }
  } else {
    ThreadPool pool(nthreads);
    std::vector<std::future<void>> futures;
    futures.reserve(result_nodes_list.size());
    for (size_t tree_no = 0; tree_no < result_nodes_list.size(); ++tree_no) {
      futures.push_back(pool.enqueue([&, tree_no]() {
        auto& result_nodes = result_nodes_list[tree_no];
        const auto& tree_interval = tree_intervals[tree_no];
        pruning(tree_no, tree_interval, result_nodes, treeNode,
                comparison_operator, threshold);
      }));
    }
    for (auto& fut : futures) {
      fut.get();
    }
  }
  processNode(treeNode, result_nodes_list, tree_intervals);
  if (opt_level == 0) {
    return saveModelWithNewName(mp_in, graph, model_path, prune_suffix);
  }

  // step 3: merge
  auto roots = model2trees(treeNode, nthreads);
  for (size_t i = 0; i < roots.size(); i++) {
    dfs(roots[i]);
  }

  auto regressor = TreeEnsembleRegressor::from_trees(roots);
  toTrees(treeNode, regressor);

  return saveModelWithNewName(mp_in, graph, model_path, merge_suffix);
}

std::string ReTreeRule::match(std::string& model_path,
                              uint8_t comparison_operator, float threshold,
                              int nthreads, int opt_level) {
  std::string output_model_path = model_path;
  ModelProto mp_in;
  loadModel(&mp_in, model_path, true);
  std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));
  bool found = false;
  Node* treeNode = nullptr;

  graph->forEachNode([&found, &treeNode](Node* node) {
    // std::cout << node->name() << std::endl;
    if ((node->hasAttribute(Symbol("class_treeids")) ||
         node->hasAttribute(Symbol("target_treeids"))) &&
        node->s(Symbol("post_transform")) == "NONE") {
      found = true;
      treeNode = node;
    }
  });
  if (found)
    output_model_path =
        apply(mp_in, graph, treeNode, model_path, comparison_operator,
              threshold, nthreads, opt_level);

  return output_model_path;
}

// DTMergeRule & DTNaiveMergeRule
std::shared_ptr<TreeNode> DTMergeRule::model2tree_recursive(
    Node* treeNode, int64_t node_id, std::shared_ptr<TreeNode> parent,
    std::tuple<int, int>& tree_interval,
    std::tuple<int, int>& target_tree_interval) {
  int tree_start = std::get<0>(tree_interval);
  int tree_end = std::get<1>(tree_interval);
  int target_tree_start = std::get<0>(target_tree_interval);
  int target_tree_end = std::get<1>(target_tree_interval);

  const auto& input_nodes_falsenodeids_ =
      treeNode->is(Symbol("nodes_falsenodeids"));
  const auto& input_nodes_featureids_ =
      treeNode->is(Symbol("nodes_featureids"));
  const auto& input_nodes_hitrates_ = treeNode->fs(Symbol("nodes_hitrates"));
  const auto& input_nodes_modes_ = treeNode->ss(Symbol("nodes_modes"));
  const auto& input_nodes_truenodeids_ =
      treeNode->is(Symbol("nodes_truenodeids"));
  const auto& input_nodes_values_ = treeNode->fs(Symbol("nodes_values"));
  const auto& input_target_nodeids_ = treeNode->is(Symbol("target_nodeids"));
  const auto& input_target_weights_ = treeNode->fs(Symbol("target_weights"));

  std::vector<int64_t> input_nodes_falsenodeids(
      input_nodes_falsenodeids_.begin() + tree_start,
      input_nodes_falsenodeids_.begin() + tree_end);
  std::vector<int64_t> input_nodes_featureids(
      input_nodes_featureids_.begin() + tree_start,
      input_nodes_featureids_.begin() + tree_end);
  std::vector<double> input_nodes_hitrates(
      input_nodes_hitrates_.begin() + tree_start,
      input_nodes_hitrates_.begin() + tree_end);
  std::vector<std::string> input_nodes_modes(
      input_nodes_modes_.begin() + tree_start,
      input_nodes_modes_.begin() + tree_end);
  std::vector<int64_t> input_nodes_truenodeids(
      input_nodes_truenodeids_.begin() + tree_start,
      input_nodes_truenodeids_.begin() + tree_end);
  std::vector<double> input_nodes_values(
      input_nodes_values_.begin() + tree_start,
      input_nodes_values_.begin() + tree_end);
  std::vector<int64_t> input_target_nodeids(
      input_target_nodeids_.begin() + target_tree_start,
      input_target_nodeids_.begin() + target_tree_end);
  std::vector<double> input_target_weights(
      input_target_weights_.begin() + target_tree_start,
      input_target_weights_.begin() + target_tree_end);

  std::unordered_map<int, int> input_target_nodeid_map;
  for (size_t i = 0; i < input_target_nodeids.size(); ++i) {
    input_target_nodeid_map[input_target_nodeids[i]] = static_cast<int>(i);
  }

  size_t id = node_id;
  int64_t feature_id = input_nodes_featureids[id];
  std::string mode = input_nodes_modes[id];
  double value = input_nodes_values[id];
  int samples = static_cast<int>(input_nodes_hitrates[id]);

  std::optional<int64_t> target_id;
  auto it = input_target_nodeid_map.find(id);
  if (it != input_target_nodeid_map.end()) {
    target_id = it->second;
  } else {
    target_id = std::nullopt;
  }
  std::optional<double> target_weight;
  if (target_id.has_value()) {
    target_weight = input_target_weights[target_id.value()];
  } else {
    target_weight = std::nullopt;
  }

  std::shared_ptr<TreeNode> node = std::make_shared<TreeNode>(
      id, feature_id, mode, value, target_id, target_weight, samples);
  node->parent = parent;

  if (mode != "LEAF") {
    int64_t left_node_id = input_nodes_truenodeids[id];
    std::shared_ptr<TreeNode> left_node = model2tree_recursive(
        treeNode, left_node_id, node, tree_interval, target_tree_interval);
    node->left = left_node;

    int64_t right_node_id = input_nodes_falsenodeids[id];
    std::shared_ptr<TreeNode> right_node = model2tree_recursive(
        treeNode, right_node_id, node, tree_interval, target_tree_interval);
    node->right = right_node;
  }

  return node;
}

void DTMergeRule::dfs(std::shared_ptr<TreeNode> node) {
  if (node->mode == "LEAF") {
    return;
  }
  dfs(node->left);
  dfs(node->right);
  std::vector<std::tuple<std::shared_ptr<TreeNode>, int>> left_merge_nodes,
      right_merge_nodes;
  int left_merge_stats =
      ReTreeRule::find_merge_nodes(node->left, 1, node, true, left_merge_nodes);
  if (left_merge_stats == M_NO) {
    return;
  }

  int right_merge_stats = ReTreeRule::find_merge_nodes(
      node->right, 1, node, false, right_merge_nodes);
  if (right_merge_stats == M_NO) {
    return;
  }

  if (left_merge_stats != right_merge_stats) {
    return;
  }

  int max_left_path_length = 0, max_right_path_length = 0;
  for (auto& entry : left_merge_nodes) {
    max_left_path_length = std::max(max_left_path_length, std::get<1>(entry));
  }
  for (auto& entry : right_merge_nodes) {
    max_right_path_length = std::max(max_right_path_length, std::get<1>(entry));
  }

  std::vector<std::shared_ptr<TreeNode>> left_merge_nodes_only,
      right_merge_nodes_only;
  for (auto& entry : left_merge_nodes) {
    left_merge_nodes_only.push_back(std::get<0>(entry));
  }
  for (auto& entry : right_merge_nodes) {
    right_merge_nodes_only.push_back(std::get<0>(entry));
  }

  if (!left_merge_nodes_only.empty() && !right_merge_nodes_only.empty()) {
    if (CONSERVATIVE) {
      return;
    }
    if (max_left_path_length > max_right_path_length) {
      ReTreeRule::merge(node, left_merge_nodes_only, true);
    } else {
      ReTreeRule::merge(node, right_merge_nodes_only, false);
    }
    return;
  }

  if (!left_merge_nodes_only.empty()) {
    ReTreeRule::merge(node, left_merge_nodes_only, true);
    return;
  }

  ReTreeRule::merge(node, right_merge_nodes_only, false);
}

void DTNaiveMergeRule::dfs(std::shared_ptr<TreeNode> node) {
  if (node->mode == "LEAF") {
    return;
  }
  dfs(node->left);
  dfs(node->right);
  std::vector<std::tuple<std::shared_ptr<TreeNode>, int>> left_merge_nodes,
      right_merge_nodes;
  int left_merge_stats =
      ReTreeRule::find_merge_nodes(node->left, 1, node, true, left_merge_nodes);
  if (left_merge_stats == M_NO) {
    return;
  }

  int right_merge_stats = ReTreeRule::find_merge_nodes(
      node->right, 1, node, false, right_merge_nodes);
  if (right_merge_stats == M_NO) {
    return;
  }

  if (left_merge_stats != right_merge_stats) {
    return;
  }

  int max_left_path_length = 0, max_right_path_length = 0;
  for (auto& entry : left_merge_nodes) {
    max_left_path_length = std::max(max_left_path_length, std::get<1>(entry));
  }
  for (auto& entry : right_merge_nodes) {
    max_right_path_length = std::max(max_right_path_length, std::get<1>(entry));
  }

  std::vector<std::shared_ptr<TreeNode>> left_merge_nodes_only,
      right_merge_nodes_only;
  for (auto& entry : left_merge_nodes) {
    left_merge_nodes_only.push_back(std::get<0>(entry));
  }
  for (auto& entry : right_merge_nodes) {
    right_merge_nodes_only.push_back(std::get<0>(entry));
  }

  if (!left_merge_nodes_only.empty() && !right_merge_nodes_only.empty()) {
    return;
  }

  if (!left_merge_nodes_only.empty()) {
    if (node->left == left_merge_nodes_only[0]) {
      ReTreeRule::merge(node, left_merge_nodes_only, true);
      return;
    }
    return;
  }
  if (node->right == right_merge_nodes_only[0]) {
    ReTreeRule::merge(node, right_merge_nodes_only, false);
    return;
  }
  return;
}

std::string DTMergeRule::apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                               std::string& model_path, Node* treeNode,
                               int nthreads) {
  // std::ofstream outputfile(
  //     "/volumn/Retree_exp/DTMergeRule.csv",
  //     std::ios::app);
  // auto start = std::chrono::high_resolution_clock::now();
  auto roots = ReTreeRule::model2trees(treeNode, nthreads);
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> model2trees_duration =
  //     end - start;

  // start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < roots.size(); i++) {
    dfs(roots[i]);
  }
  // end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> dfs_duration = end - start;

  // start = std::chrono::high_resolution_clock::now();
  auto regressor = TreeEnsembleRegressor::from_trees(roots);
  ReTreeRule::toTrees(treeNode, regressor);
  // end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> toTrees_duration = end - start;
  // outputfile << model_path << "," << model2trees_duration.count()
  //            << "," << dfs_duration.count() << ","
  //            << toTrees_duration.count() << "\n";
  return saveModelWithNewName(mp_in, graph, model_path, "merged");
}

std::string DTMergeRule::match(std::string& model_path, int nthreads) {
  std::string output_model_path = model_path;
  ModelProto mp_in;
  loadModel(&mp_in, model_path, true);
  std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

  bool found = false;
  Node* treeNode;
  graph->forEachNode([&found, &treeNode](Node* node) {
    if (node->hasAttribute(Symbol("target_treeids"))) {
      if (node->s(Symbol("post_transform")) == "NONE") {
        found = true;
        treeNode = node;
      }
    }
  });

  if (found)
    output_model_path = apply(mp_in, graph, model_path, treeNode, nthreads);

  return output_model_path;
}

std::string DTNaiveMergeRule::apply(ModelProto& mp_in,
                                    std::shared_ptr<Graph>& graph,
                                    std::string& model_path, Node* treeNode,
                                    int nthreads) {
  // std::ofstream outputfile(
  //     "/volumn/duckdb/examples/embedded-c++/workload/DTMergeRule_cost.txt",
  //     std::ios::app);
  // auto start = std::chrono::high_resolution_clock::now();
  auto roots = ReTreeRule::model2trees(treeNode, nthreads);
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> duration = end - start;
  // outputfile << "model2trees time cost (s): " << duration.count() / 1000
  //            << "\n";
  // start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < roots.size(); i++) {
    dfs(roots[i]);
  }
  // end = std::chrono::high_resolution_clock::now();
  // duration = end - start;
  // outputfile << "dfs time cost (s): " << duration.count() / 1000 << "\n";

  // start = std::chrono::high_resolution_clock::now();
  auto regressor = TreeEnsembleRegressor::from_trees(roots);
  ReTreeRule::toTrees(treeNode, regressor);
  // end = std::chrono::high_resolution_clock::now();
  // duration = end - start;
  // outputfile << "toTree time cost (s): " << duration.count() / 1000 <<
  // "\n"; outputfile.close();
  return saveModelWithNewName(mp_in, graph, model_path, "naive_merged");
}

std::string DTNaiveMergeRule::match(std::string& model_path, int nthreads) {
  std::string output_model_path = model_path;
  ModelProto mp_in;
  loadModel(&mp_in, model_path, true);
  std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

  bool found = false;
  Node* treeNode;
  graph->forEachNode([&found, &treeNode](Node* node) {
    if (node->hasAttribute(Symbol("target_treeids"))) {
      if (node->s(Symbol("post_transform")) == "NONE") {
        found = true;
        treeNode = node;
      }
    }
  });

  if (found)
    output_model_path = apply(mp_in, graph, model_path, treeNode, nthreads);

  return output_model_path;
}

}  // namespace onnx::optimization