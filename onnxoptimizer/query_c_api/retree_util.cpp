#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/model_util.h>

#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <mutex>
#include <queue>
#include <sstream>
#include <stack>
#include <thread>
#include <tuple>
#include <unordered_set>

#include "onnx/common/ir_pb_converter.cc"
#include "onnx/common/ir_pb_converter.h"
#include "retree.h"

namespace onnx::optimization {

float round(float x) {  // retain 6 decimal places
  return std::round(x * 1e6) / 1e6;
}

std::string getNewmodelname(std::string onnx_model_path, std::string suffix) {
  return onnx_model_path.substr(0, onnx_model_path.find(".onnx")) + "_" +
         suffix + ".onnx";
}

std::string saveModelWithNewName(ModelProto& mp_in,
                                 std::shared_ptr<Graph>& graph,
                                 std::string& model_path, std::string suffix) {
  ModelProto mp_out = PrepareOutput(mp_in);
  ExportModelProto(&mp_out, graph);
  checker::check_model(mp_out);
  auto new_model_path = getNewmodelname(model_path, suffix);
  saveModel(&mp_out, new_model_path);
  return new_model_path;
}

TreeNode::TreeNode(int _id, int _feature_id, const std::string& _mode,
                   double _value, std::optional<int> _target_id,
                   std::optional<double> _target_weight, int _samples)
    : id(_id),
      feature_id(_feature_id),
      mode(_mode),
      value(_value),
      target_id(_target_id),
      target_weight(_target_weight),
      samples(_samples),
      left(nullptr),
      right(nullptr) {}

TreeEnsembleRegressor::TreeEnsembleRegressor()
    : n_targets(1), post_transform("NONE") {};

TreeEnsembleRegressor TreeEnsembleRegressor::from_trees(
    const std::vector<std::shared_ptr<TreeNode>>& roots) {
  TreeEnsembleRegressor regressor;
  std::vector<TreeEnsembleRegressor> regressors;
  for (size_t i = 0; i < roots.size(); i++) {
    regressors.push_back(from_tree(roots[i], i));
  }
  for (auto& r : regressors) {
    regressor.nodes_falsenodeids.insert(regressor.nodes_falsenodeids.end(),
                                        r.nodes_falsenodeids.begin(),
                                        r.nodes_falsenodeids.end());
    regressor.nodes_featureids.insert(regressor.nodes_featureids.end(),
                                      r.nodes_featureids.begin(),
                                      r.nodes_featureids.end());
    regressor.nodes_hitrates.insert(regressor.nodes_hitrates.end(),
                                    r.nodes_hitrates.begin(),
                                    r.nodes_hitrates.end());
    regressor.nodes_missing_value_tracks_true.insert(
        regressor.nodes_missing_value_tracks_true.end(),
        r.nodes_missing_value_tracks_true.begin(),
        r.nodes_missing_value_tracks_true.end());
    regressor.nodes_modes.insert(regressor.nodes_modes.end(),
                                 r.nodes_modes.begin(), r.nodes_modes.end());
    regressor.nodes_nodeids.insert(regressor.nodes_nodeids.end(),
                                   r.nodes_nodeids.begin(),
                                   r.nodes_nodeids.end());
    regressor.nodes_treeids.insert(regressor.nodes_treeids.end(),
                                   r.nodes_treeids.begin(),
                                   r.nodes_treeids.end());
    regressor.nodes_truenodeids.insert(regressor.nodes_truenodeids.end(),
                                       r.nodes_truenodeids.begin(),
                                       r.nodes_truenodeids.end());
    regressor.nodes_values.insert(regressor.nodes_values.end(),
                                  r.nodes_values.begin(), r.nodes_values.end());
    regressor.target_ids.insert(regressor.target_ids.end(),
                                r.target_ids.begin(), r.target_ids.end());
    regressor.target_nodeids.insert(regressor.target_nodeids.end(),
                                    r.target_nodeids.begin(),
                                    r.target_nodeids.end());
    regressor.target_treeids.insert(regressor.target_treeids.end(),
                                    r.target_treeids.begin(),
                                    r.target_treeids.end());
    regressor.target_weights.insert(regressor.target_weights.end(),
                                    r.target_weights.begin(),
                                    r.target_weights.end());
  }
  return regressor;
}

TreeEnsembleRegressor TreeEnsembleRegressor::from_tree(
    std::shared_ptr<TreeNode> root, int tree_no) {
  TreeEnsembleRegressor regressor;
  from_tree_internal(regressor, root, tree_no);

  std::unordered_map<int, int> id_map;
  for (size_t i = 0; i < regressor.nodes_nodeids.size(); ++i) {
    int old_id = regressor.nodes_nodeids[i];
    id_map[old_id] = static_cast<int>(i);
  }

  std::vector<bool> is_leaf;
  for (const auto& mode : regressor.nodes_modes) {
    is_leaf.push_back(mode == "LEAF");
  }

  for (size_t i = 0; i < regressor.nodes_falsenodeids.size(); ++i) {
    if (is_leaf[i]) {
      regressor.nodes_falsenodeids[i] = 0;
    } else {
      regressor.nodes_falsenodeids[i] = id_map[regressor.nodes_falsenodeids[i]];
    }
  }
  for (size_t i = 0; i < regressor.nodes_truenodeids.size(); ++i) {
    if (is_leaf[i]) {
      regressor.nodes_truenodeids[i] = 0;
    } else {
      regressor.nodes_truenodeids[i] = id_map[regressor.nodes_truenodeids[i]];
    }
  }
  for (size_t i = 0; i < regressor.nodes_nodeids.size(); ++i) {
    regressor.nodes_nodeids[i] = id_map[regressor.nodes_nodeids[i]];
  }
  for (size_t i = 0; i < regressor.target_nodeids.size(); ++i) {
    regressor.target_nodeids[i] = id_map[regressor.target_nodeids[i]];
  }

  return regressor;
}

void TreeEnsembleRegressor::from_tree_internal(TreeEnsembleRegressor& regressor,
                                               std::shared_ptr<TreeNode> node,
                                               int tree_no) {
  bool is_leaf = node->mode == "LEAF";

  int falsenodeid = (!is_leaf && node->right) ? node->right->id : 0;
  int truenodeid = (!is_leaf && node->left) ? node->left->id : 0;

  regressor.nodes_falsenodeids.push_back(falsenodeid);
  regressor.nodes_featureids.push_back(node->feature_id);
  regressor.nodes_hitrates.push_back(static_cast<double>(node->samples));
  regressor.nodes_missing_value_tracks_true.push_back(0);
  regressor.nodes_modes.push_back(node->mode);
  regressor.nodes_nodeids.push_back(node->id);
  regressor.nodes_treeids.push_back(tree_no);
  regressor.nodes_truenodeids.push_back(truenodeid);
  regressor.nodes_values.push_back(node->value);

  if (is_leaf) {
    regressor.target_ids.push_back(0);
    regressor.target_nodeids.push_back(node->id);
    regressor.target_treeids.push_back(tree_no);
    regressor.target_weights.push_back(node->target_weight.value_or(0.0));
  } else {
    from_tree_internal(regressor, node->left, tree_no);
    from_tree_internal(regressor, node->right, tree_no);
  }
}

class ThreadPool {
 public:
  explicit ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty())
              return;
            task = std::move(tasks.front());
            tasks.pop();
          }
          task();
        }
      });
    }
  }

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");
      tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers)
      worker.join();
  }

 private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;

  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

}  // namespace onnx::optimization