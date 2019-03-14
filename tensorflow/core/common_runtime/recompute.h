#include <condition_variable>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <functional>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class TensorBuffer;
class Tensor;
class RecomputeHelper {
 public:
  static RecomputeHelper* GlobalRecomputeHelper() {
    static RecomputeHelper* helper = new RecomputeHelper;
    return helper;
  }
  typedef std::function<void()> RecomputeDoneCallback;
  typedef std::function<void(const std::string&, const std::vector<std::string>&, RecomputeDoneCallback)> RecomputeCall;
  void RecordTensorAccess(const std::string& tensor_name, const uint64 time_, RecomputeCall recompute); 
  void RecordTensorBuffer(const std::string& tensor_name, Tensor* tensor);
  void LoadRecomputePolicy();
  void DeleteMemory(const std::string& tensor_name);
  void IncrementUsingCount(const std::string& tensor_name);
  void DecrementUsingCount(const std::string& tensor_name);
 private:
  RecomputeHelper() { LoadRecomputePolicy(); }
  typedef std::pair<std::shared_ptr<std::condition_variable>, std::shared_ptr<std::mutex>> condition_variable_and_mutex;
  enum DataStatus {
    IN,
    OUT,
    RECOMPUTING
  };
  struct Params {
    condition_variable_and_mutex cv_mu;
    volatile int data_ready;
    std::string target_tensor;
    std::vector<std::string> feed_tensors;
    TensorBuffer* buf;
    volatile int using_count;
    bool then_delete;
  };

  struct DeleteTriggerInfo {
    std::string tensor_name;
    int access_count;
    int delete_trigger_count;
    int total_access_count;
  };

  struct ComputeTriggerInfo {
    std::string tensor_name;
    int access_count;
    int compute_trigger_count;
    int total_access_count;
    Params* params;
  };

  std::unordered_map<std::string, Params> tensor_recompute_params_map_;
  std::unordered_map<std::string, DeleteTriggerInfo> delete_triggers_;
  std::unordered_map<std::string, ComputeTriggerInfo> compute_triggers_;
  std::mutex mu_;
};
}
