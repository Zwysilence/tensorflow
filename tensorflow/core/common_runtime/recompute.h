#include <condition_variable>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <functional>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class TensorBuffer;
class Tensor;
class RecomputeHelper {
  ~RecomputeHelper() = default;
  RecomputeHelper(const RecomputeHelper&) = delete;
  RecomputeHelper(RecomputeHelper&&) = delete;
  RecomputeHelper& operator=(const RecomputeHelper&) = delete;
  RecomputeHelper& operator=(RecomputeHelper&&) = delete;
 public:
  static RecomputeHelper* GlobalRecomputeHelper() {
    static RecomputeHelper* helper = new RecomputeHelper;
    return helper;
  }
  typedef std::function<void()> RecomputeDoneCallback;
  typedef std::function<void(const std::string&, const std::vector<std::string>&, RecomputeDoneCallback)> RecomputeCall;
  void RecordTensorAccess(const std::string& tensor_name, const uint64 time_);
  void RecordTensorAccess(const std::string& tensor_name, const std::string& readable_name, const uint64 time_) {
    readable_names_[tensor_name] = readable_name;
    RecordTensorAccess(tensor_name, time_);
  }
  void RecordTensorBuffer(const std::string& tensor_name, Tensor* tensor);
  void RecordRecomputeCall(const std::string& tensor_name, RecomputeCall call);
  void RecomputeTensor(const std::string& tensor_name);
  void LoadRecomputePolicy();
  void DeleteMemory(const std::string& tensor_name);
  void IncrementUsingCount(const std::string& tensor_name);
  void DecrementUsingCount(const std::string& tensor_name);
  void SetRecomputing(const std::vector<std::string>& recompute_nodes);
  void SaveRecomputedTensor(const std::string& target, bool is_ref, const std::pair<std::string, Tensor*>& recomputed);
 private:
  void SetRecomputedTensors(const std::string& target);
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

  struct TriggerInfo {
    std::string tensor_name;
    int access_count;
    int total_access_count;
    int delete_trigger_count; // delete itself
    std::vector<std::vector<std::string>> recompute_tensors;
  };

  std::unordered_map<std::string, Params> tensor_recompute_params_;
  std::unordered_map<std::string, RecomputeCall> recompute_calls_;
  std::unordered_map<std::string, TriggerInfo> triggers_;
  std::unordered_map<std::string, std::string> readable_names_;
  std::unordered_map<std::string, std::vector<std::string>> node_to_tensors_;
  std::unordered_map<std::string, std::unordered_map<std::string, Tensor>> recomputed_tensors_;
  std::mutex mu_;
};
}
