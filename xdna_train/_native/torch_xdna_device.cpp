#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/convolution_ops.h>
#include <ATen/ops/convolution_backward_ops.h>
#include <ATen/ops/mm_ops.h>
#include <ATen/ops/addmm_ops.h>
#include <ATen/ops/bmm_ops.h>
#include <ATen/ops/mul_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/sqrt_ops.h>
#include <ATen/ops/addcmul_ops.h>
#include <ATen/ops/addcdiv_ops.h>
#include <ATen/ops/lerp_ops.h>
#include <ATen/ops/sigmoid_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ops/add.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/leaky_relu_backward.h>
#include <ATen/ops/upsample_nearest2d.h>
#include <ATen/ops/upsample_nearest2d_backward.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_batch_norm_backward.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/extension.h>
#include "xrt_shim.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <array>
#include <filesystem>
#include <spawn.h>
#include <sys/wait.h>
#include <fstream>
#include <map>
#include <set>
#include <future>
#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>
#include <memory>
#include <string>

extern "C" void* shim_bo_map(ShimBo*);
extern "C" size_t shim_bo_size(ShimBo*);
extern "C" void xdna_conv_set_threads(int);
extern "C" void xdna_pack_conv3x3_halo_bf16(
    const uint16_t*, uint16_t*, int, int, int, int);
extern "C" void xdna_pack_conv3x3_halo_padded_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int);
extern "C" void xdna_pack_conv3x3_halo_padded_channels_last_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int);
extern "C" void xdna_yxbc_to_nchw_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int);
extern "C" void xdna_yxbc_to_channels_last_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int);
extern "C" void xdna_pack_conv3x3_halo_cat2_upsample2_channels_last_bf16(
    const uint16_t*, int, const uint16_t*, int, uint16_t*,
    int, int, int, int);
extern "C" void xdna_yxbc_slice_to_channels_last_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int, int, int);
extern "C" void xdna_pack_conv3x3_halo_padded_channels_last_stripe_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int, int, int);
extern "C" void xdna_yxbc_slice_stripe_to_channels_last_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int, int, int, int, int);
extern "C" void xdna_pack_conv3x3_weight_fwd_bf16(
    const uint16_t*, uint16_t*, int, int);
extern "C" void xdna_pack_conv3x3_weight_dx_slice_bf16(
    const uint16_t*, uint16_t*, int, int, int, int);
extern "C" void xdna_pack_conv3x3_weight_dx_slice_padded_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int);
extern "C" void xdna_yxbc_slice_stripe_to_channels_last_partial_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int, int, int, int, int, int);
extern "C" void xdna_pack_conv3x3_dy_channels_last_bf16(
    const uint16_t*, uint16_t*, int, int, int, int);
extern "C" void xdna_pack_conv3x3_dy_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int);
extern "C" void xdna_conv3x3w_pack_input_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int);
extern "C" void xdna_conv3x3w_pack_weights_bfp16(
    const uint16_t*, uint8_t*, int, int, int, int);
extern "C" void xdna_conv3x3w_pack_weights_dx_bfp16(
    const uint16_t*, uint8_t*, int, int, int, int);
extern "C" void xdna_conv3x3w_unpack_output_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int, int);
extern "C" ShimRun* shim_run_kernel_start(
    ShimKernel*, unsigned int, ShimBo*, size_t, ShimBo* const*, size_t);

namespace {

constexpr c10::DeviceType kXdnaType = c10::DeviceType::PrivateUse1;
constexpr size_t kPageAlignment = 4096;
constexpr uint64_t kXdnaAllocationMagic = 0x58444e41424f3031ULL; // "XDNABO01"
thread_local c10::DeviceIndex g_current_device = 0;

// Intra-op width for host (CPU) compute issued from XDNA kernels.
//
// Autograd runs backward for non-CPU devices on its own worker thread, and the
// CPU dW overlap uses std::async/queue threads.  Fresh threads take their
// OpenMP width from ATen's process-wide value, which other code lowers:
// DataLoader(pin_memory=True) calls torch.set_num_threads(1) in its pin
// thread.  Every XDNA backward op then ran single-threaded (4x slower NIS
// step).  Follow the width the user sets on the thread that loaded the
// backend (the Python main thread) and apply it to every other thread
// before host compute.  XDNA_HOST_THREADS=N pins the width instead.
std::atomic<int> g_host_threads{0};
std::thread::id g_host_threads_owner;
bool g_host_threads_fixed = false;

void init_host_threads() {
  g_host_threads_owner = std::this_thread::get_id();
  if (const char* raw = std::getenv("XDNA_HOST_THREADS")) {
    const int parsed = std::atoi(raw);
    if (parsed > 0) {
      g_host_threads.store(parsed, std::memory_order_relaxed);
      g_host_threads_fixed = true;
      return;
    }
  }
  g_host_threads.store(at::get_num_threads(), std::memory_order_relaxed);
}

void sync_host_threads() {
  if (std::this_thread::get_id() == g_host_threads_owner) {
    if (!g_host_threads_fixed) {
      g_host_threads.store(at::get_num_threads(), std::memory_order_relaxed);
    }
    return;
  }
  const int want = g_host_threads.load(std::memory_order_relaxed);
  if (want > 0 && at::get_num_threads() != want) {
    at::set_num_threads(want);
  }
}

ShimDevice* xdna_device();

enum class XdnaCoherency : uint8_t {
  Clean = 0,
  HostDirty = 1,
  DeviceDirty = 2,
};

struct CachedBo {
  ShimBo* bo = nullptr;
  void* mapped = nullptr;
  size_t bytes = 0;
};

struct ArenaBacking {
  ShimBo* bo = nullptr;
  uint8_t* mapped = nullptr;
  size_t bytes = 0;
  size_t used = 0;
};

int tensor_bo_flag() {
  static const int flag = []() {
    const char* raw = std::getenv("XDNA_TENSOR_BO_FLAG");
    if (raw == nullptr || raw[0] == 0 || std::strcmp(raw, "host_only") == 0)
      return 2;
    if (std::strcmp(raw, "cacheable") == 0)
      return 1;
    if (std::strcmp(raw, "normal") == 0)
      return 0;
    TORCH_CHECK(
        false,
        "XDNA_TENSOR_BO_FLAG must be host_only, cacheable, or normal; got ",
        raw);
    return 2;
  }();
  return flag;
}

struct XdnaBoCache {
  std::mutex mutex;
  std::unordered_map<size_t, std::vector<CachedBo>> free;
  size_t cached_bytes = 0;
  size_t max_cached_bytes = size_t(512) << 20;
  uint64_t fresh_allocations = 0;
  uint64_t reuses = 0;
  size_t arena_chunk_bytes = 0;
  std::vector<ArenaBacking> arenas;
  uint64_t arena_parent_allocations = 0;
  uint64_t arena_subbuffer_allocations = 0;
  size_t arena_reserved_bytes = 0;

  XdnaBoCache() {
    if (const char* raw = std::getenv("XDNA_BO_CACHE_MB")) {
      char* end = nullptr;
      const unsigned long long mb = std::strtoull(raw, &end, 10);
      if (end != raw && mb > 0) {
        max_cached_bytes = static_cast<size_t>(mb) << 20;
      }
    }
    if (const char* raw = std::getenv("XDNA_BO_ARENA_MB")) {
      char* end = nullptr;
      const unsigned long long mb = std::strtoull(raw, &end, 10);
      if (end != raw && mb > 0) {
        arena_chunk_bytes = static_cast<size_t>(mb) << 20;
      }
    }
  }

  CachedBo acquire(size_t bytes) {
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto it = free.find(bytes);
      if (it != free.end() && !it->second.empty()) {
        CachedBo out = it->second.back();
        it->second.pop_back();
        cached_bytes -= out.bytes;
        ++reuses;
        return out;
      }
    }

    if (arena_chunk_bytes != 0 && bytes <= arena_chunk_bytes) {
      std::lock_guard<std::mutex> guard(mutex);
      ArenaBacking* backing = nullptr;
      size_t offset = 0;
      for (auto& candidate : arenas) {
        const size_t aligned =
            (candidate.used + kPageAlignment - 1) & ~(kPageAlignment - 1);
        if (aligned <= candidate.bytes &&
            bytes <= candidate.bytes - aligned) {
          backing = &candidate;
          offset = aligned;
          break;
        }
      }
      if (backing == nullptr) {
        const size_t parent_bytes = std::max(arena_chunk_bytes, bytes);
        ShimBo* parent = shim_bo_alloc(
            xdna_device(), nullptr, parent_bytes, tensor_bo_flag(), 0);
        TORCH_CHECK(
            parent != nullptr,
            "XDNA arena parent allocation failed for ",
            parent_bytes,
            " bytes: ",
            shim_last_error());
        void* parent_map = shim_bo_map(parent);
        if (parent_map == nullptr) {
          shim_bo_free(parent);
          TORCH_CHECK(false, "XDNA arena parent map failed: ", shim_last_error());
        }
        arenas.push_back(ArenaBacking{
            parent,
            static_cast<uint8_t*>(parent_map),
            parent_bytes,
            0,
        });
        backing = &arenas.back();
        offset = 0;
        ++arena_parent_allocations;
        arena_reserved_bytes += parent_bytes;
      }

      ShimBo* child = shim_bo_subbuffer(backing->bo, bytes, offset);
      TORCH_CHECK(
          child != nullptr,
          "XDNA arena sub-buffer allocation failed for ",
          bytes,
          " bytes at offset ",
          offset,
          ": ",
          shim_last_error());
      backing->used = offset + bytes;
      ++fresh_allocations;
      ++arena_subbuffer_allocations;
      return CachedBo{child, backing->mapped + offset, bytes};
    }

    ShimBo* bo =
        shim_bo_alloc(xdna_device(), nullptr, bytes, tensor_bo_flag(), 0);
    TORCH_CHECK(
        bo != nullptr,
        "XDNA XRT BO allocation failed for ",
        bytes,
        " bytes: ",
        shim_last_error());
    void* mapped = shim_bo_map(bo);
    if (mapped == nullptr) {
      shim_bo_free(bo);
      TORCH_CHECK(false, "XDNA XRT BO map failed: ", shim_last_error());
    }
    {
      std::lock_guard<std::mutex> guard(mutex);
      ++fresh_allocations;
    }
    return CachedBo{bo, mapped, bytes};
  }

  void release(CachedBo item) {
    if (item.bo == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex);
    if (arena_chunk_bytes != 0 ||
        cached_bytes + item.bytes <= max_cached_bytes) {
      cached_bytes += item.bytes;
      free[item.bytes].push_back(item);
      return;
    }
    shim_bo_free(item.bo);
  }

  void empty() {
    if (arena_chunk_bytes != 0) {
      return;
    }
    std::unordered_map<size_t, std::vector<CachedBo>> old;
    {
      std::lock_guard<std::mutex> guard(mutex);
      old.swap(free);
      cached_bytes = 0;
    }
    for (auto& [_, items] : old) {
      for (auto& item : items) {
        if (item.bo != nullptr) {
          shim_bo_free(item.bo);
        }
      }
    }
  }
};

XdnaBoCache& bo_cache() {
  // Intentionally process-lifetime. Avoid static-destruction order hazards
  // between PyTorch storage and XRT during interpreter shutdown.
  static XdnaBoCache* cache = new XdnaBoCache();
  return *cache;
}

enum class XdnaVirtualKind : uint8_t {
  CatChannels4D = 1,
  UpsampleNearest2D = 2,
};

struct XdnaVirtualValue {
  XdnaVirtualKind kind = XdnaVirtualKind::CatChannels4D;
  std::vector<at::Tensor> sources;
  int64_t dim = 0;
  std::array<int64_t, 2> output_size = {0, 0};
  std::optional<double> scales_h;
  std::optional<double> scales_w;
};

struct XdnaAllocation {
  uint64_t magic = kXdnaAllocationMagic;
  ShimBo* bo = nullptr;
  void* mapped = nullptr;
  size_t bytes = 0;
  XdnaCoherency coherency = XdnaCoherency::Clean;

  // Host work may fill this mapped BO after the producing ATen op returns.
  // Consumers fence on this future only when the storage is actually read.
  std::mutex pending_mutex;
  std::shared_future<void> pending_host_write;

  // Large producer values may remain logical until a consumer actually needs
  // their bytes. The BO remains ordinary XDNA storage, so unsupported
  // consumers can materialize into it without changing tensor semantics.
  std::mutex virtual_mutex;
  std::shared_ptr<XdnaVirtualValue> virtual_value;

  // conv3x3w: this tensor packed for the NPU, kept for its deferred dW
  // (Conv3x3wPacked).  The tag identifies the packed view and version.
  std::mutex packed_mutex;
  std::shared_ptr<void> packed;
  std::array<int64_t, 11> packed_tag{};
};

ShimDevice* xdna_device() {
  static ShimDevice* device = []() {
    ShimDevice* d = shim_device_open(0);
    TORCH_CHECK(
        d != nullptr,
        "failed to open XDNA device 0: ",
        shim_last_error());
    return d;
  }();
  return device;
}

XdnaAllocation* allocation_from_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.defined() && tensor.device().type() == kXdnaType,
      "expected an XDNA tensor");
  auto* ctx = tensor.storage().data_ptr().get_context();
  auto* allocation = static_cast<XdnaAllocation*>(ctx);
  TORCH_CHECK(
      allocation != nullptr && allocation->magic == kXdnaAllocationMagic &&
          allocation->bo != nullptr,
      "XDNA tensor is not backed by the XRT allocator");
  return allocation;
}

at::Tensor cpu_alias_unchecked(const at::Tensor& t) {
  auto options = at::TensorOptions()
                     .dtype(t.scalar_type())
                     .device(c10::DeviceType::CPU);
  return at::from_blob(
      const_cast<void*>(t.const_data_ptr()),
      t.sizes(),
      t.strides(),
      [](void*) {},
      options);
}

std::shared_ptr<XdnaVirtualValue> virtual_value_of(const at::Tensor& tensor) {
  if (!tensor.defined() || tensor.device().type() != kXdnaType)
    return nullptr;
  auto* allocation = allocation_from_tensor(tensor);
  std::lock_guard<std::mutex> guard(allocation->virtual_mutex);
  return allocation->virtual_value;
}

void set_virtual_value(
    const at::Tensor& tensor,
    std::shared_ptr<XdnaVirtualValue> value) {
  auto* allocation = allocation_from_tensor(tensor);
  std::lock_guard<std::mutex> guard(allocation->virtual_mutex);
  allocation->virtual_value = std::move(value);
  allocation->coherency = XdnaCoherency::Clean;
}

void clear_virtual_value(XdnaAllocation* allocation) {
  std::lock_guard<std::mutex> guard(allocation->virtual_mutex);
  allocation->virtual_value.reset();
}

void ensure_host_current(const at::Tensor& tensor);

void materialize_virtual(
    const at::Tensor& tensor,
    XdnaAllocation* allocation) {
  std::unique_lock<std::mutex> guard(allocation->virtual_mutex);
  auto value = allocation->virtual_value;
  if (!value)
    return;

  auto out_cpu = cpu_alias_unchecked(tensor);
  if (value->kind == XdnaVirtualKind::CatChannels4D) {
    int64_t offset = 0;
    for (const auto& src : value->sources) {
      ensure_host_current(src);
      auto src_cpu = cpu_alias_unchecked(src);
      const int64_t length = src.size(value->dim);
      out_cpu.narrow(value->dim, offset, length).copy_(src_cpu);
      offset += length;
    }
  } else if (value->kind == XdnaVirtualKind::UpsampleNearest2D) {
    TORCH_CHECK(
        value->sources.size() == 1,
        "virtual nearest upsample expects one source");
    const auto& src = value->sources[0];
    ensure_host_current(src);
    auto src_cpu = cpu_alias_unchecked(src);
    at::upsample_nearest2d_out(
        out_cpu,
        src_cpu,
        value->output_size,
        value->scales_h,
        value->scales_w);
  } else {
    TORCH_CHECK(false, "unknown XDNA virtual value kind");
  }

  allocation->virtual_value.reset();
  allocation->coherency = XdnaCoherency::HostDirty;
}

void wait_pending_host_write(XdnaAllocation* allocation) {
  std::shared_future<void> pending;
  {
    std::lock_guard<std::mutex> guard(allocation->pending_mutex);
    pending = allocation->pending_host_write;
  }
  if (pending.valid()) {
    pending.get();
  }
}

void set_pending_host_write(
    const at::Tensor& tensor,
    std::shared_future<void> pending) {
  auto* allocation = allocation_from_tensor(tensor);
  {
    std::lock_guard<std::mutex> guard(allocation->pending_mutex);
    TORCH_CHECK(
        !allocation->pending_host_write.valid(),
        "XDNA allocation already has pending host work");
    allocation->pending_host_write = std::move(pending);
  }
  // The producer is a host writer.  A host consumer needs only the completion
  // fence; a device consumer additionally performs the normal sync-to-device.
  allocation->coherency = XdnaCoherency::HostDirty;
}

class AsyncHostWorkQueue {
 public:
  AsyncHostWorkQueue()
      : max_outstanding_([]() {
          const char* raw = std::getenv("XDNA_DEFER_CPU_DW_WINDOW");
          if (raw != nullptr && raw[0] != 0) {
            const int parsed = std::atoi(raw);
            if (parsed > 0) return static_cast<size_t>(parsed);
          }
          return size_t(2);
        }()),
        worker_([this]() { run(); }) {}

  std::shared_future<void> submit(std::function<void()> fn) {
    std::packaged_task<void()> task(std::move(fn));
    auto future = task.get_future().share();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      space_cv_.wait(lock, [this]() {
        return outstanding_ < max_outstanding_;
      });
      ++outstanding_;
      tasks_.emplace_back(std::move(task));
    }
    work_cv_.notify_one();
    return future;
  }

  void drain() {
    submit([]() {}).get();
  }

 private:
  void run() {
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [this]() { return !tasks_.empty(); });
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      sync_host_threads();
      task();
      {
        std::lock_guard<std::mutex> guard(mutex_);
        --outstanding_;
      }
      space_cv_.notify_all();
    }
  }

  const size_t max_outstanding_;
  size_t outstanding_ = 0;
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable space_cv_;
  std::deque<std::packaged_task<void()>> tasks_;
  std::thread worker_;
};

AsyncHostWorkQueue& cpu_dw_queue() {
  // Process lifetime by design.  This avoids shutdown-order hazards between
  // queued ATen work, PyTorch globals and XRT-backed gradient storage.
  static auto* queue = new AsyncHostWorkQueue();
  return *queue;
}

void ensure_host_current(const at::Tensor& tensor) {
  if (!tensor.defined() || tensor.device().type() != kXdnaType) {
    return;
  }
  auto* allocation = allocation_from_tensor(tensor);
  wait_pending_host_write(allocation);
  materialize_virtual(tensor, allocation);
  if (allocation->coherency == XdnaCoherency::DeviceDirty) {
    TORCH_CHECK(
        shim_bo_sync_from_device(allocation->bo) == 0,
        "XDNA sync-from-device failed: ",
        shim_last_error());
    allocation->coherency = XdnaCoherency::Clean;
  }
}

void ensure_device_current(const at::Tensor& tensor) {
  if (!tensor.defined() || tensor.device().type() != kXdnaType) {
    return;
  }
  auto* allocation = allocation_from_tensor(tensor);
  wait_pending_host_write(allocation);
  materialize_virtual(tensor, allocation);
  if (allocation->coherency == XdnaCoherency::HostDirty) {
    TORCH_CHECK(
        shim_bo_sync_to_device(allocation->bo) == 0,
        "XDNA sync-to-device failed: ",
        shim_last_error());
    allocation->coherency = XdnaCoherency::Clean;
  }
}

void mark_host_dirty(const at::Tensor& tensor) {
  if (tensor.defined() && tensor.device().type() == kXdnaType) {
    auto* allocation = allocation_from_tensor(tensor);
    clear_virtual_value(allocation);
    allocation->coherency = XdnaCoherency::HostDirty;
  }
}

void mark_device_dirty(const at::Tensor& tensor) {
  if (tensor.defined() && tensor.device().type() == kXdnaType) {
    auto* allocation = allocation_from_tensor(tensor);
    clear_virtual_value(allocation);
    allocation->coherency = XdnaCoherency::DeviceDirty;
  }
}

struct XdnaAllocator final : c10::Allocator {
  static void deleter(void* ctx) {
    auto* allocation = static_cast<XdnaAllocation*>(ctx);
    if (allocation == nullptr) {
      return;
    }
    if (allocation->bo != nullptr) {
      wait_pending_host_write(allocation);
      bo_cache().release(CachedBo{
          allocation->bo,
          allocation->mapped,
          allocation->bytes,
      });
      allocation->bo = nullptr;
    }
    allocation->magic = 0;
    delete allocation;
  }

  c10::DataPtr allocate(size_t nbytes) override {
    const size_t requested = std::max<size_t>(nbytes, 1);
    const size_t actual =
        (requested + kPageAlignment - 1) & ~(kPageAlignment - 1);
    // Data tensors use generic host-visible XRT memory group 0. Physical
    // experiments confirmed that current whole-array kernels accept these BOs
    // directly even though xrt::kernel::group_id() carries a context tag in
    // its high bits.
    CachedBo cached = bo_cache().acquire(actual);
    auto* allocation = new XdnaAllocation();
    allocation->magic = kXdnaAllocationMagic;
    allocation->bo = cached.bo;
    allocation->mapped = cached.mapped;
    allocation->bytes = cached.bytes;
    allocation->coherency = XdnaCoherency::Clean;
    return c10::DataPtr(
        cached.mapped,
        allocation,
        &XdnaAllocator::deleter,
        c10::Device(kXdnaType, g_current_device));
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &XdnaAllocator::deleter;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    if (count != 0) {
      std::memcpy(dest, src, count);
    }
  }
};

XdnaAllocator g_xdna_allocator;

// Output adoption for host compute.
//
// Many XDNA ops run a CPU kernel on aliases of XRT memory.  CPU kernels that
// only exist functionally (convolution, convolution_backward, and anything in
// the boxed fallback) allocate their result with the CPU allocator, which used
// to cost a full copy into XDNA storage per output (~10% of a NIS step).
// Inside an AdoptHostAllocations scope, large CPU allocations are served from
// the XRT BO cache instead, so adopt_host_tensor() can hand the result over
// to an XDNA tensor without copying.  Outside a scope, and for small or
// unadopted tensors, behavior is the ordinary CPU allocator's.
thread_local int g_adopt_depth = 0;
constexpr size_t kAdoptMinBytes = size_t(64) << 10;

std::mutex& host_bo_mutex() {
  static auto* value = new std::mutex();
  return *value;
}

// data pointer -> XRT allocation, for CPU DataPtrs served from the BO cache.
std::unordered_map<void*, XdnaAllocation*>& host_bo_map() {
  static auto* value = new std::unordered_map<void*, XdnaAllocation*>();
  return *value;
}

struct XdnaHostAllocator final : c10::Allocator {
  c10::Allocator* base = nullptr;

  // Context is the data pointer itself, as raw_allocate() requires.
  static void deleter(void* data);

  c10::DataPtr allocate(size_t nbytes) override {
    if (g_adopt_depth > 0 && nbytes >= kAdoptMinBytes) {
      auto dp = g_xdna_allocator.allocate(nbytes);
      auto* allocation = static_cast<XdnaAllocation*>(dp.release_context());
      void* data = allocation->mapped;
      {
        std::lock_guard<std::mutex> guard(host_bo_mutex());
        host_bo_map()[data] = allocation;
      }
      return c10::DataPtr(
          data, data, &XdnaHostAllocator::deleter, c10::Device(c10::DeviceType::CPU));
    }
    return base->allocate(nbytes);
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &XdnaHostAllocator::deleter;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }
};

XdnaHostAllocator g_host_allocator;

void XdnaHostAllocator::deleter(void* data) {
  XdnaAllocation* allocation = nullptr;
  {
    std::lock_guard<std::mutex> guard(host_bo_mutex());
    auto& map = host_bo_map();
    auto it = map.find(data);
    if (it != map.end()) {
      allocation = it->second;
      map.erase(it);
    }
  }
  if (allocation != nullptr) {
    XdnaAllocator::deleter(allocation);
  } else {
    // A raw_allocate() served by the base allocator.
    g_host_allocator.base->raw_deleter()(data);
  }
}

void install_host_allocator() {
  g_host_allocator.base = c10::GetCPUAllocator();
  c10::SetCPUAllocator(&g_host_allocator, /*priority=*/1);
}

struct AdoptHostAllocations {
  AdoptHostAllocations() { ++g_adopt_depth; }
  ~AdoptHostAllocations() { --g_adopt_depth; }
  AdoptHostAllocations(const AdoptHostAllocations&) = delete;
  AdoptHostAllocations& operator=(const AdoptHostAllocations&) = delete;
};

// The XDNA tensor sharing `t`'s memory, when `t` is a CPU tensor whose sole
// storage came from the BO cache; std::nullopt otherwise (caller copies).
std::optional<at::Tensor> adopt_host_tensor(const at::Tensor& t) {
  if (!t.defined() || t.device().type() != c10::DeviceType::CPU) {
    return std::nullopt;
  }
  const c10::Storage& handle = t.unsafeGetTensorImpl()->storage();
  c10::StorageImpl* storage = handle.unsafeGetStorageImpl();
  if (storage->data_ptr().get_deleter() != &XdnaHostAllocator::deleter ||
      handle.use_count() != 1) {
    return std::nullopt;
  }
  void* data = storage->mutable_data();
  XdnaAllocation* allocation = nullptr;
  {
    std::lock_guard<std::mutex> guard(host_bo_mutex());
    auto& map = host_bo_map();
    auto it = map.find(data);
    if (it == map.end()) return std::nullopt;
    allocation = it->second;
    map.erase(it);
  }
  // The CPU tensor gives up its memory without running a deleter.
  auto old = storage->set_data_ptr(c10::DataPtr(nullptr, c10::Device(c10::DeviceType::CPU)));
  old.release_context();
  auto xdna_storage = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      storage->nbytes(),
      c10::DataPtr(data, allocation, &XdnaAllocator::deleter, c10::Device(kXdnaType, 0)),
      &g_xdna_allocator,
      /*resizable=*/true);
  storage->set_nbytes(0);
  auto impl = c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(std::move(xdna_storage)),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      t.dtype());
  impl->set_sizes_and_strides(t.sizes(), t.strides(), t.storage_offset());
  at::Tensor out(std::move(impl));
  allocation->coherency = XdnaCoherency::HostDirty;
  return out;
}

c10::intrusive_ptr<c10::StorageImpl> make_xdna_storage_impl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable) {
  c10::Allocator* alloc = allocator ? allocator : &g_xdna_allocator;
  if (data_ptr == nullptr) {
    return c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        std::move(size_bytes),
        alloc,
        resizable);
  }
  return c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      std::move(size_bytes),
      std::move(data_ptr),
      alloc,
      resizable);
}

struct XdnaGuardImpl final : c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override {
    return kXdnaType;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_CHECK(d.type() == kXdnaType, "expected XDNA/PrivateUse1 device");
    TORCH_CHECK(d.index() <= 0, "only xdna:0 is currently supported");
    auto old = getDevice();
    g_current_device = 0;
    return old;
  }

  c10::Device getDevice() const override {
    return c10::Device(kXdnaType, g_current_device);
  }

  void setDevice(c10::Device d) const override {
    TORCH_CHECK(d.type() == kXdnaType, "expected XDNA/PrivateUse1 device");
    TORCH_CHECK(d.index() <= 0, "only xdna:0 is currently supported");
    g_current_device = 0;
    // No sync_host_threads() here: DataLoader's pin thread sets the device
    // right after torch.set_num_threads(1), and widening it to the full
    // intra-op width added a third spinning OpenMP team (+25% NIS step).
    // Host compute entry points (ScopedMappedCpu, fallback, dW workers)
    // apply the width where it is actually used.
  }

  void uncheckedSetDevice(c10::Device) const noexcept override {
    g_current_device = 0;
  }

  c10::Stream getStream(c10::Device d) const override {
    return c10::Stream(
        c10::Stream::DEFAULT,
        c10::Device(kXdnaType, d.index() < 0 ? 0 : d.index()));
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    return getStream(d);
  }

  c10::Stream exchangeStream(c10::Stream s) const override {
    return getStream(s.device());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  bool queryStream(const c10::Stream&) const override {
    return true;
  }

  void synchronizeStream(const c10::Stream&) const override {
    cpu_dw_queue().drain();
  }
  void synchronizeDevice(const c10::DeviceIndex) const override {
    cpu_dw_queue().drain();
  }
};

void resize_xdna_storage(c10::StorageImpl* storage, size_t newsize) {
  TORCH_CHECK(storage != nullptr, "XDNA resize received null storage");
  const size_t oldsize = storage->nbytes();
  if (newsize <= oldsize) {
    storage->set_nbytes(newsize);
    return;
  }
  auto next = g_xdna_allocator.allocate(newsize);
  if (oldsize != 0 && storage->data() != nullptr) {
    std::memcpy(next.get(), storage->data(), oldsize);
  }
  storage->set_data_ptr(std::move(next));
  storage->set_nbytes(newsize);
  storage->set_allocator(&g_xdna_allocator);
  storage->set_resizable(true);
}

struct XdnaHooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override {
    return true;
  }

  bool isAvailable() const override {
    return true;
  }

  const at::Generator& getDefaultGenerator(c10::DeviceIndex) const override {
    return at::detail::getDefaultCPUGenerator();
  }

  at::Generator getNewGenerator(c10::DeviceIndex = -1) const override {
    return at::detail::createCPUGenerator();
  }

  at::Device getDeviceFromPtr(void*) const override {
    return c10::Device(kXdnaType, 0);
  }

  c10::Allocator* getPinnedMemoryAllocator() const override {
    return c10::GetAllocator(c10::DeviceType::CPU);
  }

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    return device_index == 0 || device_index == -1;
  }

  void resizePrivateUse1Bytes(
      const c10::Storage& storage, size_t newsize) const override {
    resize_xdna_storage(storage.unsafeGetStorageImpl(), newsize);
  }
};

XdnaHooks g_xdna_hooks;

at::Tensor empty_memory_format(
    c10::SymIntArrayRef size,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(
      at::layout_or_default(layout_opt) == at::Layout::Strided,
      "XDNA currently supports strided tensors only");
  TORCH_CHECK(
      !at::pinned_memory_or_default(pin_memory_opt),
      "pin_memory is not meaningful for XDNA tensors");
  auto device = at::device_or_default(device_opt);
  TORCH_CHECK(device.type() == kXdnaType, "expected XDNA device");
  TORCH_CHECK(device.index() <= 0, "only xdna:0 is currently supported");
  const auto dtype = at::dtype_or_default(dtype_opt);
  constexpr c10::DispatchKeySet ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic_symint(
      size, &g_xdna_allocator, ks, dtype, memory_format_opt);
}

at::Tensor empty_strided(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  TORCH_CHECK(
      at::layout_or_default(layout_opt) == at::Layout::Strided,
      "XDNA currently supports strided tensors only");
  TORCH_CHECK(
      !at::pinned_memory_or_default(pin_memory_opt),
      "pin_memory is not meaningful for XDNA tensors");
  auto device = at::device_or_default(device_opt);
  TORCH_CHECK(device.type() == kXdnaType, "expected XDNA device");
  TORCH_CHECK(device.index() <= 0, "only xdna:0 is currently supported");
  const auto dtype = at::dtype_or_default(dtype_opt);
  constexpr c10::DispatchKeySet ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_symint_generic(
      size, stride, &g_xdna_allocator, ks, dtype);
}

at::Tensor make_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::SymInt storage_offset) {
  auto impl = c10::make_intrusive<c10::TensorImpl>(
      c10::TensorImpl::VIEW,
      c10::Storage(self.storage()),
      self.key_set(),
      self.dtype());
  impl->set_sizes_and_strides(size, stride);
  impl->set_storage_offset(storage_offset.expect_int());
  return at::Tensor(std::move(impl));
}

at::Tensor as_strided_xdna(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  return make_alias(
      self,
      size,
      stride,
      storage_offset.value_or(self.sym_storage_offset()));
}

at::Tensor reshape_alias_xdna(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return make_alias(self, size, stride, self.sym_storage_offset());
}

at::Tensor view_xdna(
    const at::Tensor& self,
    c10::SymIntArrayRef requested_size) {
  auto size = at::infer_size_dv(requested_size, self.sym_numel());
  auto stride = at::detail::computeStride(
      self.sym_sizes(), self.sym_strides(), c10::SymIntArrayRef(size));
  TORCH_CHECK(
      stride.has_value(),
      "view size is not compatible with input tensor's size and stride "
      "(at least one dimension spans across two contiguous subspaces). "
      "Use .reshape(...) instead.");
  return make_alias(
      self,
      c10::SymIntArrayRef(size),
      c10::SymIntArrayRef(*stride),
      self.sym_storage_offset());
}

const at::Tensor& resize_xdna_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    std::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(
      !memory_format.has_value() ||
          *memory_format == at::MemoryFormat::Contiguous ||
          *memory_format == at::MemoryFormat::Preserve,
      "XDNA resize_ currently supports contiguous storage only");

  auto* impl = self.unsafeGetTensorImpl();
  impl->generic_set_sizes_contiguous(size);
  const auto nbytes_sym = at::detail::computeStorageNbytesContiguous(
      size,
      c10::SymInt(static_cast<int64_t>(self.element_size())),
      self.sym_storage_offset());
  const auto nbytes = static_cast<size_t>(
      nbytes_sym.guard_int(__FILE__, __LINE__));
  auto storage = self.storage();
  if (nbytes > storage.nbytes()) {
    resize_xdna_storage(storage.unsafeGetStorageImpl(), nbytes);
  }
  return self;
}

at::Tensor cpu_alias(const at::Tensor& t) {
  ensure_host_current(t);
  return cpu_alias_unchecked(t);
}

void host_storage_copy(
    const at::Tensor& src,
    const at::Tensor& dst) {
  ensure_host_current(src);
  TORCH_CHECK(
      src.sizes().equals(dst.sizes()),
      "XDNA copy requires identical shapes, got ",
      src.sizes(),
      " and ",
      dst.sizes());
  if (src.numel() == 0) {
    return;
  }
  if (src.scalar_type() == dst.scalar_type() &&
      src.is_contiguous() && dst.is_contiguous()) {
    std::memcpy(
        dst.mutable_data_ptr(),
        src.const_data_ptr(),
        static_cast<size_t>(src.numel()) * src.element_size());
    mark_host_dirty(dst);
    return;
  }

  // XDNA canonical storage is page-aligned host/SVM memory. Re-expose the
  // same bytes as non-owning CPU tensors so ATen handles dtype conversion and
  // arbitrary strides without staging allocations.
  auto src_cpu = cpu_alias(src);
  auto dst_cpu = cpu_alias(dst);
  dst_cpu.copy_(src_cpu, false);
  mark_host_dirty(dst);
}

at::Tensor copy_from_and_resize(
    const at::Tensor& src,
    const at::Tensor& dst) {
  if (dst.device().type() == kXdnaType) {
    resize_xdna_(dst, src.sym_sizes(), at::MemoryFormat::Contiguous);
  } else {
    dst.resize_(src.sizes());
  }
  host_storage_copy(src, dst);
  return dst;
}

at::Tensor copy_from(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool) {
  host_storage_copy(src, dst);
  return dst;
}

struct XdnaGemmStream {
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
  ShimBo* instr = nullptr;
  ShimBo* tmp = nullptr;
  ShimBo* trace = nullptr;
  size_t instr_words = 0;
};

struct XdnaGemmRuntime {
  ShimKernel* kernel = nullptr;
  std::array<XdnaGemmStream, 3> streams;
  std::mutex mutex;

  ~XdnaGemmRuntime() {
    for (auto& stream : streams) {
      if (stream.instr) shim_bo_free(stream.instr);
      if (stream.tmp) shim_bo_free(stream.tmp);
      if (stream.trace) shim_bo_free(stream.trace);
    }
    if (kernel) shim_kernel_close(kernel);
  }

  static std::vector<char> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    TORCH_CHECK(in.good(), "failed to open XDNA artifact ", path);
    const auto end = in.tellg();
    TORCH_CHECK(end >= 0, "failed to size XDNA artifact ", path);
    std::vector<char> data(static_cast<size_t>(end));
    in.seekg(0, std::ios::beg);
    if (!data.empty()) {
      in.read(data.data(), static_cast<std::streamsize>(data.size()));
      TORCH_CHECK(in.good(), "failed to read XDNA artifact ", path);
    }
    return data;
  }

  void open(
      const std::string& xclbin_path,
      const std::array<std::string, 3>& instr_paths) {
    TORCH_CHECK(kernel == nullptr, "XDNA GEMM runtime already configured");
    kernel = shim_kernel_load(
        xdna_device(), xclbin_path.c_str(), nullptr, QOS_PRIORITY_NONE);
    TORCH_CHECK(
        kernel != nullptr,
        "failed to load XDNA GEMM family: ",
        shim_last_error());

    const std::array<std::array<int64_t, 3>, 3> shapes = {{
        {{256, 768, 3072}},
        {{256, 3072, 768}},
        {{3072, 256, 768}},
    }};

    for (size_t i = 0; i < streams.size(); ++i) {
      auto blob = read_file(instr_paths[i]);
      TORCH_CHECK(
          blob.size() % 4 == 0,
          "XDNA GEMM instruction blob is not word aligned: ",
          instr_paths[i]);
      auto& stream = streams[i];
      stream.m = shapes[i][0];
      stream.k = shapes[i][1];
      stream.n = shapes[i][2];
      const int gid_instr = shim_kernel_group_id(kernel, 1);
      const int gid_tmp = shim_kernel_group_id(kernel, 6);
      const int gid_trace = shim_kernel_group_id(kernel, 7);
      TORCH_CHECK(
          gid_instr >= 0 && gid_tmp >= 0 && gid_trace >= 0,
          "failed to query XDNA GEMM memory groups: ",
          shim_last_error());
      stream.instr = shim_bo_alloc(
          xdna_device(), kernel, blob.size(), 1, gid_instr);
      stream.tmp = shim_bo_alloc(xdna_device(), kernel, 1, 2, gid_tmp);
      stream.trace = shim_bo_alloc(xdna_device(), kernel, 4, 2, gid_trace);
      TORCH_CHECK(
          stream.instr && stream.tmp && stream.trace,
          "failed to allocate XDNA GEMM control buffers: ",
          shim_last_error());
      TORCH_CHECK(
          shim_bo_write(stream.instr, blob.data(), blob.size(), 0) == 0,
          "failed to write XDNA GEMM instructions: ",
          shim_last_error());
      TORCH_CHECK(
          shim_bo_sync_to_device(stream.instr) == 0,
          "failed to sync XDNA GEMM instructions: ",
          shim_last_error());
      stream.instr_words = blob.size() / 4;
    }
  }

  XdnaGemmStream* find(int64_t m, int64_t k, int64_t n) {
    for (auto& stream : streams) {
      if (stream.m == m && stream.k == k && stream.n == n) {
        return &stream;
      }
    }
    return nullptr;
  }

  at::Tensor run(
      const at::Tensor& lhs,
      const at::Tensor& rhs,
      XdnaGemmStream& stream) {
    ensure_device_current(lhs);
    ensure_device_current(rhs);
    auto out = empty_memory_format(
        {c10::SymInt(stream.m), c10::SymInt(stream.n)},
        at::ScalarType::BFloat16,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::Contiguous);

    auto* a = allocation_from_tensor(lhs);
    auto* b = allocation_from_tensor(rhs);
    auto* c = allocation_from_tensor(out);
    {
      std::lock_guard<std::mutex> guard(mutex);
      TORCH_CHECK(
          shim_run_matmul8(
              kernel,
              3,
              stream.instr,
              stream.instr_words,
              a->bo,
              b->bo,
              c->bo,
              stream.tmp,
              stream.trace) == 0,
          "XDNA GEMM dispatch failed: ",
          shim_last_error());
    }
    mark_device_dirty(out);
    return out;
  }
};

std::unique_ptr<XdnaGemmRuntime>& gemm_runtime_slot() {
  static auto* slot = new std::unique_ptr<XdnaGemmRuntime>();
  return *slot;
}

struct XdnaConvStream {
  std::string name;
  int64_t cin = 0;
  int64_t h = 0;
  int64_t m_sched = 0;
  int64_t k_gemm = 0;
  ShimBo* instr = nullptr;
  ShimBo* a = nullptr;
  ShimBo* b = nullptr;
  ShimBo* tmp = nullptr;
  ShimBo* trace = nullptr;
  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  size_t instr_words = 0;
};

struct MappedCpuStat {
  uint64_t calls = 0;
  uint64_t total_ns = 0;
};

std::mutex& mapped_cpu_stats_mutex() {
  static auto* value = new std::mutex();
  return *value;
}

std::unordered_map<std::string, MappedCpuStat>& mapped_cpu_stats_map() {
  static auto* value =
      new std::unordered_map<std::string, MappedCpuStat>();
  return *value;
}

bool profile_mapped_cpu() {
  static const bool enabled = []() {
    const char* raw = std::getenv("XDNA_PROFILE_MAPPED_CPU");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

void record_mapped_cpu_ns(const char* name, uint64_t ns) {
  if (!profile_mapped_cpu()) return;
  std::lock_guard<std::mutex> guard(mapped_cpu_stats_mutex());
  auto& stat = mapped_cpu_stats_map()[name];
  ++stat.calls;
  stat.total_ns += ns;
}

class ScopedMappedCpu {
 public:
  explicit ScopedMappedCpu(const char* name)
      : name_(name), enabled_(profile_mapped_cpu()) {
    sync_host_threads();
    if (enabled_) start_ = std::chrono::steady_clock::now();
  }
  ~ScopedMappedCpu() {
    if (!enabled_) return;
    const auto end = std::chrono::steady_clock::now();
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start_).count());
    record_mapped_cpu_ns(name_, ns);
  }
 private:
  const char* name_;
  bool enabled_;
  std::chrono::steady_clock::time_point start_{};
};

struct XdnaConvRuntime {
  static constexpr int64_t kBatch = 40;
  static constexpr int64_t kCout = 128;
  static constexpr int64_t kCBlock = 32;

  ShimKernel* kernel = nullptr;
  std::array<XdnaConvStream, 2> forward_streams;
  std::array<XdnaConvStream, 2> dx_streams;
  std::mutex mutex;

  static void close_stream(XdnaConvStream& stream) {
    for (auto* bo : {stream.instr, stream.a, stream.b, stream.tmp, stream.trace}) {
      if (bo) shim_bo_free(bo);
    }
    stream = XdnaConvStream{};
  }

  ~XdnaConvRuntime() {
    for (auto& stream : forward_streams) close_stream(stream);
    for (auto& stream : dx_streams) close_stream(stream);
    if (kernel) shim_kernel_close(kernel);
  }

  static std::vector<char> read_file(const std::string& path) {
    return XdnaGemmRuntime::read_file(path);
  }

  static size_t a_elems(
      int64_t cin, int64_t h, int64_t m_sched, int64_t k_gemm) {
    const int64_t halo =
        (h + 2) * (h + 2) * (cin / kCBlock) * kBatch * kCBlock;
    const int64_t real_m = kBatch * h * h;
    return static_cast<size_t>(
        halo + std::max<int64_t>(0, m_sched - real_m) * k_gemm);
  }

  void open_stream(
      XdnaConvStream& stream,
      const std::string& name,
      const std::string& path,
      int64_t cin,
      int64_t h,
      int64_t m_sched,
      int64_t k_gemm) {
    auto blob = read_file(path);
    TORCH_CHECK(blob.size() % 4 == 0, "Conv stream is not word aligned: ", path);
    stream.name = name;
    stream.cin = cin;
    stream.h = h;
    stream.m_sched = m_sched;
    stream.k_gemm = k_gemm;
    const int gid_instr = shim_kernel_group_id(kernel, 1);
    const int gid_tmp = shim_kernel_group_id(kernel, 6);
    const int gid_trace = shim_kernel_group_id(kernel, 7);
    stream.instr = shim_bo_alloc(
        xdna_device(), kernel, blob.size(), 1, gid_instr);
    stream.a = shim_bo_alloc(
        xdna_device(), kernel, a_elems(cin, h, m_sched, k_gemm) * 2, 2, 0);
    stream.b = shim_bo_alloc(
        xdna_device(), kernel, static_cast<size_t>(k_gemm * kCout * 2), 2, 0);
    stream.tmp = shim_bo_alloc(xdna_device(), kernel, 1, 2, gid_tmp);
    stream.trace = shim_bo_alloc(xdna_device(), kernel, 4, 2, gid_trace);
    TORCH_CHECK(
        stream.instr && stream.a && stream.b && stream.tmp && stream.trace,
        "failed to allocate XDNA Conv buffers: ",
        shim_last_error());
    TORCH_CHECK(
        shim_bo_write(stream.instr, blob.data(), blob.size(), 0) == 0,
        "failed to write Conv instructions: ",
        shim_last_error());
    TORCH_CHECK(
        shim_bo_sync_to_device(stream.instr) == 0,
        "failed to sync Conv instructions: ",
        shim_last_error());
    stream.a_ptr = shim_bo_map(stream.a);
    stream.b_ptr = shim_bo_map(stream.b);
    TORCH_CHECK(
        stream.a_ptr && stream.b_ptr,
        "failed to map XDNA Conv staging buffers: ",
        shim_last_error());
    std::memset(
        stream.a_ptr, 0, a_elems(cin, h, m_sched, k_gemm) * 2);
    std::memset(
        stream.b_ptr, 0, static_cast<size_t>(k_gemm * kCout * 2));
    stream.instr_words = blob.size() / 4;
  }

  void open(
      const std::string& xclbin,
      const std::string& fwd18,
      const std::string& dx18,
      const std::string& fwd36,
      const std::string& dx36,
      int64_t m_sched18,
      int64_t m_sched36) {
    kernel = shim_kernel_load(
        xdna_device(), xclbin.c_str(), nullptr, QOS_PRIORITY_NONE);
    TORCH_CHECK(kernel != nullptr, "failed to load XDNA Conv family: ", shim_last_error());
    int pack_threads = 12;
    if (const char* raw = std::getenv("XDNA_CONV_PACK_THREADS")) {
      const int parsed = std::atoi(raw);
      if (parsed > 0) pack_threads = parsed;
    }
    xdna_conv_set_threads(pack_threads);
    // Scheduled GEMM rows come from the artifact manifest: they depend on
    // the tile height the streams were built with (padding to whole tiles).
    open_stream(
        forward_streams[0], "fwd18", fwd18, 256, 18, m_sched18, 2304);
    open_stream(
        dx_streams[0], "dx18", dx18, 128, 18, m_sched18, 1152);
    open_stream(
        forward_streams[1], "fwd36", fwd36, 256, 36, m_sched36, 2304);
    open_stream(
        dx_streams[1], "dx36", dx36, 128, 36, m_sched36, 1152);
  }

  XdnaConvStream* find_forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      c10::SymIntArrayRef stride,
      c10::SymIntArrayRef padding,
      c10::SymIntArrayRef dilation,
      bool transposed,
      c10::SymIntArrayRef output_padding,
      c10::SymInt groups,
      const std::optional<at::Tensor>& bias) {
    if (bias.has_value() && bias->defined()) return nullptr;
    if (input.scalar_type() != at::ScalarType::BFloat16 ||
        weight.scalar_type() != at::ScalarType::BFloat16 ||
        !input.is_contiguous() || !weight.is_contiguous() ||
        input.dim() != 4 || weight.dim() != 4 ||
        input.size(0) != kBatch || input.size(1) != 256 ||
        weight.size(0) != kCout || weight.size(1) != 256 ||
        weight.size(2) != 3 || weight.size(3) != 3 ||
        input.size(2) != input.size(3) ||
        stride.size() != 2 || stride[0] != 1 || stride[1] != 1 ||
        padding.size() != 2 || padding[0] != 1 || padding[1] != 1 ||
        dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1 ||
        transposed ||
        output_padding.size() != 2 ||
        output_padding[0] != 0 || output_padding[1] != 0 ||
        groups != 1) {
      return nullptr;
    }
    const int64_t h = input.size(2);
    for (auto& stream : forward_streams) {
      if (stream.h == h) return &stream;
    }
    return nullptr;
  }

  XdnaConvStream* find_dx(
      const at::Tensor& grad_output,
      const at::Tensor& weight,
      c10::SymIntArrayRef stride,
      c10::SymIntArrayRef padding,
      c10::SymIntArrayRef dilation,
      bool transposed,
      c10::SymIntArrayRef output_padding,
      c10::SymInt groups) {
    if (grad_output.scalar_type() != at::ScalarType::BFloat16 ||
        weight.scalar_type() != at::ScalarType::BFloat16 ||
        grad_output.dim() != 4 || weight.dim() != 4 ||
        grad_output.size(0) != kBatch || grad_output.size(1) != kCout ||
        weight.size(0) != kCout || weight.size(1) != 256 ||
        weight.size(2) != 3 || weight.size(3) != 3 ||
        grad_output.size(2) != grad_output.size(3) ||
        !weight.is_contiguous() ||
        stride.size() != 2 || stride[0] != 1 || stride[1] != 1 ||
        padding.size() != 2 || padding[0] != 1 || padding[1] != 1 ||
        dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1 ||
        transposed ||
        output_padding.size() != 2 ||
        output_padding[0] != 0 || output_padding[1] != 0 ||
        groups != 1) {
      return nullptr;
    }
    const int64_t h = grad_output.size(2);
    for (auto& stream : dx_streams) {
      if (stream.h == h) return &stream;
    }
    return nullptr;
  }

  at::Tensor input_grad(
      const at::Tensor& grad_output,
      const at::Tensor& weight,
      XdnaConvStream& stream) {
    ensure_host_current(grad_output);
    ensure_host_current(weight);
    auto go_cpu = cpu_alias(grad_output);
    if (!go_cpu.is_contiguous()) {
      go_cpu = go_cpu.contiguous();
    }
    auto weight_cpu = cpu_alias(weight);
    TORCH_CHECK(weight_cpu.is_contiguous(), "XDNA Conv weight must be contiguous");

    const int64_t h = stream.h;
    auto out = empty_memory_format(
        {
            c10::SymInt(kBatch),
            c10::SymInt(256),
            c10::SymInt(h),
            c10::SymInt(h),
        },
        at::ScalarType::BFloat16,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::Contiguous);
    auto out_cpu = cpu_alias(out);

    std::lock_guard<std::mutex> guard(mutex);
    xdna_pack_conv3x3_halo_bf16(
        static_cast<const uint16_t*>(go_cpu.const_data_ptr()),
        static_cast<uint16_t*>(stream.a_ptr),
        static_cast<int>(kBatch),
        static_cast<int>(kCout),
        static_cast<int>(h),
        static_cast<int>(h));
    TORCH_CHECK(
        shim_bo_sync_to_device(stream.a) == 0,
        "failed to sync XDNA Conv dX activation: ",
        shim_last_error());

    for (int slice = 0; slice < 2; ++slice) {
      const int start = slice * static_cast<int>(kCout);
      xdna_pack_conv3x3_weight_dx_slice_bf16(
          static_cast<const uint16_t*>(weight_cpu.const_data_ptr()),
          static_cast<uint16_t*>(stream.b_ptr),
          static_cast<int>(kCout),
          256,
          start,
          static_cast<int>(kCout));
      TORCH_CHECK(
          shim_bo_sync_to_device(stream.b) == 0,
          "failed to sync XDNA Conv dX weights: ",
          shim_last_error());

      auto raw = empty_memory_format(
          {c10::SymInt(stream.m_sched), c10::SymInt(kCout)},
          at::ScalarType::BFloat16,
          at::Layout::Strided,
          c10::Device(kXdnaType, 0),
          false,
          at::MemoryFormat::Contiguous);
      auto* raw_alloc = allocation_from_tensor(raw);
      ScopedMappedCpu run_phase("npu_conv.dx.run");
      TORCH_CHECK(
          shim_run_matmul8(
              kernel,
              3,
              stream.instr,
              stream.instr_words,
              stream.a,
              stream.b,
              raw_alloc->bo,
              stream.tmp,
              stream.trace) == 0,
          "XDNA Conv dX dispatch failed: ",
          shim_last_error());
      mark_device_dirty(raw);
      ensure_host_current(raw);

      xdna_yxbc_to_nchw_bf16(
          static_cast<const uint16_t*>(raw.const_data_ptr()),
          static_cast<uint16_t*>(out_cpu.data_ptr()),
          static_cast<int>(kBatch),
          static_cast<int>(kCout),
          static_cast<int>(h * h),
          256,
          start);
    }

    mark_host_dirty(out);
    return out;
  }

  at::Tensor forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      XdnaConvStream& stream) {
    ensure_host_current(input);
    ensure_host_current(weight);
    auto pack_phase = std::make_optional<ScopedMappedCpu>("npu_conv.fwd.pack+sync");
    xdna_pack_conv3x3_halo_bf16(
        static_cast<const uint16_t*>(input.const_data_ptr()),
        static_cast<uint16_t*>(stream.a_ptr),
        static_cast<int>(kBatch),
        static_cast<int>(stream.cin),
        static_cast<int>(stream.h),
        static_cast<int>(stream.h));
    xdna_pack_conv3x3_weight_fwd_bf16(
        static_cast<const uint16_t*>(weight.const_data_ptr()),
        static_cast<uint16_t*>(stream.b_ptr),
        static_cast<int>(kCout),
        static_cast<int>(stream.cin));
    TORCH_CHECK(
        shim_bo_sync_to_device(stream.a) == 0 &&
            shim_bo_sync_to_device(stream.b) == 0,
        "failed to sync XDNA Conv operands: ",
        shim_last_error());
    pack_phase.reset();

    auto raw = empty_memory_format(
        {c10::SymInt(stream.m_sched), c10::SymInt(kCout)},
        at::ScalarType::BFloat16,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::Contiguous);
    auto* out = allocation_from_tensor(raw);
    {
      ScopedMappedCpu run_phase("npu_conv.fwd.run");
      std::lock_guard<std::mutex> guard(mutex);
      TORCH_CHECK(
          shim_run_matmul8(
              kernel,
              3,
              stream.instr,
              stream.instr_words,
              stream.a,
              stream.b,
              out->bo,
              stream.tmp,
              stream.trace) == 0,
          "XDNA Conv dispatch failed: ",
          shim_last_error());
    }
    mark_device_dirty(raw);

    // Consumer-layout policy for small BF16 Conv family.
    // The raw [Y,X,B,Cout] result is zero-copy, but CPU BatchNorm is up to
    // 20x slower on those strides than on contiguous NCHW, and the blocked
    // transpose costs ~2 ms at H36.  Transpose by default;
    // XDNA_CONV_OUTPUT_NCHW=0 keeps the raw layout for NPU-to-NPU consumers.
    if (const char* setting = std::getenv("XDNA_CONV_OUTPUT_NCHW");
        setting == nullptr || std::strcmp(setting, "0") != 0) {
      ScopedMappedCpu reorder_phase("npu_conv.fwd.sync_back+reorder");
      ensure_host_current(raw);
      const int64_t hh = stream.h;
      auto reordered = empty_memory_format(
          {c10::SymInt(kBatch), c10::SymInt(kCout),
           c10::SymInt(hh), c10::SymInt(hh)},
          at::ScalarType::BFloat16,
          at::Layout::Strided,
          c10::Device(kXdnaType, 0),
          false,
          at::MemoryFormat::Contiguous);
      xdna_yxbc_to_nchw_bf16(
          static_cast<const uint16_t*>(raw.const_data_ptr()),
          static_cast<uint16_t*>(reordered.data_ptr()),
          static_cast<int>(kBatch),
          static_cast<int>(kCout),
          static_cast<int>(hh * hh),
          static_cast<int>(kCout),
          0);
      mark_host_dirty(reordered);
      return reordered;
    }

    // GEMM rows are physically [Y,X,Batch,Cout]. Present them to PyTorch as
    // logical NCHW through strides; no output transpose/copy is needed.
    const int64_t h = stream.h;
    std::vector<c10::SymInt> sizes = {
        c10::SymInt(kBatch), c10::SymInt(kCout),
        c10::SymInt(h), c10::SymInt(h)};
    std::vector<c10::SymInt> strides = {
        c10::SymInt(kCout),
        c10::SymInt(1),
        c10::SymInt(h * kBatch * kCout),
        c10::SymInt(kBatch * kCout)};
    return make_alias(raw, sizes, strides, c10::SymInt(0));
  }
};

std::unique_ptr<XdnaConvRuntime>& conv_runtime_slot() {
  static auto* slot = new std::unique_ptr<XdnaConvRuntime>();
  return *slot;
}

// conv3x3w: 3x3 / stride 1 / pad 1 convolution on the whole array for any
// batch, height, channel count and width <= 36 (xdna_train/_designs/conv3x3w.py).
// One xclbin; an instruction stream per (groups, padded width, channel chunks),
// compiled on first use by a background job and cached on disk.  Until a
// stream exists the conv takes the next path (older NPU families or CPU), so
// a new shape never blocks training.  Arithmetic: bf16 activations x bfp16
// weights with a bf16 running sum (~1.2% rel. RMS vs fp32; CPU bf16 ~0.2%).
// Compiles IRON instruction streams in the background: one niced build at a
// time, each key at most once per process.  `cmd` is the design script
// invocation; a build appends "--out <dir> <args>" and must write <dir>/<key>.bin.
struct IronStreamBuilder {
  std::string dir;
  std::string cmd;  // empty: only streams already on disk are used
  bool sync = false;

  std::mutex m;
  std::condition_variable cv;
  std::deque<std::pair<std::string, std::string>> queue;  // key, args
  std::set<std::string> seen;
  bool started = false;

  std::filesystem::path path(const std::string& key) const {
    return std::filesystem::path(dir) / (key + ".bin");
  }
  bool exists(const std::string& key) const { return std::filesystem::exists(path(key)); }

  // posix_spawn, not std::system: fork() would mark this process's memory
  // copy-on-write, and the first host write to a pinned XRT buffer after it
  // moves the page away from the NPU, which keeps reading the stale copy.
  // glibc's posix_spawn shares the address space (vfork semantics).
  void build(const std::string& args) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string line = cmd + " --out '" + dir + "' " + args + " >> '" + dir +
        "/build.log' 2>&1";
    std::string sh = "/bin/sh", flag = "-c";
    char* argv[] = {sh.data(), flag.data(), const_cast<char*>(line.c_str()), nullptr};
    pid_t pid = 0;
    if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, argv, environ) == 0) {
      int status = 0;
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
    }
  }

  // Make sure a build of `key` is under way (or done).
  void request(const std::string& key, const std::string& args) {
    if (cmd.empty()) return;
    std::unique_lock<std::mutex> lock(m);
    if (!seen.insert(key).second) return;
    if (sync) {
      lock.unlock();
      build(args);
      return;
    }
    queue.emplace_back(key, args);
    if (!started) {
      started = true;
      std::thread([this] {
        for (;;) {
          std::string args;
          {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] { return !queue.empty(); });
            args = queue.front().second;
            queue.pop_front();
          }
          build(args);
        }
      }).detach();
    }
    cv.notify_one();
  }

  // True if the stream exists; otherwise queues its build.
  bool ready(const std::string& key, const std::string& args) {
    if (exists(key)) return true;
    request(key, args);
    return exists(key);
  }
};

// Packed conv3x3w input [groups][chunks][2126][64] bf16 in an XRT BO.  Packing
// writes only real pixels and channels, so a buffer is zeroed when created and
// again whenever the packed (B, C, H, W) changes; released buffers go back to
// a pool by size.  Forward and dX hand theirs to the deferred dW, which needs
// exactly these layouts (no repacking).
struct Conv3x3wPacked {
  ShimBo* bo = nullptr;
  void* ptr = nullptr;
  size_t bytes = 0;
  std::array<int64_t, 5> shape{};  // B, C, H, W, chunks of the last pack
};

std::mutex& packed_pool_mutex() {
  static auto* m = new std::mutex();
  return *m;
}

std::multimap<size_t, Conv3x3wPacked*>& packed_pool() {
  static auto* pool = new std::multimap<size_t, Conv3x3wPacked*>();
  return *pool;
}

std::shared_ptr<Conv3x3wPacked> acquire_packed(size_t bytes) {
  Conv3x3wPacked* buf = nullptr;
  {
    std::lock_guard<std::mutex> lock(packed_pool_mutex());
    auto it = packed_pool().find(bytes);
    if (it != packed_pool().end()) {
      buf = it->second;
      packed_pool().erase(it);
    }
  }
  if (!buf) {
    buf = new Conv3x3wPacked();
    buf->bo = shim_bo_alloc(xdna_device(), nullptr, bytes, 2, 0);
    TORCH_CHECK(buf->bo, "failed to allocate conv3x3w input: ", shim_last_error());
    buf->ptr = shim_bo_map(buf->bo);
    TORCH_CHECK(buf->ptr, "failed to map conv3x3w input: ", shim_last_error());
    buf->bytes = bytes;
    std::memset(buf->ptr, 0, bytes);
  }
  return std::shared_ptr<Conv3x3wPacked>(buf, [](Conv3x3wPacked* b) {
    std::lock_guard<std::mutex> lock(packed_pool_mutex());
    packed_pool().emplace(b->bytes, b);
  });
}

std::mutex& out_pool_mutex() {
  static auto* m = new std::mutex();
  return *m;
}

std::multimap<size_t, Conv3x3wPacked*>& out_pool() {
  static auto* pool = new std::multimap<size_t, Conv3x3wPacked*>();
  return *pool;
}

// An NPU output buffer the CPU only ever reads.  Host memory is not coherent
// with NPU writes: a dirty cache line the CPU left in an output buffer (e.g. a
// recycled tensor) can be written back after the NPU wrote it, silently
// replacing its result.  So outputs never come from the tensor allocator, and
// a new buffer is flushed once.
std::shared_ptr<Conv3x3wPacked> acquire_npu_out(size_t bytes) {
  Conv3x3wPacked* buf = nullptr;
  {
    std::lock_guard<std::mutex> lock(out_pool_mutex());
    auto it = out_pool().find(bytes);
    if (it != out_pool().end()) {
      buf = it->second;
      out_pool().erase(it);
    }
  }
  if (!buf) {
    buf = new Conv3x3wPacked();
    buf->bo = shim_bo_alloc(xdna_device(), nullptr, bytes, 2, 0);
    TORCH_CHECK(buf->bo, "failed to allocate conv3x3w output: ", shim_last_error());
    buf->ptr = shim_bo_map(buf->bo);
    TORCH_CHECK(buf->ptr, "failed to map conv3x3w output: ", shim_last_error());
    buf->bytes = bytes;
    TORCH_CHECK(shim_bo_sync_to_device(buf->bo) == 0,
                "failed to sync conv3x3w output: ", shim_last_error());
  }
  return std::shared_ptr<Conv3x3wPacked>(buf, [](Conv3x3wPacked* b) {
    std::lock_guard<std::mutex> lock(out_pool_mutex());
    out_pool().emplace(b->bytes, b);
  });
}

// x (host NCHW bf16, contiguous) -> buf, then sync to the device.
void pack_conv3x3w_input(Conv3x3wPacked& buf, const at::Tensor& x_cpu, int64_t chunks) {
  const std::array<int64_t, 5> shape = {
      x_cpu.size(0), x_cpu.size(1), x_cpu.size(2), x_cpu.size(3), chunks};
  if (buf.shape != shape) {
    if (buf.shape != std::array<int64_t, 5>{}) std::memset(buf.ptr, 0, buf.bytes);
    buf.shape = shape;
  }
  xdna_conv3x3w_pack_input_bf16(
      static_cast<const uint16_t*>(x_cpu.const_data_ptr()),
      static_cast<uint16_t*>(buf.ptr),
      int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]), int(chunks));
  TORCH_CHECK(shim_bo_sync_to_device(buf.bo) == 0,
              "failed to sync conv3x3w input: ", shim_last_error());
}

std::array<int64_t, 11> packed_tag_of(const at::Tensor& t, int64_t chunks) {
  return {reinterpret_cast<int64_t>(t.const_data_ptr()),
          static_cast<int64_t>(t.unsafeGetTensorImpl()->version_counter().current_version()),
          t.size(0), t.size(1), t.size(2), t.size(3),
          t.stride(0), t.stride(1), t.stride(2), t.stride(3), chunks};
}

void attach_packed(const at::Tensor& t, int64_t chunks, std::shared_ptr<Conv3x3wPacked> buf) {
  auto* a = allocation_from_tensor(t);
  std::lock_guard<std::mutex> lock(a->packed_mutex);
  a->packed = std::move(buf);
  a->packed_tag = packed_tag_of(t, chunks);
}

// The packed copy attached to t (and detaches it), if it still matches t.
std::shared_ptr<Conv3x3wPacked> take_packed(const at::Tensor& t, int64_t chunks) {
  auto* a = allocation_from_tensor(t);
  std::lock_guard<std::mutex> lock(a->packed_mutex);
  std::shared_ptr<Conv3x3wPacked> buf;
  if (a->packed && a->packed_tag == packed_tag_of(t, chunks)) {
    buf = std::static_pointer_cast<Conv3x3wPacked>(a->packed);
  }
  a->packed.reset();
  return buf;
}

// Defined with the dW runtime: would a conv of this shape get an NPU dW?
bool conv3x3w_dw_wants(int64_t batch, int64_t cin, int64_t cout, int64_t h, int64_t w);

struct Conv3x3wPlan {
  int64_t batch = 0, cin = 0, cout = 0, h = 0, w = 0;
  int64_t groups = 0, wp = 0, chunks = 0;
  std::string key;
};

struct Conv3x3wWeightSlot {
  ShimBo* bo = nullptr;
  void* ptr = nullptr;
  bool valid = false;
  // Tag: storage pointer + TensorImpl version counter identify the contents;
  // the storage is pinned while tagged so its address cannot be reused.
  at::Storage pin;
  const void* data = nullptr;
  uint32_t version = 0;
  std::array<int64_t, 8> geom{};  // sizes, strides
  bool dx = false;
  int64_t nb = 0;
  uint64_t last_use = 0;
};

// Waits for (and frees) an in-flight async dispatch, also when unwinding.
struct Conv3x3wRun {
  ShimRun* r = nullptr;
  Conv3x3wRun() = default;
  Conv3x3wRun(const Conv3x3wRun&) = delete;
  ~Conv3x3wRun() {
    if (r) {
      shim_run_wait(r);
      shim_run_free(r);
    }
  }
  void wait() {
    ShimRun* h = r;
    r = nullptr;
    const int rc = shim_run_wait(h);
    shim_run_free(h);
    TORCH_CHECK(rc == 0, "conv3x3w dispatch failed: ", shim_last_error());
  }
};

struct Conv3x3wStream {
  ShimBo* instr = nullptr;
  size_t instr_words = 0;
  // bfp16 weight BOs, one per (weight tensor, fwd/dx, 128-channel block): the
  // stream is shared by every layer with the same shape key, so the tag says
  // which encoding a BO currently holds.
  std::vector<Conv3x3wWeightSlot> wslots;
  uint64_t use_clock = 0;
};

struct XdnaConv3x3wRuntime {
  static constexpr int64_t kN = 128;
  static constexpr int64_t kKStep = 64;
  static constexpr int64_t kGroupOutRows = 2048;
  static constexpr int64_t kGroupRows = 2126;
  static constexpr int64_t kMaxWidth = 36;
  static constexpr int64_t kMaxGroups = 1024;

  IronStreamBuilder builder;
  int64_t min_macs = 0;

  std::mutex mutex;  // kernel, streams, dispatch
  ShimKernel* kernel = nullptr;
  std::map<std::string, std::unique_ptr<Conv3x3wStream>> streams;

  static int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

  std::filesystem::path xclbin_path() const {
    return std::filesystem::path(builder.dir) / "conv3x3w.xclbin";
  }

  static double env_double(const char* name, double fallback) {
    if (const char* raw = std::getenv(name)) {
      const double v = std::atof(raw);
      if (v > 0.0) return v;
    }
    return fallback;
  }

  // The NPU computes the padded (H+2)x(W+2) grid with Cin padded to 64-channel
  // chunks and Cout to 128-channel blocks, plus host pack/unpack of the real
  // bf16 tensors, so skinny convs (RGB input) lose to the CPU.  Rates are per
  // millisecond: NPU_GMACS / CPU_GMACS in 1e9 MAC/ms, HOST_GBS in 1e9 B/s.
  // XDNA_CONV3X3W_FORCE=1 bypasses the model.
  static bool cost_model_prefers_npu(const Conv3x3wPlan& p, int64_t real_macs) {
    static const bool force = [] {
      const char* raw = std::getenv("XDNA_CONV3X3W_FORCE");
      return raw != nullptr && raw[0] != 0 && std::strcmp(raw, "0") != 0;
    }();
    if (force) return true;
    static const double overhead_ms = env_double("XDNA_CONV3X3W_NPU_OVERHEAD_MS", 0.15);
    static const double npu_gmacs = env_double("XDNA_CONV3X3W_NPU_GMACS", 5.5);
    static const double host_gbs = env_double("XDNA_CONV3X3W_HOST_GBS", 20.0);
    static const double cpu_gmacs = env_double("XDNA_CONV3X3W_CPU_GMACS", 0.9);
    const double padded_macs = double(p.batch) * (p.h + 2) * p.wp *
        double(p.chunks * kKStep) * double(ceil_div(p.cout, kN) * kN) * 9.0;
    const double io_bytes =
        2.0 * double(p.batch) * (double(p.cin) + double(p.cout)) * p.h * p.w;
    const double npu_ms =
        overhead_ms + padded_macs / (npu_gmacs * 1e9) + io_bytes / (host_gbs * 1e6);
    const double cpu_ms = double(real_macs) / (cpu_gmacs * 1e9);
    return npu_ms < cpu_ms;
  }

  // Shape check only (no stream lookup).  `x` is the conv input (or dY for
  // dX), `weight_cout`/`weight_cin` the effective output/input channels.
  std::optional<Conv3x3wPlan> plan(
      const at::Tensor& x,
      const at::Tensor& weight,
      bool for_dx,
      c10::SymIntArrayRef stride,
      c10::SymIntArrayRef padding,
      c10::SymIntArrayRef dilation,
      bool transposed,
      c10::SymIntArrayRef output_padding,
      const c10::SymInt& groups) const {
    if (x.scalar_type() != at::ScalarType::BFloat16 ||
        weight.scalar_type() != at::ScalarType::BFloat16 ||
        x.dim() != 4 || weight.dim() != 4 ||
        weight.size(2) != 3 || weight.size(3) != 3 ||
        stride.size() != 2 || stride[0] != 1 || stride[1] != 1 ||
        padding.size() != 2 || padding[0] != 1 || padding[1] != 1 ||
        dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1 ||
        transposed ||
        output_padding.size() != 2 ||
        output_padding[0] != 0 || output_padding[1] != 0 ||
        groups != 1) {
      return std::nullopt;
    }
    Conv3x3wPlan p;
    p.batch = x.size(0);
    p.cin = for_dx ? weight.size(0) : weight.size(1);
    p.cout = for_dx ? weight.size(1) : weight.size(0);
    p.h = x.size(2);
    p.w = x.size(3);
    if (x.size(1) != p.cin || p.batch < 1 || p.h < 1 || p.w < 1 ||
        p.w > kMaxWidth || p.cin < 1 || p.cout < 1) {
      return std::nullopt;
    }
    const int64_t real_macs = p.batch * p.h * p.w * p.cin * p.cout * 9;
    if (real_macs < min_macs) return std::nullopt;
    p.wp = p.w + 2;
    p.groups = ceil_div(p.batch * (p.h + 2) * p.wp, kGroupOutRows);
    p.chunks = ceil_div(p.cin, kKStep);
    if (!cost_model_prefers_npu(p, real_macs)) return std::nullopt;
    if (p.groups > kMaxGroups) return std::nullopt;
    p.key = "g" + std::to_string(p.groups) + "_w" + std::to_string(p.wp) +
        "_c" + std::to_string(p.chunks);
    return p;
  }

  // The stream for p if it is ready, else nullptr (and a build is queued).
  Conv3x3wStream* stream(const Conv3x3wPlan& p) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = streams.find(p.key);
      if (it != streams.end()) return it->second.get();
    }
    if (!builder.ready(p.key, "--groups " + std::to_string(p.groups) + " --wp " +
                                  std::to_string(p.wp) + " --chunks " + std::to_string(p.chunks))) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (!kernel) {
      kernel = shim_kernel_load(
          xdna_device(), xclbin_path().c_str(), nullptr, QOS_PRIORITY_NONE);
      TORCH_CHECK(kernel != nullptr, "failed to load conv3x3w xclbin: ", shim_last_error());
    }
    auto blob = XdnaGemmRuntime::read_file(builder.path(p.key).string());
    TORCH_CHECK(blob.size() % 4 == 0, "conv3x3w stream is not word aligned: ", p.key);
    auto s = std::make_unique<Conv3x3wStream>();
    s->instr = shim_bo_alloc(
        xdna_device(), kernel, blob.size(), 1, shim_kernel_group_id(kernel, 1));
    TORCH_CHECK(s->instr, "failed to allocate conv3x3w stream: ", shim_last_error());
    TORCH_CHECK(
        shim_bo_write(s->instr, blob.data(), blob.size(), 0) == 0 &&
            shim_bo_sync_to_device(s->instr) == 0,
        "failed to upload conv3x3w stream: ", shim_last_error());
    s->instr_words = blob.size() / 4;
    auto* raw = s.get();
    streams.emplace(p.key, std::move(s));
    return raw;
  }

  static bool weight_cache_enabled() {
    const char* e = std::getenv("XDNA_CONV3X3W_WEIGHT_CACHE");
    return !(e && e[0] == '0');
  }

  // The weight BO for (weight, dx, nb) in s.  Sets `hit` when it already holds
  // that encoding; otherwise the caller encodes into it.  Slots used by the
  // current call (last_use >= call_start) are never taken.
  Conv3x3wWeightSlot& weight_slot(
      Conv3x3wStream& s, const at::Tensor& weight, bool dx, int64_t nb,
      size_t b_bytes, uint64_t call_start, bool& hit) {
    constexpr size_t kMaxSlots = 256;
    const bool cache = weight_cache_enabled() && !weight.is_inference();
    std::array<int64_t, 8> geom{};
    for (int i = 0; i < 4; ++i) {
      geom[i] = weight.size(i);
      geom[4 + i] = weight.stride(i);
    }
    const void* data = weight.data_ptr();
    const uint32_t version = cache
        ? static_cast<uint32_t>(weight.unsafeGetTensorImpl()->version_counter().current_version())
        : 0;
    Conv3x3wWeightSlot* slot = nullptr;
    hit = false;
    if (cache) {
      for (auto& c : s.wslots) {
        if (c.valid && c.data == data && c.version == version && c.dx == dx &&
            c.nb == nb && c.geom == geom) {
          slot = &c;
          hit = true;
          break;
        }
      }
    }
    if (!slot) {
      for (auto& c : s.wslots) {
        if (!c.valid && c.last_use < call_start) { slot = &c; break; }
      }
    }
    if (!slot && s.wslots.size() < kMaxSlots) {
      auto* bo = shim_bo_alloc(xdna_device(), kernel, b_bytes, 2, 0);
      TORCH_CHECK(bo, "failed to allocate conv3x3w weights: ", shim_last_error());
      void* ptr = shim_bo_map(bo);
      TORCH_CHECK(ptr, "failed to map conv3x3w weights: ", shim_last_error());
      s.wslots.emplace_back();
      slot = &s.wslots.back();
      slot->bo = bo;
      slot->ptr = ptr;
    }
    if (!slot) {
      for (auto& c : s.wslots) {
        if (c.last_use < call_start && (!slot || c.last_use < slot->last_use)) slot = &c;
      }
    }
    TORCH_CHECK(slot, "conv3x3w: out of weight buffers");
    slot->last_use = ++s.use_clock;
    if (!hit) {
      slot->valid = false;
      slot->pin = at::Storage();
      if (cache) {
        slot->pin = weight.storage();
        slot->data = data;
        slot->version = version;
        slot->geom = geom;
        slot->dx = dx;
        slot->nb = nb;
      }
    }
    return *slot;
  }

  // out[:, :, :, :] = conv3x3(x, weight) (+ bias).  `weight` is the forward
  // weight [cout][cin][3][3]; for_dx convolves with its flipped/transposed form.
  at::Tensor run(
      const at::Tensor& x,
      const at::Tensor& weight,
      bool for_dx,
      const std::optional<at::Tensor>& bias,
      const Conv3x3wPlan& p,
      Conv3x3wStream& s,
      const char* phase,
      std::shared_ptr<Conv3x3wPacked>* keep_input = nullptr) {
    ensure_host_current(x);
    auto x_cpu = cpu_alias(x);
    if (!x_cpu.is_contiguous()) x_cpu = x_cpu.contiguous();
    auto out = empty_memory_format(
        {c10::SymInt(p.batch), c10::SymInt(p.cout), c10::SymInt(p.h), c10::SymInt(p.w)},
        at::ScalarType::BFloat16, at::Layout::Strided, c10::Device(kXdnaType, 0),
        false, at::MemoryFormat::Contiguous);
    auto out_cpu = cpu_alias(out);
    const size_t b_bytes = static_cast<size_t>(p.chunks * 9 * kKStep * kN / 8 * 9);
    const int64_t n_blocks = ceil_div(p.cout, kN);
    // Two raw outputs so block nb can be unpacked while nb+1 runs.
    std::array<std::shared_ptr<Conv3x3wPacked>, 2> raw;
    for (int64_t i = 0; i < std::min<int64_t>(n_blocks, 2); ++i) {
      raw[i] = acquire_npu_out(static_cast<size_t>(p.groups * kGroupOutRows * kN * 2));
    }

    auto xin = acquire_packed(static_cast<size_t>(p.groups * p.chunks * kGroupRows * kKStep * 2));
    xdna_conv_set_threads(std::max(1, g_host_threads.load(std::memory_order_relaxed)));
    {
      ScopedMappedCpu pack_phase("npu_conv3x3w.pack");
      pack_conv3x3w_input(*xin, x_cpu, p.chunks);
    }
    if (keep_input) *keep_input = xin;
    std::lock_guard<std::mutex> guard(mutex);
    const uint64_t call_start = s.use_clock + 1;
    std::optional<at::Tensor> w_cpu;
    auto prepare = [&](int64_t nb) -> Conv3x3wWeightSlot& {
      bool hit = false;
      auto& slot = weight_slot(s, weight, for_dx, nb, b_bytes, call_start, hit);
      if (hit) {
        ScopedMappedCpu hit_phase("npu_conv3x3w.weights_hit");
        return slot;
      }
      ScopedMappedCpu wpack_phase("npu_conv3x3w.weights");
      if (!w_cpu) {
        ensure_host_current(weight);
        auto w = cpu_alias(weight);
        w_cpu = w.is_contiguous() ? w : w.contiguous();
      }
      const auto* wp = static_cast<const uint16_t*>(w_cpu->const_data_ptr());
      if (for_dx) {
        xdna_conv3x3w_pack_weights_dx_bfp16(
            wp, static_cast<uint8_t*>(slot.ptr),
            int(p.cin), int(p.cout), int(p.chunks), int(nb));
      } else {
        xdna_conv3x3w_pack_weights_bfp16(
            wp, static_cast<uint8_t*>(slot.ptr),
            int(p.cout), int(p.cin), int(p.chunks), int(nb));
      }
      TORCH_CHECK(shim_bo_sync_to_device(slot.bo) == 0,
                  "failed to sync conv3x3w weights: ", shim_last_error());
      slot.valid = weight_cache_enabled() && static_cast<bool>(slot.pin);
      return slot;
    };
    auto start = [&](Conv3x3wWeightSlot& slot, const std::shared_ptr<Conv3x3wPacked>& r,
                     Conv3x3wRun& run) {
      ShimBo* data[3] = {xin->bo, slot.bo, r->bo};
      run.r = shim_run_kernel_start(kernel, 3, s.instr, s.instr_words, data, 3);
      TORCH_CHECK(run.r, "conv3x3w dispatch failed: ", shim_last_error());
    };

    Conv3x3wRun cur;
    Conv3x3wWeightSlot* slot = &prepare(0);
    // The run phase is the host wait for the NPU, i.e. what the overlapped
    // CPU work did not hide.
    start(*slot, raw[0], cur);
    for (int64_t nb = 0; nb < n_blocks; ++nb) {
      Conv3x3wWeightSlot* next = nullptr;
      if (nb + 1 < n_blocks) next = &prepare(nb + 1);
      {
        ScopedMappedCpu run_phase(phase);
        cur.wait();
      }
      if (next) start(*next, raw[(nb + 1) & 1], cur);
      auto& rw = raw[nb & 1];
      {
        ScopedMappedCpu sync_phase("npu_conv3x3w.unpack_sync");
        TORCH_CHECK(shim_bo_sync_from_device(rw->bo) == 0,
                    "failed to sync conv3x3w output: ", shim_last_error());
      }
      ScopedMappedCpu unpack_phase("npu_conv3x3w.unpack");
      xdna_conv3x3w_unpack_output_bf16(
          static_cast<const uint16_t*>(rw->ptr),
          static_cast<uint16_t*>(out_cpu.data_ptr()),
          int(p.batch), int(p.h), int(p.w),
          int(std::min<int64_t>(kN, p.cout - nb * kN)), int(p.cout), int(nb * kN));
    }
    if (bias.has_value() && bias->defined()) {
      ensure_host_current(*bias);
      out_cpu.add_(cpu_alias(*bias).view({1, -1, 1, 1}));
    }
    mark_host_dirty(out);
    return out;
  }

  std::optional<at::Tensor> try_forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      const std::optional<at::Tensor>& bias,
      c10::SymIntArrayRef stride,
      c10::SymIntArrayRef padding,
      c10::SymIntArrayRef dilation,
      bool transposed,
      c10::SymIntArrayRef output_padding,
      const c10::SymInt& groups) {
    auto p = plan(input, weight, false, stride, padding, dilation, transposed,
                  output_padding, groups);
    if (!p) return std::nullopt;
    if (bias.has_value() && bias->defined() &&
        (bias->dim() != 1 || bias->size(0) != p->cout)) {
      return std::nullopt;
    }
    auto* s = stream(*p);
    if (!s) return std::nullopt;
    const bool keep = at::GradMode::is_enabled() && weight.requires_grad() &&
        conv3x3w_dw_wants(p->batch, p->cin, p->cout, p->h, p->w);
    std::shared_ptr<Conv3x3wPacked> packed;
    auto out = run(input, weight, false, bias, *p, *s, "npu_conv3x3w.fwd.run",
                   keep ? &packed : nullptr);
    if (packed) attach_packed(input, p->chunks, std::move(packed));
    return out;
  }

  // dX of a 3x3 / pad 1 conv is the same conv of dY with the kernel
  // flipped spatially and in/out channels swapped.
  at::Tensor input_grad(
      const at::Tensor& grad_output,
      const at::Tensor& weight,
      const Conv3x3wPlan& p,
      Conv3x3wStream& s,
      std::shared_ptr<Conv3x3wPacked>* keep_dy = nullptr) {
    return run(grad_output, weight, true, std::nullopt, p, s, "npu_conv3x3w.dx.run", keep_dy);
  }
};

std::unique_ptr<XdnaConv3x3wRuntime>& conv3x3w_runtime_slot() {
  static auto* slot = new std::unique_ptr<XdnaConv3x3wRuntime>();
  return *slot;
}

// conv3x3w_dw: weight gradient of the same convolutions on the NPU
// (xdna_train/_designs/conv3x3w_dw.py): bfp16 operands in 8-pixel blocks with
// fp32 sums.  X and dY use the forward family's packed input layout.  A
// dispatch covers 128 input x 128 output channels; streams per
// (groups, padded width, chunks, cochunks, pass) are built on demand.
//
// It is its own xclbin, and a hardware-context switch costs ~2.6 ms, so dW is
// never interleaved with the dX chain: backward returns the gradient as
// pending host work, and the first reader (normally the optimizer) runs it.
// All deferred dWs of a step then run back to back in one context.
struct Conv3x3wDwPlan {
  int64_t batch = 0, cin = 0, cout = 0, h = 0, w = 0;
  int64_t groups = 0, wp = 0, chunks = 0, cochunks = 0;
  std::string buf_key;
  std::vector<std::pair<std::string, std::string>> passes;  // key, build args
};

struct XdnaConv3x3wDwRuntime {
  static constexpr int64_t kKStep = 64;
  static constexpr int64_t kGroupOutRows = 2048;
  static constexpr int64_t kGroupRows = 2126;
  static constexpr int64_t kMaxWidth = 36;
  static constexpr int64_t kMaxChunks = 15;  // shim stride limit (2^20 words)
  static constexpr int64_t kAcc = 3 * 64 * 32;
  static constexpr int64_t kCols = 6, kRows = 4;

  IronStreamBuilder builder;
  int64_t min_macs = 0;

  std::mutex mutex;
  ShimKernel* kernel = nullptr;
  std::map<std::string, std::pair<ShimBo*, size_t>> streams;  // instr, words
  ShimBo* c = nullptr;
  void* c_ptr = nullptr;

  static int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

  std::optional<Conv3x3wDwPlan> plan(
      const at::Tensor& input,
      const at::Tensor& grad_output,
      const at::Tensor& weight,
      c10::SymIntArrayRef stride,
      c10::SymIntArrayRef padding,
      c10::SymIntArrayRef dilation,
      bool transposed,
      c10::SymIntArrayRef output_padding,
      const c10::SymInt& groups) const {
    if (input.scalar_type() != at::ScalarType::BFloat16 ||
        grad_output.scalar_type() != at::ScalarType::BFloat16 ||
        weight.scalar_type() != at::ScalarType::BFloat16 ||
        input.dim() != 4 || grad_output.dim() != 4 || weight.dim() != 4 ||
        weight.size(2) != 3 || weight.size(3) != 3 ||
        stride.size() != 2 || stride[0] != 1 || stride[1] != 1 ||
        padding.size() != 2 || padding[0] != 1 || padding[1] != 1 ||
        dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1 ||
        transposed || output_padding.size() != 2 ||
        output_padding[0] != 0 || output_padding[1] != 0 || groups != 1) {
      return std::nullopt;
    }
    Conv3x3wDwPlan p;
    p.batch = input.size(0);
    p.cin = input.size(1);
    p.h = input.size(2);
    p.w = input.size(3);
    p.cout = grad_output.size(1);
    if (weight.size(0) != p.cout || weight.size(1) != p.cin ||
        grad_output.size(0) != p.batch || grad_output.size(2) != p.h ||
        grad_output.size(3) != p.w || p.w > kMaxWidth || p.batch < 1 || p.h < 1 ||
        p.w < 1) {
      return std::nullopt;
    }
    if (p.batch * p.h * p.w * p.cin * p.cout * 9 < min_macs) return std::nullopt;
    p.wp = p.w + 2;
    p.groups = ceil_div(p.batch * (p.h + 2) * p.wp, kGroupOutRows);
    p.chunks = ceil_div(p.cin, kKStep);
    p.cochunks = ceil_div(p.cout, kKStep);
    if (p.chunks > kMaxChunks || p.cochunks > kMaxChunks || p.groups > 1024) {
      return std::nullopt;
    }
    const std::string base = "g" + std::to_string(p.groups) + "_w" + std::to_string(p.wp) +
        "_c" + std::to_string(p.chunks) + "_d" + std::to_string(p.cochunks);
    p.buf_key = base;
    for (int64_t ci = 0; ci < ceil_div(p.chunks, 2); ++ci) {
      for (int64_t co = 0; co < ceil_div(p.cochunks, 2); ++co) {
        p.passes.emplace_back(
            base + "_p" + std::to_string(ci) + "_q" + std::to_string(co),
            "--groups " + std::to_string(p.groups) + " --wp " + std::to_string(p.wp) +
                " --chunks " + std::to_string(p.chunks) + " --cochunks " +
                std::to_string(p.cochunks) + " --ci-pass " + std::to_string(ci) +
                " --co-pass " + std::to_string(co));
      }
    }
    return p;
  }

  // True once every pass stream exists (missing ones are queued for build).
  bool ready(const Conv3x3wDwPlan& p) {
    bool all = true;
    for (const auto& [key, args] : p.passes) all &= builder.ready(key, args);
    return all;
  }

  ShimBo* alloc(size_t bytes, void** ptr) {
    auto* bo = shim_bo_alloc(xdna_device(), kernel, bytes, 2, 0);
    TORCH_CHECK(bo, "failed to allocate conv3x3w_dw buffer: ", shim_last_error());
    *ptr = shim_bo_map(bo);
    TORCH_CHECK(*ptr, "failed to map conv3x3w_dw buffer: ", shim_last_error());
    // Flush the zeros: dirty lines left in the CPU cache can be written back
    // after the NPU has written this buffer, overwriting its result.
    std::memset(*ptr, 0, bytes);
    TORCH_CHECK(shim_bo_sync_to_device(bo) == 0, "failed to sync conv3x3w_dw buffer: ",
                shim_last_error());
    return bo;
  }

  void load_kernel() {
    if (kernel) return;
    const auto xclbin = std::filesystem::path(builder.dir) / "conv3x3w_dw.xclbin";
    kernel = shim_kernel_load(xdna_device(), xclbin.c_str(), nullptr, QOS_PRIORITY_NONE);
    TORCH_CHECK(kernel != nullptr, "failed to load conv3x3w_dw xclbin: ", shim_last_error());
    c = alloc(static_cast<size_t>(kCols * kRows * kAcc * 4), &c_ptr);
  }

  // gw [cout][cin][3][3] bf16 (host memory) = dW of the conv.  x_buf / d_buf
  // are X and dY already packed by the forward and dX runs, else null.
  void compute(const at::Tensor& input, const at::Tensor& grad_output,
               uint16_t* gw, const Conv3x3wDwPlan& p,
               std::shared_ptr<Conv3x3wPacked> x_buf,
               std::shared_ptr<Conv3x3wPacked> d_buf) {
    {
      std::lock_guard<std::mutex> guard(mutex);
      load_kernel();
    }
    xdna_conv_set_threads(std::max(1, g_host_threads.load(std::memory_order_relaxed)));
    auto pack = [&](std::shared_ptr<Conv3x3wPacked>& buf, const at::Tensor& t, int64_t chunks) {
      if (buf) return;
      ScopedMappedCpu pack_phase("npu_conv3x3w.dw.pack");
      ensure_host_current(t);
      auto t_cpu = cpu_alias(t);
      if (!t_cpu.is_contiguous()) t_cpu = t_cpu.contiguous();
      buf = acquire_packed(static_cast<size_t>(p.groups * chunks * kGroupRows * kKStep * 2));
      pack_conv3x3w_input(*buf, t_cpu, chunks);
    };
    pack(x_buf, input, p.chunks);
    pack(d_buf, grad_output, p.cochunks);

    std::lock_guard<std::mutex> guard(mutex);
    const int64_t co_passes = ceil_div(p.cochunks, 2);
    for (size_t i = 0; i < p.passes.size(); ++i) {
      const auto& key = p.passes[i].first;
      auto it = streams.find(key);
      if (it == streams.end()) {
        auto blob = XdnaGemmRuntime::read_file(builder.path(key).string());
        auto* instr = shim_bo_alloc(xdna_device(), kernel, blob.size(), 1,
                                    shim_kernel_group_id(kernel, 1));
        TORCH_CHECK(instr && shim_bo_write(instr, blob.data(), blob.size(), 0) == 0 &&
                        shim_bo_sync_to_device(instr) == 0,
                    "failed to upload conv3x3w_dw stream: ", shim_last_error());
        it = streams.emplace(key, std::make_pair(instr, blob.size() / 4)).first;
      }
      {
        ScopedMappedCpu run_phase("npu_conv3x3w.dw.run");
        ShimBo* data[3] = {x_buf->bo, d_buf->bo, c};
        TORCH_CHECK(shim_run_kernel(kernel, 3, it->second.first, it->second.second, data, 3) == 0,
                    "conv3x3w_dw dispatch failed: ", shim_last_error());
        TORCH_CHECK(shim_bo_sync_from_device(c) == 0,
                    "failed to sync conv3x3w_dw output: ", shim_last_error());
      }
      // C [6 col][4 row][3 dx][8 ci blk][4 co blk][8 ci][8 co] fp32; column =
      // 3 * local input chunk + dy, row = 32-channel slice of the 128 outputs.
      ScopedMappedCpu unpack_phase("npu_conv3x3w.dw.unpack");
      const float* cf = static_cast<const float*>(c_ptr);
      const int64_t ci_pass = static_cast<int64_t>(i) / co_passes;
      const int64_t co_pass = static_cast<int64_t>(i) % co_passes;
      for (int64_t col = 0; col < kCols; ++col) {
        const int64_t ci0 = (2 * ci_pass + col / 3) * 64, dy = col % 3;
        if (ci0 >= p.cin) continue;
        for (int64_t row = 0; row < kRows; ++row) {
          const int64_t co0 = 2 * co_pass * 64 + row * 32;
          if (co0 >= p.cout) continue;
          const float* t = cf + (col * kRows + row) * kAcc;
          for (int64_t dx = 0; dx < 3; ++dx)
            for (int64_t cib = 0; cib < 8; ++cib)
              for (int64_t cob = 0; cob < 4; ++cob)
                for (int64_t a = 0; a < 8; ++a) {
                  const int64_t ci = ci0 + cib * 8 + a;
                  if (ci >= p.cin) continue;
                  const float* src = t + (((dx * 8 + cib) * 4 + cob) * 8 + a) * 8;
                  for (int64_t e = 0; e < 8; ++e) {
                    const int64_t co = co0 + cob * 8 + e;
                    if (co >= p.cout) continue;
                    gw[((co * p.cin + ci) * 3 + dy) * 3 + dx] =
                        c10::BFloat16(src[e]).x;
                  }
                }
        }
      }
    }
  }
};

std::unique_ptr<XdnaConv3x3wDwRuntime>& conv3x3w_dw_runtime_slot() {
  static auto* slot = new std::unique_ptr<XdnaConv3x3wDwRuntime>();
  return *slot;
}

// A deferred NPU dW: inputs (and their packed copies, when the forward / dX
// runs left them) held until the gradient's first reader runs it.
struct Conv3x3wDwJob {
  at::Tensor input, grad_output;
  std::shared_ptr<Conv3x3wPacked> x_buf, d_buf;
};

bool conv3x3w_dw_wants(int64_t batch, int64_t cin, int64_t cout, int64_t h, int64_t w) {
  auto& rt = conv3x3w_dw_runtime_slot();
  return rt && w <= XdnaConv3x3wDwRuntime::kMaxWidth &&
      XdnaConv3x3wDwRuntime::ceil_div(cin, 64) <= XdnaConv3x3wDwRuntime::kMaxChunks &&
      XdnaConv3x3wDwRuntime::ceil_div(cout, 64) <= XdnaConv3x3wDwRuntime::kMaxChunks &&
      batch * h * w * cin * cout * 9 >= rt->min_macs;
}

struct XdnaFullFrameConvStream {
  std::string name;
  int64_t cin = 0;
  int64_t h = 0;
  int64_t w = 0;
  int64_t w_sched = 0;
  int64_t m_sched = 0;
  int64_t k_gemm = 0;
  int64_t n_phys = 128;
  ShimBo* instr = nullptr;
  ShimBo* a = nullptr;
  ShimBo* b = nullptr;
  ShimBo* tmp = nullptr;
  ShimBo* trace = nullptr;
  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  size_t instr_words = 0;
};

struct XdnaFullFrameConvRuntime {
  static constexpr int64_t kBatch = 4;
  static constexpr int64_t kCout = 128;
  static constexpr int64_t kCBlock = 32;

  ShimKernel* kernel = nullptr;
  std::array<XdnaFullFrameConvStream, 3> forward_streams;
  std::array<XdnaFullFrameConvStream, 3> dx_streams;
  std::mutex mutex;

  ~XdnaFullFrameConvRuntime() {
    for (auto& stream : forward_streams) {
      for (auto* bo : {stream.instr, stream.a, stream.b, stream.tmp, stream.trace}) {
        if (bo) shim_bo_free(bo);
      }
    }
    for (auto& stream : dx_streams) {
      for (auto* bo : {stream.instr, stream.a, stream.b, stream.tmp, stream.trace}) {
        if (bo) shim_bo_free(bo);
      }
    }
    if (kernel) shim_kernel_close(kernel);
  }

  static std::vector<char> read_file(const std::string& path) {
    return XdnaGemmRuntime::read_file(path);
  }

  static size_t a_elems(
      int64_t cin, int64_t h, int64_t w_sched) {
    return static_cast<size_t>(
        (h + 2) * (w_sched + 2) * (cin / kCBlock) * kBatch * kCBlock);
  }

  void open_stream(
      XdnaFullFrameConvStream& stream,
      const std::string& name,
      const std::string& path,
      int64_t cin,
      int64_t h,
      int64_t w,
      int64_t w_sched,
      int64_t m_sched,
      int64_t k_gemm,
      int64_t n_phys = kCout) {
    auto blob = read_file(path);
    TORCH_CHECK(blob.size() % 4 == 0, "full-frame Conv stream is not word aligned: ", path);
    stream.name = name;
    stream.cin = cin;
    stream.h = h;
    stream.w = w;
    stream.w_sched = w_sched;
    stream.m_sched = m_sched;
    stream.k_gemm = k_gemm;
    stream.n_phys = n_phys;

    const int gid_instr = shim_kernel_group_id(kernel, 1);
    const int gid_tmp = shim_kernel_group_id(kernel, 6);
    const int gid_trace = shim_kernel_group_id(kernel, 7);
    stream.instr = shim_bo_alloc(
        xdna_device(), kernel, blob.size(), 1, gid_instr);
    stream.a = shim_bo_alloc(
        xdna_device(), kernel, a_elems(cin, h, w_sched) * 2, 2, 0);
    stream.b = shim_bo_alloc(
        xdna_device(), kernel, static_cast<size_t>(k_gemm * n_phys * 2), 2, 0);
    stream.tmp = shim_bo_alloc(xdna_device(), kernel, 1, 2, gid_tmp);
    stream.trace = shim_bo_alloc(xdna_device(), kernel, 4, 2, gid_trace);
    TORCH_CHECK(
        stream.instr && stream.a && stream.b && stream.tmp && stream.trace,
        "failed to allocate full-frame XDNA Conv buffers: ",
        shim_last_error());
    TORCH_CHECK(
        shim_bo_write(stream.instr, blob.data(), blob.size(), 0) == 0 &&
            shim_bo_sync_to_device(stream.instr) == 0,
        "failed to initialize full-frame Conv instructions: ",
        shim_last_error());
    stream.a_ptr = shim_bo_map(stream.a);
    stream.b_ptr = shim_bo_map(stream.b);
    TORCH_CHECK(
        stream.a_ptr && stream.b_ptr,
        "failed to map full-frame XDNA Conv staging buffers: ",
        shim_last_error());
    std::memset(stream.a_ptr, 0, a_elems(cin, h, w_sched) * 2);
    std::memset(
        stream.b_ptr, 0, static_cast<size_t>(k_gemm * n_phys * 2));
    stream.instr_words = blob.size() / 4;
  }

  void open(
      const std::string& xclbin,
      const std::string& res146,
      const std::string& fwd292,
      const std::string& fwd584,
      const std::string& dx292,
      const std::string& dx584) {
    kernel = shim_kernel_load(
        xdna_device(), xclbin.c_str(), nullptr, QOS_PRIORITY_NONE);
    TORCH_CHECK(
        kernel != nullptr,
        "failed to load full-frame XDNA Conv family: ",
        shim_last_error());
    int pack_threads = 12;
    if (const char* raw = std::getenv("XDNA_CONV_PACK_THREADS")) {
      const int parsed = std::atoi(raw);
      if (parsed > 0) pack_threads = parsed;
    }
    xdna_conv_set_threads(pack_threads);

    int64_t tile_m = 32;
    if (const char* raw = std::getenv("XDNA_FULLFRAME_TM")) {
      const int parsed = std::atoi(raw);
      if (parsed > 0 && parsed % kBatch == 0) tile_m = parsed;
    }
    const int64_t x_per_tile = tile_m / kBatch;
    const auto sched_width = [x_per_tile](int64_t w) {
      return ((w + x_per_tile - 1) / x_per_tile) * x_per_tile;
    };
    const auto sched_m = [&sched_width](int64_t h, int64_t w) {
      return kBatch * h * sched_width(w);
    };

    TORCH_CHECK(!res146.empty(), "full-frame residual stream is required");
    open_stream(
        forward_streams[0], "res146", res146,
        128, 146, 141, sched_width(141), sched_m(146, 141), 1152);
    if (!fwd292.empty()) {
      open_stream(
          forward_streams[1], "fwd292", fwd292,
          256, 292, 282, sched_width(282), sched_m(292, 282), 2304);
    }
    if (!fwd584.empty()) {
      open_stream(
          forward_streams[2], "fwd584", fwd584,
          256, 584, 564, sched_width(564), sched_m(584, 564), 2304);
    }
    if (!dx292.empty()) {
      open_stream(
          dx_streams[0], "dx292", dx292,
          128, 292, 282, sched_width(282), sched_m(292, 282), 1152);
    }
    if (!dx584.empty()) {
      constexpr int64_t kDx584StripeH = 73;
      open_stream(
          dx_streams[1], "dx584s73", dx584,
          128, kDx584StripeH, 564, sched_width(564),
          sched_m(kDx584StripeH, 564), 1152);
    }
  }

  void open_conv11_only(
      const std::string& xclbin,
      const std::string& insts,
      int64_t n_phys) {
    kernel = shim_kernel_load(
        xdna_device(), xclbin.c_str(), nullptr, QOS_PRIORITY_NONE);
    TORCH_CHECK(
        kernel != nullptr,
        "failed to load fast conv11 XDNA program: ",
        shim_last_error());
    int pack_threads = 12;
    if (const char* raw = std::getenv("XDNA_CONV_PACK_THREADS")) {
      const int parsed = std::atoi(raw);
      if (parsed > 0) pack_threads = parsed;
    }
    xdna_conv_set_threads(pack_threads);
    TORCH_CHECK(
        n_phys == 128 || n_phys == 256,
        "conv11 dX physical N must be 128 or 256");
    int64_t tile_m = 32;
    if (const char* raw = std::getenv("XDNA_FULLFRAME_TM")) {
      const int parsed = std::atoi(raw);
      if (parsed > 0 && parsed % kBatch == 0) tile_m = parsed;
    }
    const int64_t x_per_tile = tile_m / kBatch;
    const int64_t w_sched = ((564 + x_per_tile - 1) / x_per_tile) * x_per_tile;
    const int64_t m_sched = kBatch * 584 * w_sched;
    open_stream(
        dx_streams[2], n_phys == 256 ? "dx584c64n256" : "dx584c64", insts,
        64, 584, 564, w_sched, m_sched, 576, n_phys);
  }

  static bool common_args(
      const at::Tensor& input,
      const at::Tensor& weight,
      c10::SymIntArrayRef stride,
      c10::SymIntArrayRef padding,
      c10::SymIntArrayRef dilation,
      bool transposed,
      c10::SymIntArrayRef output_padding,
      c10::SymInt groups,
      const std::optional<at::Tensor>& bias) {
    return !(bias.has_value() && bias->defined()) &&
        input.scalar_type() == at::ScalarType::BFloat16 &&
        weight.scalar_type() == at::ScalarType::BFloat16 &&
        (input.is_contiguous() ||
         input.is_contiguous(at::MemoryFormat::ChannelsLast)) &&
        weight.is_contiguous() &&
        input.dim() == 4 && weight.dim() == 4 &&
        input.size(0) == kBatch &&
        weight.size(0) == kCout &&
        weight.size(2) == 3 && weight.size(3) == 3 &&
        stride.size() == 2 && stride[0] == 1 && stride[1] == 1 &&
        padding.size() == 2 && padding[0] == 1 && padding[1] == 1 &&
        dilation.size() == 2 && dilation[0] == 1 && dilation[1] == 1 &&
        !transposed &&
        output_padding.size() == 2 &&
        output_padding[0] == 0 && output_padding[1] == 0 &&
        groups == 1;
  }

  XdnaFullFrameConvStream* find_forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      c10::SymIntArrayRef stride,
      c10::SymIntArrayRef padding,
      c10::SymIntArrayRef dilation,
      bool transposed,
      c10::SymIntArrayRef output_padding,
      c10::SymInt groups,
      const std::optional<at::Tensor>& bias) {
    if (!common_args(
            input, weight, stride, padding, dilation, transposed,
            output_padding, groups, bias)) {
      return nullptr;
    }
    for (auto& stream : forward_streams) {
      if (input.size(1) == stream.cin &&
          weight.size(1) == stream.cin &&
          input.size(2) == stream.h &&
          input.size(3) == stream.w) {
        return &stream;
      }
    }
    return nullptr;
  }

  XdnaFullFrameConvStream* find_dx(
      const at::Tensor& grad_output,
      const at::Tensor& weight,
      c10::SymIntArrayRef stride,
      c10::SymIntArrayRef padding,
      c10::SymIntArrayRef dilation,
      bool transposed,
      c10::SymIntArrayRef output_padding,
      c10::SymInt groups) {
    if (grad_output.scalar_type() != at::ScalarType::BFloat16 ||
        weight.scalar_type() != at::ScalarType::BFloat16 ||
        grad_output.dim() != 4 || weight.dim() != 4 ||
        grad_output.size(0) != kBatch ||
        grad_output.size(1) != weight.size(0) ||
        (weight.size(1) != 128 && weight.size(1) != 195 &&
         weight.size(1) != 256) ||
        weight.size(2) != 3 || weight.size(3) != 3 ||
        !weight.is_contiguous() ||
        stride.size() != 2 || stride[0] != 1 || stride[1] != 1 ||
        padding.size() != 2 || padding[0] != 1 || padding[1] != 1 ||
        dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1 ||
        transposed ||
        output_padding.size() != 2 ||
        output_padding[0] != 0 || output_padding[1] != 0 ||
        groups != 1) {
      return nullptr;
    }
    // 128->128 residual dX has exactly the same M/K/N geometry as the
    // resident res146 forward stream; only the weight packing differs. Reuse
    // that physical program instead of introducing another shape-specific
    // instruction stream.
    if (weight.size(1) == 128 &&
        grad_output.size(2) == 146 && grad_output.size(3) == 141 &&
        forward_streams[0].instr) {
      return &forward_streams[0];
    }
    for (auto& stream : dx_streams) {
      if (!stream.instr ||
          stream.cin != grad_output.size(1) ||
          grad_output.size(3) != stream.w)
        continue;
      if (grad_output.size(2) == stream.h)
        return &stream;
      if (grad_output.size(2) % stream.h == 0)
        return &stream;
    }
    return nullptr;
  }

  at::Tensor dispatch_raw(
      XdnaFullFrameConvStream& stream,
      at::ScalarType dtype) {
    auto raw = empty_memory_format(
        {c10::SymInt(stream.m_sched), c10::SymInt(stream.n_phys)},
        dtype,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::Contiguous);
    auto* out = allocation_from_tensor(raw);
    TORCH_CHECK(
        shim_run_matmul8(
            kernel, 3, stream.instr, stream.instr_words,
            stream.a, stream.b, out->bo, stream.tmp, stream.trace) == 0,
        "full-frame XDNA Conv dispatch failed: ",
        shim_last_error());
    mark_device_dirty(raw);
    return raw;
  }

  at::Tensor logical_alias(
      const at::Tensor& raw,
      const XdnaFullFrameConvStream& stream,
      int64_t channels) {
    std::vector<c10::SymInt> sizes = {
        c10::SymInt(kBatch), c10::SymInt(channels),
        c10::SymInt(stream.h), c10::SymInt(stream.w)};
    std::vector<c10::SymInt> strides = {
        c10::SymInt(stream.n_phys),
        c10::SymInt(1),
        c10::SymInt(stream.w_sched * kBatch * stream.n_phys),
        c10::SymInt(kBatch * stream.n_phys)};
    return make_alias(raw, sizes, strides, c10::SymInt(0));
  }

  at::Tensor forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      XdnaFullFrameConvStream& stream) {
    ensure_host_current(input);
    ensure_host_current(weight);

    at::Tensor raw;
    {
      std::lock_guard<std::mutex> guard(mutex);
      if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
        xdna_pack_conv3x3_halo_padded_channels_last_bf16(
            static_cast<const uint16_t*>(input.const_data_ptr()),
            static_cast<uint16_t*>(stream.a_ptr),
            static_cast<int>(kBatch),
            static_cast<int>(stream.cin),
            static_cast<int>(stream.h),
            static_cast<int>(stream.w),
            static_cast<int>(stream.w_sched));
      } else {
        xdna_pack_conv3x3_halo_padded_bf16(
            static_cast<const uint16_t*>(input.const_data_ptr()),
            static_cast<uint16_t*>(stream.a_ptr),
            static_cast<int>(kBatch),
            static_cast<int>(stream.cin),
            static_cast<int>(stream.h),
            static_cast<int>(stream.w),
            static_cast<int>(stream.w_sched));
      }
      xdna_pack_conv3x3_weight_fwd_bf16(
          static_cast<const uint16_t*>(weight.const_data_ptr()),
          static_cast<uint16_t*>(stream.b_ptr),
          static_cast<int>(kCout),
          static_cast<int>(stream.cin));
      TORCH_CHECK(
          shim_bo_sync_to_device(stream.a) == 0 &&
              shim_bo_sync_to_device(stream.b) == 0,
          "failed to sync full-frame Conv operands: ",
          shim_last_error());
      raw = dispatch_raw(stream, at::ScalarType::BFloat16);
    }

    // The array naturally produces [Y,X_sched,B,C]. PyTorch BatchNorm is
    // extremely slow on that logical-stride view, especially backward. Pay one
    // explicit compact reorder here and keep the region in channels-last
    // storage thereafter.
    ensure_host_current(raw);
    auto out = empty_memory_format(
        {
            c10::SymInt(kBatch),
            c10::SymInt(kCout),
            c10::SymInt(stream.h),
            c10::SymInt(stream.w),
        },
        at::ScalarType::BFloat16,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::ChannelsLast);
    xdna_yxbc_to_channels_last_bf16(
        static_cast<const uint16_t*>(raw.const_data_ptr()),
        static_cast<uint16_t*>(out.mutable_data_ptr()),
        static_cast<int>(kBatch),
        static_cast<int>(kCout),
        static_cast<int>(stream.h),
        static_cast<int>(stream.w),
        static_cast<int>(stream.w_sched));
    mark_host_dirty(out);
    return out;
  }

  at::Tensor forward_cat2_upsample2(
      const at::Tensor& src0,
      const at::Tensor& src1,
      const at::Tensor& weight,
      XdnaFullFrameConvStream& stream) {
    TORCH_CHECK(
        src0.scalar_type() == at::ScalarType::BFloat16 &&
        src1.scalar_type() == at::ScalarType::BFloat16 &&
        weight.scalar_type() == at::ScalarType::BFloat16,
        "fused XDNA cat+upsample Conv requires BF16");
    TORCH_CHECK(
        src0.dim() == 4 && src1.dim() == 4 &&
        src0.size(0) == kBatch && src1.size(0) == kBatch &&
        src0.size(2) == src1.size(2) && src0.size(3) == src1.size(3) &&
        src0.size(2) * 2 == stream.h && src0.size(3) * 2 == stream.w &&
        src0.size(1) + src1.size(1) == stream.cin &&
        src0.size(1) % kCBlock == 0 && src1.size(1) % kCBlock == 0,
        "fused XDNA cat+upsample Conv shape mismatch");
    TORCH_CHECK(
        src0.is_contiguous(at::MemoryFormat::ChannelsLast) &&
        src1.is_contiguous(at::MemoryFormat::ChannelsLast) &&
        weight.is_contiguous(),
        "fused XDNA cat+upsample Conv requires channels-last sources and contiguous weight");
    TORCH_CHECK(
        weight.size(0) == kCout && weight.size(1) == stream.cin &&
        weight.size(2) == 3 && weight.size(3) == 3,
        "fused XDNA cat+upsample Conv weight mismatch");

    ensure_host_current(src0);
    ensure_host_current(src1);
    ensure_host_current(weight);

    at::Tensor raw;
    {
      std::lock_guard<std::mutex> guard(mutex);
      xdna_pack_conv3x3_halo_cat2_upsample2_channels_last_bf16(
          static_cast<const uint16_t*>(src0.const_data_ptr()),
          static_cast<int>(src0.size(1)),
          static_cast<const uint16_t*>(src1.const_data_ptr()),
          static_cast<int>(src1.size(1)),
          static_cast<uint16_t*>(stream.a_ptr),
          static_cast<int>(kBatch),
          static_cast<int>(src0.size(2)),
          static_cast<int>(src0.size(3)),
          static_cast<int>(stream.w_sched));
      xdna_pack_conv3x3_weight_fwd_bf16(
          static_cast<const uint16_t*>(weight.const_data_ptr()),
          static_cast<uint16_t*>(stream.b_ptr),
          static_cast<int>(kCout),
          static_cast<int>(stream.cin));
      TORCH_CHECK(
          shim_bo_sync_to_device(stream.a) == 0 &&
              shim_bo_sync_to_device(stream.b) == 0,
          "failed to sync fused cat+upsample Conv operands: ",
          shim_last_error());
      raw = dispatch_raw(stream, at::ScalarType::BFloat16);
    }

    ensure_host_current(raw);
    auto out = empty_memory_format(
        {c10::SymInt(kBatch), c10::SymInt(kCout),
         c10::SymInt(stream.h), c10::SymInt(stream.w)},
        at::ScalarType::BFloat16,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::ChannelsLast);
    xdna_yxbc_to_channels_last_bf16(
        static_cast<const uint16_t*>(raw.const_data_ptr()),
        static_cast<uint16_t*>(out.mutable_data_ptr()),
        static_cast<int>(kBatch), static_cast<int>(kCout),
        static_cast<int>(stream.h), static_cast<int>(stream.w),
        static_cast<int>(stream.w_sched));
    mark_host_dirty(out);
    return out;
  }

  XdnaFullFrameConvStream* find_fused_cat2_upsample2(
      const at::Tensor& src0,
      const at::Tensor& src1,
      const at::Tensor& weight) {
    if (src0.dim() != 4 || src1.dim() != 4 || weight.dim() != 4 ||
        src0.size(0) != kBatch || src1.size(0) != kBatch ||
        src0.size(2) != src1.size(2) || src0.size(3) != src1.size(3) ||
        weight.size(0) != kCout || weight.size(2) != 3 || weight.size(3) != 3) {
      return nullptr;
    }
    for (auto& stream : forward_streams) {
      if (stream.instr &&
          stream.h == src0.size(2) * 2 &&
          stream.w == src0.size(3) * 2 &&
          stream.cin == src0.size(1) + src1.size(1) &&
          weight.size(1) == stream.cin) {
        return &stream;
      }
    }
    return nullptr;
  }

  at::Tensor input_grad(
      const at::Tensor& grad_output,
      const at::Tensor& weight,
      XdnaFullFrameConvStream& stream) {
    ensure_host_current(grad_output);
    ensure_host_current(weight);
    auto weight_cpu = cpu_alias(weight);
    TORCH_CHECK(
        weight_cpu.is_contiguous(),
        "full-frame decoder dX weight must be contiguous");

    const int64_t full_h = grad_output.size(2);
    const int64_t full_w = grad_output.size(3);
    TORCH_CHECK(
        full_w == stream.w && full_h % stream.h == 0,
        "full-frame dX stripe does not tile logical gradient: full=",
        full_h, "x", full_w, " stripe=", stream.h, "x", stream.w);

    const int64_t cin = weight.size(1);
    const int64_t grad_c = weight.size(0);
    TORCH_CHECK(
        stream.cin == grad_c,
        "full-frame dX stream input channels mismatch: stream=",
        stream.cin, " grad=", grad_c);
    const int nslices =
        static_cast<int>((cin + stream.n_phys - 1) / stream.n_phys);

    auto out = empty_memory_format(
        {c10::SymInt(kBatch), c10::SymInt(cin),
         c10::SymInt(full_h), c10::SymInt(full_w)},
        at::ScalarType::BFloat16,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::ChannelsLast);

    at::Tensor go_cl;
    const uint16_t* go_ptr = nullptr;
    if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      go_ptr = static_cast<const uint16_t*>(grad_output.const_data_ptr());
    } else {
      auto go_cpu = cpu_alias(grad_output);
      go_cl = go_cpu.contiguous(at::MemoryFormat::ChannelsLast);
      go_ptr = static_cast<const uint16_t*>(go_cl.const_data_ptr());
    }

    std::lock_guard<std::mutex> guard(mutex);
    for (int64_t y0 = 0; y0 < full_h; y0 += stream.h) {
      xdna_pack_conv3x3_halo_padded_channels_last_stripe_bf16(
          go_ptr,
          static_cast<uint16_t*>(stream.a_ptr),
          static_cast<int>(kBatch),
          static_cast<int>(stream.cin),
          static_cast<int>(full_h),
          static_cast<int>(full_w),
          static_cast<int>(y0),
          static_cast<int>(stream.h),
          static_cast<int>(stream.w_sched));
      TORCH_CHECK(
          shim_bo_sync_to_device(stream.a) == 0,
          "failed to sync full-frame decoder dX stripe: ",
          shim_last_error());

      for (int slice = 0; slice < nslices; ++slice) {
        const int start_channel = slice * static_cast<int>(stream.n_phys);
        const int copy_channels = static_cast<int>(
            std::min<int64_t>(stream.n_phys, cin - start_channel));
        if (copy_channels == stream.n_phys) {
          xdna_pack_conv3x3_weight_dx_slice_bf16(
              static_cast<const uint16_t*>(weight_cpu.const_data_ptr()),
              static_cast<uint16_t*>(stream.b_ptr),
              static_cast<int>(grad_c),
              static_cast<int>(cin),
              start_channel,
              static_cast<int>(stream.n_phys));
        } else {
          xdna_pack_conv3x3_weight_dx_slice_padded_bf16(
              static_cast<const uint16_t*>(weight_cpu.const_data_ptr()),
              static_cast<uint16_t*>(stream.b_ptr),
              static_cast<int>(grad_c),
              static_cast<int>(cin),
              start_channel,
              copy_channels,
              static_cast<int>(stream.n_phys));
        }
        TORCH_CHECK(
            shim_bo_sync_to_device(stream.b) == 0,
            "failed to sync full-frame decoder dX weight slice: ",
            shim_last_error());

        auto raw = dispatch_raw(stream, at::ScalarType::BFloat16);
        ensure_host_current(raw);
        if (copy_channels == stream.n_phys) {
          xdna_yxbc_slice_stripe_to_channels_last_bf16(
              static_cast<const uint16_t*>(raw.const_data_ptr()),
              static_cast<uint16_t*>(out.mutable_data_ptr()),
              static_cast<int>(kBatch),
              static_cast<int>(cin),
              static_cast<int>(stream.n_phys),
              start_channel,
              static_cast<int>(full_h),
              static_cast<int>(y0),
              static_cast<int>(stream.h),
              static_cast<int>(full_w),
              static_cast<int>(stream.w_sched));
        } else {
          xdna_yxbc_slice_stripe_to_channels_last_partial_bf16(
              static_cast<const uint16_t*>(raw.const_data_ptr()),
              static_cast<uint16_t*>(out.mutable_data_ptr()),
              static_cast<int>(kBatch),
              static_cast<int>(cin),
              static_cast<int>(stream.n_phys),
              copy_channels,
              start_channel,
              static_cast<int>(full_h),
              static_cast<int>(y0),
              static_cast<int>(stream.h),
              static_cast<int>(full_w),
              static_cast<int>(stream.w_sched));
        }
      }
    }
    mark_host_dirty(out);
    return out;
  }

};

struct XdnaFullFrameDwRuntime {
  static constexpr int64_t kBatch = 4;
  static constexpr int64_t kCin = 256;
  static constexpr int64_t kCout = 128;
  static constexpr int64_t kH = 584;
  static constexpr int64_t kW = 564;
  static constexpr int64_t kM = 9 * kCin;
  static constexpr int64_t kK = kBatch * kH * kW;

  ShimKernel* kernel = nullptr;
  ShimBo* instr = nullptr;
  ShimBo* a = nullptr;
  ShimBo* b = nullptr;
  ShimBo* tmp = nullptr;
  ShimBo* trace = nullptr;
  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  size_t instr_words = 0;
  std::mutex mutex;

  ~XdnaFullFrameDwRuntime() {
    for (auto* bo : {instr, a, b, tmp, trace}) {
      if (bo) shim_bo_free(bo);
    }
    if (kernel) shim_kernel_close(kernel);
  }

  void open(const std::string& xclbin, const std::string& insts) {
    auto blob = XdnaGemmRuntime::read_file(insts);
    TORCH_CHECK(
        blob.size() % 4 == 0,
        "full-frame dW instruction stream is not word aligned: ",
        insts);
    kernel = shim_kernel_load(
        xdna_device(), xclbin.c_str(), nullptr, QOS_PRIORITY_NONE);
    TORCH_CHECK(
        kernel != nullptr,
        "failed to load full-frame dW XDNA program: ",
        shim_last_error());

    int pack_threads = 12;
    if (const char* raw = std::getenv("XDNA_CONV_PACK_THREADS")) {
      const int parsed = std::atoi(raw);
      if (parsed > 0) pack_threads = parsed;
    }
    xdna_conv_set_threads(pack_threads);

    const int gid_instr = shim_kernel_group_id(kernel, 1);
    const int gid_tmp = shim_kernel_group_id(kernel, 6);
    const int gid_trace = shim_kernel_group_id(kernel, 7);
    const size_t halo_elems =
        static_cast<size_t>(kH + 2) * (kW + 2) * kCin * kBatch;
    const size_t dy_elems = static_cast<size_t>(kK) * kCout;

    instr = shim_bo_alloc(
        xdna_device(), kernel, blob.size(), 1, gid_instr);
    a = shim_bo_alloc(
        xdna_device(), kernel, halo_elems * sizeof(uint16_t), 2, 0);
    b = shim_bo_alloc(
        xdna_device(), kernel, dy_elems * sizeof(uint16_t), 2, 0);
    tmp = shim_bo_alloc(xdna_device(), kernel, 1, 2, gid_tmp);
    trace = shim_bo_alloc(xdna_device(), kernel, 4, 2, gid_trace);
    TORCH_CHECK(
        instr && a && b && tmp && trace,
        "failed to allocate full-frame dW XDNA buffers: ",
        shim_last_error());
    TORCH_CHECK(
        shim_bo_write(instr, blob.data(), blob.size(), 0) == 0 &&
            shim_bo_sync_to_device(instr) == 0,
        "failed to initialize full-frame dW instruction stream: ",
        shim_last_error());
    a_ptr = shim_bo_map(a);
    b_ptr = shim_bo_map(b);
    TORCH_CHECK(
        a_ptr && b_ptr,
        "failed to map full-frame dW staging buffers: ",
        shim_last_error());
    instr_words = blob.size() / 4;
  }

  bool supports(
      const at::Tensor& input,
      const at::Tensor& grad_output,
      const at::Tensor& weight,
      c10::SymIntArrayRef stride,
      c10::SymIntArrayRef padding,
      c10::SymIntArrayRef dilation,
      bool transposed,
      c10::SymIntArrayRef output_padding,
      c10::SymInt groups) const {
    return input.scalar_type() == at::ScalarType::BFloat16 &&
        grad_output.scalar_type() == at::ScalarType::BFloat16 &&
        weight.scalar_type() == at::ScalarType::BFloat16 &&
        input.dim() == 4 && grad_output.dim() == 4 && weight.dim() == 4 &&
        input.size(0) == kBatch && input.size(1) == kCin &&
        input.size(2) == kH && input.size(3) == kW &&
        grad_output.size(0) == kBatch &&
        grad_output.size(1) == kCout &&
        grad_output.size(2) == kH && grad_output.size(3) == kW &&
        weight.size(0) == kCout && weight.size(1) == kCin &&
        weight.size(2) == 3 && weight.size(3) == 3 &&
        stride.size() == 2 && stride[0] == 1 && stride[1] == 1 &&
        padding.size() == 2 && padding[0] == 1 && padding[1] == 1 &&
        dilation.size() == 2 && dilation[0] == 1 && dilation[1] == 1 &&
        !transposed &&
        output_padding.size() == 2 &&
        output_padding[0] == 0 && output_padding[1] == 0 &&
        groups == 1;
  }

  at::Tensor weight_grad(
      const at::Tensor& input,
      const at::Tensor& grad_output,
      const at::Tensor& weight) {
    ensure_host_current(input);
    ensure_host_current(grad_output);

    at::Tensor input_cl;
    const uint16_t* input_ptr = nullptr;
    if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      input_ptr = static_cast<const uint16_t*>(input.const_data_ptr());
    } else {
      auto input_cpu = cpu_alias(input);
      input_cl = input_cpu.contiguous(at::MemoryFormat::ChannelsLast);
      input_ptr = static_cast<const uint16_t*>(input_cl.const_data_ptr());
    }

    at::Tensor go_cl;
    const uint16_t* go_ptr = nullptr;
    if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      go_ptr = static_cast<const uint16_t*>(grad_output.const_data_ptr());
    } else {
      auto go_cpu = cpu_alias(grad_output);
      go_cl = go_cpu.contiguous(at::MemoryFormat::ChannelsLast);
      go_ptr = static_cast<const uint16_t*>(go_cl.const_data_ptr());
    }

    std::lock_guard<std::mutex> guard(mutex);
    xdna_pack_conv3x3_halo_padded_channels_last_bf16(
        input_ptr,
        static_cast<uint16_t*>(a_ptr),
        static_cast<int>(kBatch),
        static_cast<int>(kCin),
        static_cast<int>(kH),
        static_cast<int>(kW),
        static_cast<int>(kW));
    xdna_pack_conv3x3_dy_channels_last_bf16(
        go_ptr,
        static_cast<uint16_t*>(b_ptr),
        static_cast<int>(kBatch),
        static_cast<int>(kCout),
        static_cast<int>(kH),
        static_cast<int>(kW));
    TORCH_CHECK(
        shim_bo_sync_to_device(a) == 0 &&
            shim_bo_sync_to_device(b) == 0,
        "failed to sync full-frame dW operands: ",
        shim_last_error());

    auto raw = empty_memory_format(
        {c10::SymInt(kM), c10::SymInt(kCout)},
        at::ScalarType::Float,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::Contiguous);
    auto* raw_alloc = allocation_from_tensor(raw);
    TORCH_CHECK(
        shim_run_matmul8(
            kernel, 3, instr, instr_words,
            a, b, raw_alloc->bo, tmp, trace) == 0,
        "full-frame dW XDNA dispatch failed: ",
        shim_last_error());
    mark_device_dirty(raw);
    ensure_host_current(raw);

    auto gw = empty_memory_format(
        weight.sym_sizes(),
        weight.scalar_type(),
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::Contiguous);
    auto raw_cpu = cpu_alias(raw);
    auto gw_cpu = cpu_alias(gw);
    auto logical = raw_cpu
        .view({3, 3, kCin, kCout})
        .permute({3, 2, 0, 1});
    gw_cpu.copy_(logical);
    mark_host_dirty(gw);
    return gw;
  }
};

std::unique_ptr<XdnaFullFrameDwRuntime>& fullframe_dw_runtime_slot() {
  static auto* slot = new std::unique_ptr<XdnaFullFrameDwRuntime>();
  return *slot;
}

std::unique_ptr<XdnaFullFrameConvRuntime>& fullframe_conv_runtime_slot() {
  static auto* slot = new std::unique_ptr<XdnaFullFrameConvRuntime>();
  return *slot;
}

std::unique_ptr<XdnaFullFrameConvRuntime>& fast_dx584_runtime_slot() {
  static auto* slot = new std::unique_ptr<XdnaFullFrameConvRuntime>();
  return *slot;
}

std::unique_ptr<XdnaFullFrameConvRuntime>& fast_conv11_runtime_slot() {
  static auto* slot = new std::unique_ptr<XdnaFullFrameConvRuntime>();
  return *slot;
}


at::Tensor xdna_from_cpu(const at::Tensor& cpu);
at::Tensor adopt_or_copy(const at::Tensor& cpu);

at::Tensor xdna_empty_like_shape(
    c10::SymIntArrayRef sizes,
    at::ScalarType dtype) {
  return empty_memory_format(
      sizes,
      dtype,
      at::Layout::Strided,
      c10::Device(kXdnaType, 0),
      false,
      at::MemoryFormat::Contiguous);
}

at::Tensor xdna_empty_like_layout(const at::Tensor& reference) {
  const auto format =
      reference.dim() == 4 &&
          reference.is_contiguous(at::MemoryFormat::ChannelsLast)
      ? at::MemoryFormat::ChannelsLast
      : at::MemoryFormat::Contiguous;
  return empty_memory_format(
      reference.sym_sizes(),
      reference.scalar_type(),
      at::Layout::Strided,
      c10::Device(kXdnaType, 0),
      false,
      format);
}


at::Tensor leaky_relu_xdna(
    const at::Tensor& self,
    const at::Scalar& negative_slope) {
  ScopedMappedCpu mapped_phase("aten::leaky_relu");
  ensure_host_current(self);
  auto out = xdna_empty_like_layout(self);
  auto in_cpu = cpu_alias(self);
  auto out_cpu = cpu_alias(out);
  at::leaky_relu_out(out_cpu, in_cpu, negative_slope);
  mark_host_dirty(out);
  return out;
}

at::Tensor leaky_relu_backward_xdna(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negative_slope,
    bool self_is_result) {
  ScopedMappedCpu mapped_phase("aten::leaky_relu_backward");
  ensure_host_current(grad_output);
  ensure_host_current(self);
  auto out = xdna_empty_like_layout(self);
  auto go_cpu = cpu_alias(grad_output);
  auto self_cpu = cpu_alias(self);
  if (self.dim() == 4) {
    const bool self_cl =
        self.is_contiguous(at::MemoryFormat::ChannelsLast);
    const bool grad_cl =
        grad_output.is_contiguous(at::MemoryFormat::ChannelsLast);
    if (self_cl && !grad_cl) {
      go_cpu = go_cpu.contiguous(at::MemoryFormat::ChannelsLast);
    } else if (!self_cl && grad_cl && self.is_contiguous()) {
      go_cpu = go_cpu.contiguous(at::MemoryFormat::Contiguous);
    }
  }
  auto out_cpu = cpu_alias(out);
  at::leaky_relu_backward_out(
      out_cpu, go_cpu, self_cpu, negative_slope, self_is_result);
  mark_host_dirty(out);
  return out;
}

at::Tensor add_tensor_xdna(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  ScopedMappedCpu mapped_phase("aten::add.Tensor");
  ensure_host_current(self);
  ensure_host_current(other);
  auto self_cpu = cpu_alias(self);
  auto other_cpu = cpu_alias(other);

  if (self.scalar_type() != other.scalar_type()) {
    return xdna_from_cpu(at::add(self_cpu, other_cpu, alpha));
  }

  const auto shape = at::infer_size_dimvector(self.sizes(), other.sizes());
  std::vector<c10::SymInt> sym_shape;
  sym_shape.reserve(shape.size());
  for (auto v : shape) sym_shape.emplace_back(v);
  const bool preserve_channels_last =
      self.dim() == 4 && other.dim() == 4 &&
      self.sym_sizes().equals(other.sym_sizes()) &&
      (self.is_contiguous(at::MemoryFormat::ChannelsLast) ||
       other.is_contiguous(at::MemoryFormat::ChannelsLast));
  auto out = preserve_channels_last
      ? empty_memory_format(
            sym_shape,
            self.scalar_type(),
            at::Layout::Strided,
            c10::Device(kXdnaType, 0),
            false,
            at::MemoryFormat::ChannelsLast)
      : xdna_empty_like_shape(sym_shape, self.scalar_type());
  auto out_cpu = cpu_alias(out);
  at::add_out(out_cpu, self_cpu, other_cpu, alpha);
  mark_host_dirty(out);
  return out;
}


at::Tensor& add_out_xdna(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::add.out");
  ensure_host_current(self);
  ensure_host_current(other);
  auto self_cpu = cpu_alias(self);
  auto other_cpu = cpu_alias(other);
  auto out_cpu = cpu_alias(out);
  at::add_out(out_cpu, self_cpu, other_cpu, alpha);
  mark_host_dirty(out);
  return out;
}


at::Tensor& mul_out_xdna(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::mul.out");
  ensure_host_current(self);
  ensure_host_current(other);
  auto self_cpu = cpu_alias(self);
  auto other_cpu = cpu_alias(other);
  auto out_cpu = cpu_alias(out);
  at::_ops::mul_out::call(self_cpu, other_cpu, out_cpu);
  mark_host_dirty(out);
  return out;
}

at::Tensor& div_out_xdna(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::div.out");
  ensure_host_current(self);
  ensure_host_current(other);
  auto self_cpu = cpu_alias(self);
  auto other_cpu = cpu_alias(other);
  auto out_cpu = cpu_alias(out);
  at::_ops::div_out::call(self_cpu, other_cpu, out_cpu);
  mark_host_dirty(out);
  return out;
}

at::Tensor& sqrt_out_xdna(
    const at::Tensor& self,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::sqrt.out");
  ensure_host_current(self);
  auto self_cpu = cpu_alias(self);
  auto out_cpu = cpu_alias(out);
  at::_ops::sqrt_out::call(self_cpu, out_cpu);
  mark_host_dirty(out);
  return out;
}

at::Tensor& addcmul_out_xdna(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::addcmul.out");
  ensure_host_current(self);
  ensure_host_current(tensor1);
  ensure_host_current(tensor2);
  auto self_cpu = cpu_alias(self);
  auto tensor1_cpu = cpu_alias(tensor1);
  auto tensor2_cpu = cpu_alias(tensor2);
  auto out_cpu = cpu_alias(out);
  at::_ops::addcmul_out::call(
      self_cpu, tensor1_cpu, tensor2_cpu, value, out_cpu);
  mark_host_dirty(out);
  return out;
}

at::Tensor& addcdiv_out_xdna(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::addcdiv.out");
  ensure_host_current(self);
  ensure_host_current(tensor1);
  ensure_host_current(tensor2);
  auto self_cpu = cpu_alias(self);
  auto tensor1_cpu = cpu_alias(tensor1);
  auto tensor2_cpu = cpu_alias(tensor2);
  auto out_cpu = cpu_alias(out);
  at::_ops::addcdiv_out::call(
      self_cpu, tensor1_cpu, tensor2_cpu, value, out_cpu);
  mark_host_dirty(out);
  return out;
}

at::Tensor& lerp_scalar_out_xdna(
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Scalar& weight,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::lerp.Scalar_out");
  ensure_host_current(self);
  ensure_host_current(end);
  auto self_cpu = cpu_alias(self);
  auto end_cpu = cpu_alias(end);
  auto out_cpu = cpu_alias(out);
  at::_ops::lerp_Scalar_out::call(self_cpu, end_cpu, weight, out_cpu);
  mark_host_dirty(out);
  return out;
}

at::Tensor& sigmoid_out_xdna(
    const at::Tensor& self,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::sigmoid.out");
  ensure_host_current(self);
  auto self_cpu = cpu_alias(self);
  auto out_cpu = cpu_alias(out);
  at::_ops::sigmoid_out::call(self_cpu, out_cpu);
  mark_host_dirty(out);
  return out;
}

at::Tensor& sub_out_xdna(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::sub.out");
  ensure_host_current(self);
  ensure_host_current(other);
  auto self_cpu = cpu_alias(self);
  auto other_cpu = cpu_alias(other);
  auto out_cpu = cpu_alias(out);
  at::_ops::sub_out::call(self_cpu, other_cpu, alpha, out_cpu);
  mark_host_dirty(out);
  return out;
}


at::Tensor& addmm_out_xdna(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::addmm.out");
  ensure_host_current(self);
  ensure_host_current(mat1);
  ensure_host_current(mat2);
  auto self_cpu = cpu_alias(self);
  auto mat1_cpu = cpu_alias(mat1);
  auto mat2_cpu = cpu_alias(mat2);
  auto out_cpu = cpu_alias(out);
  at::_ops::addmm_out::call(
      self_cpu, mat1_cpu, mat2_cpu, beta, alpha, out_cpu);
  mark_host_dirty(out);
  return out;
}

at::Tensor& bmm_out_xdna(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  ScopedMappedCpu mapped_phase("aten::bmm.out");
  ensure_host_current(self);
  ensure_host_current(mat2);
  auto self_cpu = cpu_alias(self);
  auto mat2_cpu = cpu_alias(mat2);
  auto out_cpu = cpu_alias(out);
  at::_ops::bmm_out::call(self_cpu, mat2_cpu, out_cpu);
  mark_host_dirty(out);
  return out;
}

bool lazy_decoder_values_enabled() {
  const char* raw = std::getenv("XDNA_LAZY_DECODER_VALUES");
  return raw != nullptr && raw[0] != 0 && std::strcmp(raw, "0") != 0;
}

// Fast layout ops.  ATen's CPU cat / nearest-upsample kernels run far below
// memory bandwidth on this mapped memory, so the common contiguous cases are
// plain OpenMP copy loops.  Widths come from g_host_threads, never a global
// omp_set_num_threads.
int layout_threads(size_t bytes) {
  if (bytes < (1u << 20)) return 1;
  const int n = g_host_threads.load(std::memory_order_relaxed);
  return n > 0 ? n : 1;
}

// Concatenate contiguous tensors along `dim`: for each outer index every
// source contributes one contiguous block.
void cat_contiguous_blocks(
    const std::vector<const char*>& src,
    const std::vector<int64_t>& block_bytes,
    char* dst,
    int64_t outer,
    int64_t out_block_bytes) {
  const int64_t n = static_cast<int64_t>(src.size());
  std::vector<int64_t> off(static_cast<size_t>(n));
  int64_t acc = 0;
  for (int64_t i = 0; i < n; ++i) {
    off[static_cast<size_t>(i)] = acc;
    acc += block_bytes[static_cast<size_t>(i)];
  }
  const int nt = layout_threads(static_cast<size_t>(outer * out_block_bytes));
#pragma omp parallel for num_threads(nt) collapse(2) schedule(static)
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < n; ++i) {
      const size_t bi = static_cast<size_t>(i);
      std::memcpy(
          dst + o * out_block_bytes + off[bi],
          src[bi] + o * block_bytes[bi],
          static_cast<size_t>(block_bytes[bi]));
    }
  }
}

template <typename T>
void upsample2x_nchw(const T* in, T* out, int64_t planes, int64_t h, int64_t w) {
  const int nt = layout_threads(static_cast<size_t>(planes * h * w * 4 * sizeof(T)));
#pragma omp parallel for num_threads(nt) collapse(2) schedule(static)
  for (int64_t p = 0; p < planes; ++p) {
    for (int64_t y = 0; y < h; ++y) {
      const T* src = in + (p * h + y) * w;
      T* row0 = out + (p * h + y) * 2 * (2 * w);
      for (int64_t x = 0; x < w; ++x) {
        row0[2 * x] = src[x];
        row0[2 * x + 1] = src[x];
      }
      std::memcpy(row0 + 2 * w, row0, static_cast<size_t>(2 * w) * sizeof(T));
    }
  }
}

template <typename T>
void upsample2x_nhwc(const T* in, T* out, int64_t n, int64_t c, int64_t h, int64_t w) {
  const int nt = layout_threads(static_cast<size_t>(n * c * h * w * 4 * sizeof(T)));
#pragma omp parallel for num_threads(nt) collapse(2) schedule(static)
  for (int64_t b = 0; b < n; ++b) {
    for (int64_t y = 0; y < h; ++y) {
      for (int64_t x = 0; x < w; ++x) {
        const T* src = in + ((b * h + y) * w + x) * c;
        T* d00 = out + ((b * 2 * h + 2 * y) * 2 * w + 2 * x) * c;
        std::memcpy(d00, src, static_cast<size_t>(c) * sizeof(T));
        std::memcpy(d00 + c, src, static_cast<size_t>(c) * sizeof(T));
        std::memcpy(d00 + 2 * w * c, src, static_cast<size_t>(c) * sizeof(T));
        std::memcpy(d00 + (2 * w + 1) * c, src, static_cast<size_t>(c) * sizeof(T));
      }
    }
  }
}

// Sum of each 2x2 block in fp32, starting from 0 and in raster order like
// ATen's accumulation (bit-exact for fp32; bf16 rounds once at the end).
template <typename T>
void upsample2x_backward_nchw(const T* go, T* gi, int64_t planes, int64_t h, int64_t w) {
  const int nt = layout_threads(static_cast<size_t>(planes * h * w * 4 * sizeof(T)));
#pragma omp parallel for num_threads(nt) collapse(2) schedule(static)
  for (int64_t p = 0; p < planes; ++p) {
    for (int64_t y = 0; y < h; ++y) {
      const T* r0 = go + (p * h + y) * 2 * (2 * w);
      const T* r1 = r0 + 2 * w;
      T* dst = gi + (p * h + y) * w;
      for (int64_t x = 0; x < w; ++x) {
        float acc = 0.f;
        acc += static_cast<float>(r0[2 * x]);
        acc += static_cast<float>(r0[2 * x + 1]);
        acc += static_cast<float>(r1[2 * x]);
        acc += static_cast<float>(r1[2 * x + 1]);
        dst[x] = static_cast<T>(acc);
      }
    }
  }
}

template <typename T>
void upsample2x_backward_nhwc(const T* go, T* gi, int64_t n, int64_t c, int64_t h, int64_t w) {
  const int nt = layout_threads(static_cast<size_t>(n * c * h * w * 4 * sizeof(T)));
#pragma omp parallel for num_threads(nt) collapse(2) schedule(static)
  for (int64_t b = 0; b < n; ++b) {
    for (int64_t y = 0; y < h; ++y) {
      for (int64_t x = 0; x < w; ++x) {
        const T* p00 = go + ((b * 2 * h + 2 * y) * 2 * w + 2 * x) * c;
        const T* p01 = p00 + c;
        const T* p10 = p00 + 2 * w * c;
        const T* p11 = p10 + c;
        T* dst = gi + ((b * h + y) * w + x) * c;
        for (int64_t k = 0; k < c; ++k) {
          float acc = 0.f;
          acc += static_cast<float>(p00[k]);
          acc += static_cast<float>(p01[k]);
          acc += static_cast<float>(p10[k]);
          acc += static_cast<float>(p11[k]);
          dst[k] = static_cast<T>(acc);
        }
      }
    }
  }
}

// Exact-2x nearest resize of bf16/fp32 4-D tensors in NCHW or NHWC memory
// order; anything else returns false and keeps the ATen path.
bool nearest2x_scales_ok(std::optional<double> sh, std::optional<double> sw) {
  return (!sh || *sh == 2.0) && (!sw || *sw == 2.0);
}

enum class Layout2x { None, Nchw, Nhwc };

Layout2x layout_2x(const at::Tensor& a, const at::Tensor& b) {
  if (a.is_contiguous() && b.is_contiguous()) return Layout2x::Nchw;
  if (a.is_contiguous(at::MemoryFormat::ChannelsLast) &&
      b.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return Layout2x::Nhwc;
  }
  return Layout2x::None;
}

bool upsample2x_dtype_ok(at::ScalarType t) {
  return t == at::ScalarType::BFloat16 || t == at::ScalarType::Float;
}

at::Tensor cat_xdna(
    const at::ITensorListRef& tensors,
    int64_t dim) {
  ScopedMappedCpu mapped_phase("aten::cat");
  TORCH_CHECK(tensors.size() > 0, "XDNA cat expects at least one tensor");
  const auto first = *tensors.begin();
  TORCH_CHECK(
      first.device().type() == kXdnaType,
      "XDNA cat expects XDNA tensors");
  const int64_t ndim = first.dim();
  dim = c10::maybe_wrap_dim(dim, ndim);
  const auto dtype = first.scalar_type();

  std::vector<c10::SymInt> out_shape(
      first.sym_sizes().begin(), first.sym_sizes().end());
  out_shape[static_cast<size_t>(dim)] = 0;

  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        tensor.device().type() == kXdnaType,
        "XDNA cat expects all tensors on xdna");
    TORCH_CHECK(
        tensor.dim() == ndim && tensor.scalar_type() == dtype,
        "XDNA cat currently requires matching rank and dtype");
    for (int64_t d = 0; d < ndim; ++d) {
      if (d != dim) {
        TORCH_CHECK(
            tensor.sym_size(d) == first.sym_size(d),
            "XDNA cat size mismatch");
      }
    }
    out_shape[static_cast<size_t>(dim)] =
        out_shape[static_cast<size_t>(dim)] + tensor.sym_size(dim);
  }

  bool preserve_channels_last = ndim == 4;
  if (preserve_channels_last) {
    preserve_channels_last = false;
    for (const auto& tensor : tensors) {
      if (tensor.is_contiguous(at::MemoryFormat::ChannelsLast)) {
        preserve_channels_last = true;
        break;
      }
    }
  }
  auto out = preserve_channels_last
      ? empty_memory_format(
            out_shape,
            dtype,
            at::Layout::Strided,
            c10::Device(kXdnaType, 0),
            false,
            at::MemoryFormat::ChannelsLast)
      : xdna_empty_like_shape(out_shape, dtype);

  bool lazy_cat = lazy_decoder_values_enabled() &&
      dim == 1 && ndim == 4 &&
      dtype == at::ScalarType::BFloat16 &&
      tensors.size() == 2 && preserve_channels_last;
  if (lazy_cat) {
    auto value = std::make_shared<XdnaVirtualValue>();
    value->kind = XdnaVirtualKind::CatChannels4D;
    value->dim = dim;
    for (const auto& tensor : tensors) {
      if (!tensor.is_contiguous(at::MemoryFormat::ChannelsLast) ||
          tensor.size(1) % 32 != 0) {
        lazy_cat = false;
        break;
      }
      value->sources.push_back(tensor);
    }
    if (lazy_cat) {
      set_virtual_value(out, std::move(value));
      return out;
    }
  }

  bool fast_cat = out.is_contiguous();
  for (const auto& tensor : tensors) {
    fast_cat = fast_cat && tensor.is_contiguous();
  }
  if (fast_cat) {
    std::vector<const char*> src;
    std::vector<int64_t> block_bytes;
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= first.size(d);
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= first.size(d);
    const int64_t esz = static_cast<int64_t>(first.element_size());
    int64_t out_block_bytes = 0;
    for (const auto& tensor : tensors) {
      ensure_host_current(tensor);
      const int64_t bytes = tensor.size(dim) * inner * esz;
      if (bytes == 0) continue;
      src.push_back(static_cast<const char*>(cpu_alias(tensor).const_data_ptr()));
      block_bytes.push_back(bytes);
      out_block_bytes += bytes;
    }
    if (outer > 0 && out_block_bytes > 0) {
      cat_contiguous_blocks(
          src, block_bytes, static_cast<char*>(cpu_alias(out).data_ptr()),
          outer, out_block_bytes);
    }
    mark_host_dirty(out);
    return out;
  }

  auto out_cpu = cpu_alias(out);
  int64_t offset = 0;
  for (const auto& tensor : tensors) {
    ensure_host_current(tensor);
    auto src_cpu = cpu_alias(tensor);
    const int64_t length = tensor.size(dim);
    out_cpu.narrow(dim, offset, length).copy_(src_cpu);
    offset += length;
  }
  mark_host_dirty(out);
  return out;
}

at::Tensor upsample_nearest2d_xdna(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  ScopedMappedCpu mapped_phase("aten::upsample_nearest2d");
  TORCH_CHECK(
      self.dim() == 4 && output_size.size() == 2,
      "XDNA upsample_nearest2d expects NCHW input and 2-D output size");
  std::vector<c10::SymInt> shape = {
      self.sym_size(0),
      self.sym_size(1),
      output_size[0],
      output_size[1],
  };
  auto out =
      self.is_contiguous(at::MemoryFormat::ChannelsLast)
      ? empty_memory_format(
            shape,
            self.scalar_type(),
            at::Layout::Strided,
            c10::Device(kXdnaType, 0),
            false,
            at::MemoryFormat::ChannelsLast)
      : xdna_empty_like_shape(shape, self.scalar_type());
  std::array<int64_t, 2> output_size_i = {
      output_size[0].guard_int(__FILE__, __LINE__),
      output_size[1].guard_int(__FILE__, __LINE__),
  };

  if (lazy_decoder_values_enabled() &&
      self.scalar_type() == at::ScalarType::BFloat16 &&
      self.is_contiguous(at::MemoryFormat::ChannelsLast) &&
      output_size_i[0] == self.size(2) * 2 &&
      output_size_i[1] == self.size(3) * 2) {
    auto source_value = virtual_value_of(self);
    if (source_value &&
        source_value->kind == XdnaVirtualKind::CatChannels4D &&
        source_value->dim == 1 &&
        source_value->sources.size() == 2) {
      auto value = std::make_shared<XdnaVirtualValue>();
      value->kind = XdnaVirtualKind::UpsampleNearest2D;
      value->sources = {self};
      value->output_size = output_size_i;
      value->scales_h = scales_h;
      value->scales_w = scales_w;
      set_virtual_value(out, std::move(value));
      return out;
    }
  }

  ensure_host_current(self);
  auto in_cpu = cpu_alias(self);
  auto out_cpu = cpu_alias(out);
  if (upsample2x_dtype_ok(self.scalar_type()) && self.numel() > 0 &&
      output_size_i[0] == 2 * self.size(2) &&
      output_size_i[1] == 2 * self.size(3) &&
      nearest2x_scales_ok(scales_h, scales_w)) {
    const Layout2x lay = layout_2x(self, out);
    const int64_t n = self.size(0), c = self.size(1), h = self.size(2), w = self.size(3);
    if (lay != Layout2x::None) {
      const bool bf = self.scalar_type() == at::ScalarType::BFloat16;
      const void* ip = in_cpu.const_data_ptr();
      void* op = out_cpu.data_ptr();
      if (lay == Layout2x::Nchw) {
        if (bf) upsample2x_nchw(static_cast<const uint16_t*>(ip), static_cast<uint16_t*>(op), n * c, h, w);
        else upsample2x_nchw(static_cast<const uint32_t*>(ip), static_cast<uint32_t*>(op), n * c, h, w);
      } else {
        if (bf) upsample2x_nhwc(static_cast<const uint16_t*>(ip), static_cast<uint16_t*>(op), n, c, h, w);
        else upsample2x_nhwc(static_cast<const uint32_t*>(ip), static_cast<uint32_t*>(op), n, c, h, w);
      }
      mark_host_dirty(out);
      return out;
    }
  }
  at::upsample_nearest2d_out(
      out_cpu, in_cpu, output_size_i, scales_h, scales_w);
  mark_host_dirty(out);
  return out;
}

at::Tensor upsample_nearest2d_backward_xdna(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  ScopedMappedCpu mapped_phase("aten::upsample_nearest2d_backward");
  TORCH_CHECK(
      input_size.size() == 4,
      "XDNA upsample_nearest2d_backward expects 4-D input size");
  ensure_host_current(grad_output);
  auto out =
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)
      ? empty_memory_format(
            input_size,
            grad_output.scalar_type(),
            at::Layout::Strided,
            c10::Device(kXdnaType, 0),
            false,
            at::MemoryFormat::ChannelsLast)
      : xdna_empty_like_shape(input_size, grad_output.scalar_type());
  auto go_cpu = cpu_alias(grad_output);
  auto out_cpu = cpu_alias(out);
  std::array<int64_t, 2> output_size_i = {
      output_size[0].guard_int(__FILE__, __LINE__),
      output_size[1].guard_int(__FILE__, __LINE__),
  };
  std::array<int64_t, 4> input_size_i = {
      input_size[0].guard_int(__FILE__, __LINE__),
      input_size[1].guard_int(__FILE__, __LINE__),
      input_size[2].guard_int(__FILE__, __LINE__),
      input_size[3].guard_int(__FILE__, __LINE__),
  };
  if (upsample2x_dtype_ok(grad_output.scalar_type()) && grad_output.numel() > 0 &&
      grad_output.dim() == 4 &&
      output_size_i[0] == 2 * input_size_i[2] &&
      output_size_i[1] == 2 * input_size_i[3] &&
      grad_output.size(2) == output_size_i[0] &&
      grad_output.size(3) == output_size_i[1] &&
      grad_output.size(0) == input_size_i[0] &&
      grad_output.size(1) == input_size_i[1] &&
      nearest2x_scales_ok(scales_h, scales_w)) {
    const Layout2x lay = layout_2x(grad_output, out);
    const int64_t n = input_size_i[0], c = input_size_i[1];
    const int64_t h = input_size_i[2], w = input_size_i[3];
    if (lay != Layout2x::None) {
      const bool bf = grad_output.scalar_type() == at::ScalarType::BFloat16;
      const void* ip = go_cpu.const_data_ptr();
      void* op = out_cpu.data_ptr();
      using BF = c10::BFloat16;
      if (lay == Layout2x::Nchw) {
        if (bf) upsample2x_backward_nchw(static_cast<const BF*>(ip), static_cast<BF*>(op), n * c, h, w);
        else upsample2x_backward_nchw(static_cast<const float*>(ip), static_cast<float*>(op), n * c, h, w);
      } else {
        if (bf) upsample2x_backward_nhwc(static_cast<const BF*>(ip), static_cast<BF*>(op), n, c, h, w);
        else upsample2x_backward_nhwc(static_cast<const float*>(ip), static_cast<float*>(op), n, c, h, w);
      }
      mark_host_dirty(out);
      return out;
    }
  }
  at::upsample_nearest2d_backward_out(
      out_cpu, go_cpu, output_size_i, input_size_i, scales_h, scales_w);
  mark_host_dirty(out);
  return out;
}


std::optional<at::Tensor> optional_cpu_alias_xdna(
    const std::optional<at::Tensor>& tensor) {
  if (!tensor.has_value() || !tensor->defined()) {
    return std::nullopt;
  }
  ensure_host_current(*tensor);
  return cpu_alias(*tensor);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_xdna(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps) {
  ScopedMappedCpu mapped_phase("aten::native_batch_norm");
  TORCH_CHECK(input.dim() >= 2, "XDNA BatchNorm expects rank >= 2 input");
  ensure_host_current(input);
  auto input_cpu = cpu_alias(input);
  auto weight_cpu = optional_cpu_alias_xdna(weight);
  auto bias_cpu = optional_cpu_alias_xdna(bias);
  auto running_mean_cpu = optional_cpu_alias_xdna(running_mean);
  auto running_var_cpu = optional_cpu_alias_xdna(running_var);

  auto out = xdna_empty_like_layout(input);
  const int64_t channels = input.size(1);
  std::vector<c10::SymInt> stat_shape = training
      ? std::vector<c10::SymInt>{c10::SymInt(channels)}
      : std::vector<c10::SymInt>{c10::SymInt(0)};
  auto save_mean = xdna_empty_like_shape(stat_shape, input.scalar_type());
  auto save_invstd = xdna_empty_like_shape(stat_shape, input.scalar_type());

  auto out_cpu = cpu_alias(out);
  auto save_mean_cpu = cpu_alias(save_mean);
  auto save_invstd_cpu = cpu_alias(save_invstd);

  at::native_batch_norm_out(
      out_cpu,
      save_mean_cpu,
      save_invstd_cpu,
      input_cpu,
      weight_cpu,
      bias_cpu,
      running_mean_cpu,
      running_var_cpu,
      training,
      momentum,
      eps);

  mark_host_dirty(out);
  mark_host_dirty(save_mean);
  mark_host_dirty(save_invstd);
  if (running_mean.has_value() && running_mean->defined()) {
    mark_host_dirty(*running_mean);
  }
  if (running_var.has_value() && running_var->defined()) {
    mark_host_dirty(*running_var);
  }
  return {out, save_mean, save_invstd};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_batch_norm_backward_xdna(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    const std::optional<at::Tensor>& save_mean,
    const std::optional<at::Tensor>& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask) {
  ScopedMappedCpu mapped_phase("aten::native_batch_norm_backward");
  ensure_host_current(grad_out);
  ensure_host_current(input);
  auto go_cpu = cpu_alias(grad_out);
  auto input_cpu = cpu_alias(input);
  if (input.dim() == 4) {
    const bool input_cl =
        input.is_contiguous(at::MemoryFormat::ChannelsLast);
    const bool grad_cl =
        grad_out.is_contiguous(at::MemoryFormat::ChannelsLast);
    if (input_cl && !grad_cl) {
      go_cpu = go_cpu.contiguous(at::MemoryFormat::ChannelsLast);
    } else if (!input_cl && grad_cl && input.is_contiguous()) {
      go_cpu = go_cpu.contiguous(at::MemoryFormat::Contiguous);
    }
  }
  auto weight_cpu = optional_cpu_alias_xdna(weight);
  auto running_mean_cpu = optional_cpu_alias_xdna(running_mean);
  auto running_var_cpu = optional_cpu_alias_xdna(running_var);
  auto save_mean_cpu = optional_cpu_alias_xdna(save_mean);
  auto save_invstd_cpu = optional_cpu_alias_xdna(save_invstd);

  at::Tensor gi;
  at::Tensor gw;
  at::Tensor gb;
  at::Tensor gi_cpu;
  at::Tensor gw_cpu;
  at::Tensor gb_cpu;

  if (output_mask[0]) {
    gi = xdna_empty_like_layout(input);
    gi_cpu = cpu_alias(gi);
  } else {
    gi_cpu = at::empty({0}, input_cpu.options());
  }
  if (output_mask[1]) {
    const int64_t channels = input.size(1);
    gw = xdna_empty_like_shape(
        {c10::SymInt(channels)}, input.scalar_type());
    gw_cpu = cpu_alias(gw);
  } else {
    gw_cpu = at::empty({0}, input_cpu.options());
  }
  if (output_mask[2]) {
    const int64_t channels = input.size(1);
    gb = xdna_empty_like_shape(
        {c10::SymInt(channels)}, input.scalar_type());
    gb_cpu = cpu_alias(gb);
  } else {
    gb_cpu = at::empty({0}, input_cpu.options());
  }

  at::native_batch_norm_backward_out(
      gi_cpu,
      gw_cpu,
      gb_cpu,
      go_cpu,
      input_cpu,
      weight_cpu,
      running_mean_cpu,
      running_var_cpu,
      save_mean_cpu,
      save_invstd_cpu,
      train,
      eps,
      output_mask);

  if (gi.defined()) mark_host_dirty(gi);
  if (gw.defined()) mark_host_dirty(gw);
  if (gb.defined()) mark_host_dirty(gb);
  return {gi, gw, gb};
}

at::Tensor cpu_mm_xdna(const at::Tensor& lhs, const at::Tensor& rhs) {
  ScopedMappedCpu mapped_phase("aten::mm.cpu_mapped");
  TORCH_CHECK(
      lhs.device().type() == kXdnaType && rhs.device().type() == kXdnaType,
      "cpu_mm_xdna expects XDNA tensors");
  TORCH_CHECK(
      lhs.dim() == 2 && rhs.dim() == 2 && lhs.sym_size(1) == rhs.sym_size(0),
      "cpu_mm_xdna expects compatible 2-D matrices");

  ensure_host_current(lhs);
  ensure_host_current(rhs);
  auto lhs_cpu = cpu_alias(lhs);
  auto rhs_cpu = cpu_alias(rhs);
  std::vector<c10::SymInt> out_size = {lhs.sym_size(0), rhs.sym_size(1)};
  auto out = empty_memory_format(
      out_size,
      lhs.scalar_type(),
      at::Layout::Strided,
      c10::Device(kXdnaType, 0),
      false,
      at::MemoryFormat::Contiguous);
  auto out_cpu = cpu_alias(out);
  at::_ops::mm_out::call(lhs_cpu, rhs_cpu, out_cpu);
  mark_host_dirty(out);
  return out;
}

at::Tensor mm_xdna(
    const at::Tensor& lhs,
    const at::Tensor& rhs) {
  TORCH_CHECK(
      lhs.device().type() == kXdnaType && rhs.device().type() == kXdnaType,
      "mm_xdna expects XDNA tensors");
  TORCH_CHECK(
      lhs.dim() == 2 && rhs.dim() == 2 && lhs.sym_size(1) == rhs.sym_size(0),
      "mm_xdna expects compatible 2-D matrices");

  auto& slot = gemm_runtime_slot();
  if (slot &&
      lhs.scalar_type() == at::ScalarType::BFloat16 &&
      rhs.scalar_type() == at::ScalarType::BFloat16 &&
      lhs.is_contiguous() &&
      rhs.is_contiguous()) {
    const int64_t m = lhs.size(0);
    const int64_t k = lhs.size(1);
    const int64_t n = rhs.size(1);
    if (auto* stream = slot->find(m, k, n)) {
      return slot->run(lhs, rhs, *stream);
    }
  }
  return cpu_mm_xdna(lhs, rhs);
}

at::Tensor xdna_from_cpu(const at::Tensor& cpu) {
  if (!cpu.defined()) {
    return at::Tensor();
  }
  auto out = empty_strided(
      cpu.sym_sizes(),
      cpu.sym_strides(),
      cpu.scalar_type(),
      at::Layout::Strided,
      c10::Device(kXdnaType, 0),
      false);
  host_storage_copy(cpu, out);
  return out;
}

// XDNA tensor holding a CPU kernel's result: zero-copy when the result was
// allocated inside an AdoptHostAllocations scope, a copy otherwise.
at::Tensor adopt_or_copy(const at::Tensor& cpu) {
  if (auto adopted = adopt_host_tensor(cpu)) return *adopted;
  return xdna_from_cpu(cpu);
}

int64_t concrete(c10::SymInt x) {
  return x.guard_int(__FILE__, __LINE__);
}

std::vector<c10::SymInt> conv_output_shape(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    bool transposed,
    c10::SymIntArrayRef output_padding,
    c10::SymInt groups) {
  TORCH_CHECK(input.dim() >= 3, "convolution input rank must be >= 3");
  const int64_t spatial = input.dim() - 2;
  TORCH_CHECK(
      stride.size() == spatial && padding.size() == spatial &&
          dilation.size() == spatial && output_padding.size() == spatial,
      "unexpected convolution parameter rank");

  std::vector<c10::SymInt> out;
  out.reserve(static_cast<size_t>(input.dim()));
  out.push_back(input.sym_size(0));
  if (!transposed) {
    out.push_back(weight.sym_size(0));
  } else {
    out.push_back(weight.sym_size(1) * groups);
  }

  for (int64_t d = 0; d < spatial; ++d) {
    auto in = input.sym_size(d + 2);
    auto kernel = weight.sym_size(d + 2);
    if (!transposed) {
      out.push_back(
          (in + padding[d] * 2 - dilation[d] * (kernel - 1) - 1) /
              stride[d] +
          1);
    } else {
      out.push_back(
          (in - 1) * stride[d] - padding[d] * 2 +
          dilation[d] * (kernel - 1) + output_padding[d] + 1);
    }
  }
  return out;
}

at::Tensor convolution_overrideable_xdna(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    bool transposed,
    c10::SymIntArrayRef output_padding,
    c10::SymInt groups) {
  if (auto& w3 = conv3x3w_runtime_slot()) {
    if (auto out = w3->try_forward(
            input, weight, bias, stride, padding, dilation, transposed,
            output_padding, groups)) {
      return *out;
    }
  }
  auto& conv_slot = conv_runtime_slot();
  if (conv_slot) {
    if (auto* stream = conv_slot->find_forward(
            input,
            weight,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias)) {
      return conv_slot->forward(input, weight, *stream);
    }
  }
  auto& full_conv_slot = fullframe_conv_runtime_slot();
  if (full_conv_slot &&
      (!bias.has_value() || !bias->defined()) &&
      stride.size() == 2 && stride[0] == 1 && stride[1] == 1 &&
      padding.size() == 2 && padding[0] == 1 && padding[1] == 1 &&
      dilation.size() == 2 && dilation[0] == 1 && dilation[1] == 1 &&
      !transposed &&
      output_padding.size() == 2 &&
      output_padding[0] == 0 && output_padding[1] == 0 &&
      groups == 1) {
    auto up_value = virtual_value_of(input);
    if (up_value &&
        up_value->kind == XdnaVirtualKind::UpsampleNearest2D &&
        up_value->sources.size() == 1) {
      const auto& cat_tensor = up_value->sources[0];
      auto cat_value = virtual_value_of(cat_tensor);
      if (cat_value &&
          cat_value->kind == XdnaVirtualKind::CatChannels4D &&
          cat_value->dim == 1 &&
          cat_value->sources.size() == 2) {
        auto* stream = full_conv_slot->find_fused_cat2_upsample2(
            cat_value->sources[0], cat_value->sources[1], weight);
        if (stream) {
          return full_conv_slot->forward_cat2_upsample2(
              cat_value->sources[0],
              cat_value->sources[1],
              weight,
              *stream);
        }
      }
    }
  }
  auto& fast_full_slot = fast_dx584_runtime_slot();
  if (fast_full_slot) {
    if (auto* stream = fast_full_slot->find_forward(
            input,
            weight,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias)) {
      return fast_full_slot->forward(input, weight, *stream);
    }
  }
  if (full_conv_slot) {
    if (auto* stream = full_conv_slot->find_forward(
            input,
            weight,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias)) {
      return full_conv_slot->forward(input, weight, *stream);
    }
  }

  ScopedMappedCpu mapped_phase("aten::convolution.cpu_mapped");
  ensure_host_current(input);
  ensure_host_current(weight);
  auto input_cpu = cpu_alias(input);
  auto weight_cpu = cpu_alias(weight);
  std::optional<at::Tensor> bias_cpu = std::nullopt;
  if (bias.has_value() && bias->defined()) {
    ensure_host_current(*bias);
    bias_cpu = cpu_alias(*bias);
  }

  // convolution has no real out= kernel (convolution_out computes into a
  // temporary and copies), so run it functionally and adopt the result.
  at::Tensor out_cpu;
  {
    AdoptHostAllocations adopt;
    out_cpu = at::_ops::convolution::call(
        input_cpu,
        weight_cpu,
        bias_cpu,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups);
  }
  return adopt_or_copy(out_cpu);
}

bool lazy_decoder_dw_enabled() {
  const char* raw = std::getenv("XDNA_LAZY_DECODER_DW");
  return raw != nullptr && raw[0] != 0 && std::strcmp(raw, "0") != 0;
}

std::optional<std::array<at::Tensor, 2>>
virtual_cat2_upsample2_sources(const at::Tensor& input) {
  auto up = virtual_value_of(input);
  if (!up || up->kind != XdnaVirtualKind::UpsampleNearest2D ||
      up->sources.size() != 1)
    return std::nullopt;
  const auto& cat_tensor = up->sources[0];
  auto cat = virtual_value_of(cat_tensor);
  if (!cat || cat->kind != XdnaVirtualKind::CatChannels4D ||
      cat->dim != 1 || cat->sources.size() != 2)
    return std::nullopt;
  return std::array<at::Tensor, 2>{cat->sources[0], cat->sources[1]};
}

at::Tensor virtual_cat2_upsample2_weight_grad_cpu(
    const at::Tensor& src0,
    const at::Tensor& src1,
    const at::Tensor& grad_output,
    at::IntArrayRef weight_shape) {
  TORCH_CHECK(
      src0.device().is_cpu() && src1.device().is_cpu() &&
          grad_output.device().is_cpu(),
      "virtual decoder dW expects CPU aliases");
  TORCH_CHECK(
      src0.scalar_type() == at::ScalarType::BFloat16 &&
          src1.scalar_type() == at::ScalarType::BFloat16 &&
          grad_output.scalar_type() == at::ScalarType::BFloat16,
      "virtual decoder dW expects BF16");
  TORCH_CHECK(
      src0.dim() == 4 && src1.dim() == 4 && grad_output.dim() == 4 &&
          src0.size(0) == src1.size(0) &&
          src0.size(2) == src1.size(2) &&
          src0.size(3) == src1.size(3),
      "virtual decoder dW source shape mismatch");
  TORCH_CHECK(
      weight_shape.size() == 4 && weight_shape[2] == 3 &&
          weight_shape[3] == 3,
      "virtual decoder dW currently supports 3x3 weights");

  const int64_t batch = src0.size(0);
  const int64_t h = src0.size(2);
  const int64_t w = src0.size(3);
  const int64_t cout = grad_output.size(1);
  const int64_t oh = grad_output.size(2);
  const int64_t ow = grad_output.size(3);
  TORCH_CHECK(
      oh == h * 2 && ow == w * 2,
      "virtual decoder dW requires exact nearest 2x geometry");
  TORCH_CHECK(
      weight_shape[0] == cout &&
          weight_shape[1] == src0.size(1) + src1.size(1),
      "virtual decoder dW weight/channel mismatch");

  auto result = at::empty(weight_shape, grad_output.options());
  auto dy_hwc = grad_output.permute({0, 2, 3, 1});
  const int64_t reduce = batch * h * w;
  std::array<at::Tensor, 2> xflat = {
      src0.permute({0, 2, 3, 1}).reshape({reduce, src0.size(1)}),
      src1.permute({0, 2, 3, 1}).reshape({reduce, src1.size(1)}),
  };
  const std::array<int64_t, 3> offsets = {
      0, src0.size(1), src0.size(1) + src1.size(1)};

  const auto ceil_div2 = [](int64_t value) {
    return value >= 0 ? (value + 1) / 2 : value / 2;
  };

  for (int64_t ky = 0; ky < 3; ++ky) {
    for (int64_t kx = 0; kx < 3; ++kx) {
      auto reduced = at::zeros(
          {batch, h, w, cout}, grad_output.options());

      for (int64_t ry = 0; ry < 2; ++ry) {
        const int64_t yoff = ry - ky + 1;
        const int64_t sy0 = std::max<int64_t>(0, ceil_div2(-yoff));
        const int64_t sy1 = std::min<int64_t>(
            h, (oh - 1 - yoff) / 2 + 1);
        if (sy1 <= sy0)
          continue;
        const int64_t y0 = 2 * sy0 + yoff;
        const int64_t y1 = 2 * sy1 + yoff;

        for (int64_t rx = 0; rx < 2; ++rx) {
          const int64_t xoff = rx - kx + 1;
          const int64_t sx0 = std::max<int64_t>(0, ceil_div2(-xoff));
          const int64_t sx1 = std::min<int64_t>(
              w, (ow - 1 - xoff) / 2 + 1);
          if (sx1 <= sx0)
            continue;
          const int64_t x0 = 2 * sx0 + xoff;
          const int64_t x1 = 2 * sx1 + xoff;

          reduced
              .slice(1, sy0, sy1)
              .slice(2, sx0, sx1)
              .add_(
                  dy_hwc
                      .slice(1, y0, y1, 2)
                      .slice(2, x0, x1, 2));
        }
      }

      auto gflat = reduced.reshape({reduce, cout});
      for (size_t part = 0; part < xflat.size(); ++part) {
        auto grad_part =
            at::mm(xflat[part].transpose(0, 1), gflat)
                .transpose(0, 1);
        result
            .slice(1, offsets[part], offsets[part + 1])
            .select(2, ky)
            .select(2, kx)
            .copy_(grad_part);
      }
    }
  }
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
convolution_backward_overrideable_xdna(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    bool transposed,
    c10::SymIntArrayRef output_padding,
    c10::SymInt groups,
    std::array<bool, 3> output_mask) {
  at::Tensor gi;
  at::Tensor gw;
  at::Tensor gb;

  // Resolve accelerator candidates before executing either branch. dX and dW
  // are independent once X, W and dY exist, so a CPU dW can overlap a resident
  // NPU dX instead of serializing the heterogeneous engines.
  Conv3x3wStream* w3_dx = nullptr;
  std::optional<Conv3x3wPlan> w3_dx_plan;
  XdnaConvStream* small_dx = nullptr;
  XdnaFullFrameConvStream* full_dx = nullptr;
  XdnaFullFrameConvRuntime* full_dx_owner = nullptr;
  auto& w3_slot = conv3x3w_runtime_slot();
  if (output_mask[0] && w3_slot) {
    w3_dx_plan = w3_slot->plan(
        grad_output, weight, true, stride, padding, dilation, transposed,
        output_padding, groups);
    if (w3_dx_plan) w3_dx = w3_slot->stream(*w3_dx_plan);
  }
  auto& conv_slot = conv_runtime_slot();
  auto& full_conv_slot = fullframe_conv_runtime_slot();
  auto& fast_dx584_slot = fast_dx584_runtime_slot();
  auto& fast_conv11_slot = fast_conv11_runtime_slot();
  if (output_mask[0] && !w3_dx && conv_slot) {
    small_dx = conv_slot->find_dx(
        grad_output,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups);
  }
  if (output_mask[0] && !small_dx && fast_conv11_slot) {
    full_dx = fast_conv11_slot->find_dx(
        grad_output, weight, stride, padding, dilation,
        transposed, output_padding, groups);
    if (full_dx) full_dx_owner = fast_conv11_slot.get();
  }
  if (output_mask[0] && !small_dx && !full_dx && fast_dx584_slot) {
    full_dx = fast_dx584_slot->find_dx(
        grad_output, weight, stride, padding, dilation,
        transposed, output_padding, groups);
    if (full_dx) full_dx_owner = fast_dx584_slot.get();
  }
  if (output_mask[0] && !small_dx && !full_dx && full_conv_slot) {
    full_dx = full_conv_slot->find_dx(
        grad_output, weight, stride, padding, dilation,
        transposed, output_padding, groups);
    if (full_dx) full_dx_owner = full_conv_slot.get();
  }
  const bool have_npu_dx =
      w3_dx != nullptr || small_dx != nullptr || full_dx != nullptr;

  auto& full_dw_slot = fullframe_dw_runtime_slot();
  const bool have_npu_dw =
      output_mask[1] && full_dw_slot &&
      full_dw_slot->supports(
          input,
          grad_output,
          weight,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups);

  // Large 3x3 dWs go to the NPU, deferred (see XdnaConv3x3wDwRuntime); small
  // ones stay on the CPU, overlapped with the NPU dX below.
  std::optional<Conv3x3wDwPlan> w3_dw_plan;
  std::shared_ptr<Conv3x3wPacked> w3_dy_packed;
  std::shared_ptr<Conv3x3wDwJob> w3_dw_job;
  auto& w3_dw_slot = conv3x3w_dw_runtime_slot();
  if (output_mask[1] && w3_dw_slot && !have_npu_dw) {
    w3_dw_plan = w3_dw_slot->plan(
        input, grad_output, weight, stride, padding, dilation, transposed,
        output_padding, groups);
    if (w3_dw_plan && !w3_dw_slot->ready(*w3_dw_plan)) w3_dw_plan.reset();
  }

  auto cpu_mask = output_mask;
  std::future<at::Tensor> overlapped_dw;
  bool has_overlapped_dw = false;
  const bool overlap_cpu_dw = []() {
    const char* raw = std::getenv("XDNA_DISABLE_CPU_DW_OVERLAP");
    return !(raw != nullptr && raw[0] != 0 && std::strcmp(raw, "0") != 0);
  }();
  const bool defer_cpu_dw = []() {
    const char* raw = std::getenv("XDNA_DEFER_CPU_DW");
    return raw != nullptr && raw[0] != 0 && std::strcmp(raw, "0") != 0;
  }();

  std::optional<std::array<at::Tensor, 2>> virtual_dw_sources;
  if (lazy_decoder_dw_enabled() && !defer_cpu_dw &&
      have_npu_dx && output_mask[1] && !have_npu_dw &&
      stride.size() == 2 && stride[0] == 1 && stride[1] == 1 &&
      padding.size() == 2 && padding[0] == 1 && padding[1] == 1 &&
      dilation.size() == 2 && dilation[0] == 1 && dilation[1] == 1 &&
      !transposed &&
      output_padding.size() == 2 &&
      output_padding[0] == 0 && output_padding[1] == 0 &&
      groups == 1 &&
      weight.dim() == 4 && weight.size(2) == 3 && weight.size(3) == 3) {
    virtual_dw_sources = virtual_cat2_upsample2_sources(input);
  }

  // General heterogeneous scheduling path: while NPU computes the
  // dependency-critical dX, let oneDNN compute the independent dW on CPU from
  // zero-copy aliases of the same host-visible XRT BOs. The weight gradient is
  // tiny compared with activations, so copying the finished CPU dW into XDNA
  // storage after joining is negligible and keeps the implementation simple.
  if (w3_dw_plan) {
    gw = xdna_empty_like_shape(weight.sym_sizes(), weight.scalar_type());
    // The job must not hold gw itself (gw's allocation owns this future: a
    // cycle); gw's deleter also waits on it.  d_buf is filled in by the dX
    // run below.
    auto job = std::make_shared<Conv3x3wDwJob>();
    job->input = input;
    job->grad_output = grad_output;
    job->x_buf = take_packed(input, w3_dw_plan->chunks);
    w3_dw_job = job;
    auto* gw_ptr = static_cast<uint16_t*>(cpu_alias(gw).data_ptr());
    auto* rt = w3_dw_slot.get();
    std::shared_future<void> pending =
        std::async(std::launch::deferred, [job, gw_ptr, rt, plan = *w3_dw_plan]() {
          rt->compute(job->input, job->grad_output, gw_ptr, plan,
                      std::move(job->x_buf), std::move(job->d_buf));
          *job = Conv3x3wDwJob();
        }).share();
    set_pending_host_write(gw, std::move(pending));
    cpu_mask[1] = false;
  } else if (overlap_cpu_dw && have_npu_dx && output_mask[1] && !have_npu_dw) {
    ensure_host_current(grad_output);
    ensure_host_current(weight);
    auto go_cpu = cpu_alias(grad_output);
    at::Tensor input_cpu;
    if (!virtual_dw_sources.has_value()) {
      ensure_host_current(input);
      input_cpu = cpu_alias(input);
      if (input.dim() == 4) {
        const bool input_cl =
            input.is_contiguous(at::MemoryFormat::ChannelsLast);
        const bool grad_cl =
            grad_output.is_contiguous(at::MemoryFormat::ChannelsLast);
        if (input_cl && !grad_cl) {
          go_cpu = go_cpu.contiguous(at::MemoryFormat::ChannelsLast);
        } else if (!input_cl && grad_cl && input.is_contiguous()) {
          go_cpu = go_cpu.contiguous(at::MemoryFormat::Contiguous);
        }
      }
    }
    auto input_cpu_keep = input_cpu;
    auto go_cpu_keep = go_cpu;
    auto weight_cpu_keep = cpu_alias(weight);
    std::vector<c10::SymInt> stride_v(stride.begin(), stride.end());
    std::vector<c10::SymInt> padding_v(padding.begin(), padding.end());
    std::vector<c10::SymInt> dilation_v(dilation.begin(), dilation.end());
    std::vector<c10::SymInt> output_padding_v(
        output_padding.begin(), output_padding.end());
    const c10::SymInt groups_v = groups;

    if (defer_cpu_dw) {
      // Return the gradient storage immediately.  A single bounded worker
      // computes CPU dWs in dependency-independent order while autograd keeps
      // advancing the NPU dX chain.  Any actual consumer of gw fences through
      // ensure_host_current()/ensure_device_current().
      gw = xdna_empty_like_shape(weight.sym_sizes(), weight.scalar_type());
      auto gw_cpu_keep = cpu_alias(gw);
      // CPU aliases are intentionally non-owning. Retain the XDNA tensors
      // themselves so their BOs cannot return to the cache while queued dW
      // still reads/writes the mapped addresses.
      auto go_xdna_keep = grad_output;
      auto input_xdna_keep = input;
      auto weight_xdna_keep = weight;
      auto gw_xdna_keep = gw;
      auto pending = cpu_dw_queue().submit(
          [go_cpu_keep,
           input_cpu_keep,
           weight_cpu_keep,
           gw_cpu_keep,
           go_xdna_keep,
           input_xdna_keep,
           weight_xdna_keep,
           gw_xdna_keep,
           stride_v = std::move(stride_v),
           padding_v = std::move(padding_v),
           dilation_v = std::move(dilation_v),
           transposed,
           output_padding_v = std::move(output_padding_v),
           groups_v]() mutable {
            const std::array<bool, 3> dw_mask = {false, true, false};
            auto grads = at::_ops::convolution_backward::call(
                go_cpu_keep,
                input_cpu_keep,
                weight_cpu_keep,
                std::nullopt,
                c10::SymIntArrayRef(stride_v),
                c10::SymIntArrayRef(padding_v),
                c10::SymIntArrayRef(dilation_v),
                transposed,
                c10::SymIntArrayRef(output_padding_v),
                groups_v,
                dw_mask);
            gw_cpu_keep.copy_(std::get<1>(grads), false);
          });
      set_pending_host_write(gw, std::move(pending));
    } else if (virtual_dw_sources.has_value()) {
      const auto src0_xdna = (*virtual_dw_sources)[0];
      const auto src1_xdna = (*virtual_dw_sources)[1];
      ensure_host_current(src0_xdna);
      ensure_host_current(src1_xdna);
      auto src0_cpu = cpu_alias(src0_xdna);
      auto src1_cpu = cpu_alias(src1_xdna);
      std::vector<int64_t> weight_shape(
          weight.sizes().begin(), weight.sizes().end());
      overlapped_dw = std::async(
          std::launch::async,
          [src0_cpu,
           src1_cpu,
           go_cpu_keep,
           src0_xdna,
           src1_xdna,
           weight_shape = std::move(weight_shape)]() mutable {
            sync_host_threads();
            return virtual_cat2_upsample2_weight_grad_cpu(
                src0_cpu, src1_cpu, go_cpu_keep, weight_shape);
          });
      has_overlapped_dw = true;
    } else {
      overlapped_dw = std::async(
          std::launch::async,
          [go_cpu_keep,
           input_cpu_keep,
           weight_cpu_keep,
           stride_v = std::move(stride_v),
           padding_v = std::move(padding_v),
           dilation_v = std::move(dilation_v),
           transposed,
           output_padding_v = std::move(output_padding_v),
           groups_v]() mutable {
            sync_host_threads();
            const std::array<bool, 3> dw_mask = {false, true, false};

            // The CPU and NPU share the same LPDDR/fabric while dW and dX
            // overlap. For the validated batch-4 training path, reducing dW
            // one sample at a time gives oneDNN a smaller working set and
            // lowers instantaneous memory pressure enough to improve both the
            // CPU dW and the concurrent NPU dX critical region. Keep this
            // tunable: 0 disables, positive values select a batch chunk.
            int64_t batch_chunk =
                input_cpu_keep.dim() == 4 && input_cpu_keep.size(0) == 4 ? 1 : 0;
            if (const char* raw = std::getenv("XDNA_CPU_DW_BATCH_CHUNK")) {
              const long parsed = std::strtol(raw, nullptr, 10);
              batch_chunk = parsed > 0 ? parsed : 0;
            }
            if (batch_chunk > 0 &&
                input_cpu_keep.dim() == 4 &&
                go_cpu_keep.dim() == 4 &&
                input_cpu_keep.size(0) == go_cpu_keep.size(0) &&
                input_cpu_keep.size(0) > batch_chunk) {
              auto accum = at::zeros(
                  weight_cpu_keep.sizes(),
                  weight_cpu_keep.options().dtype(at::ScalarType::Float));
              const int64_t batch = input_cpu_keep.size(0);
              for (int64_t b0 = 0; b0 < batch; b0 += batch_chunk) {
                const int64_t b1 = std::min<int64_t>(batch, b0 + batch_chunk);
                auto grads = at::_ops::convolution_backward::call(
                    go_cpu_keep.slice(0, b0, b1),
                    input_cpu_keep.slice(0, b0, b1),
                    weight_cpu_keep,
                    std::nullopt,
                    c10::SymIntArrayRef(stride_v),
                    c10::SymIntArrayRef(padding_v),
                    c10::SymIntArrayRef(dilation_v),
                    transposed,
                    c10::SymIntArrayRef(output_padding_v),
                    groups_v,
                    dw_mask);
                accum.add_(std::get<1>(grads).to(at::ScalarType::Float));
              }
              return accum.to(weight_cpu_keep.scalar_type());
            }

            auto grads = at::_ops::convolution_backward::call(
                go_cpu_keep,
                input_cpu_keep,
                weight_cpu_keep,
                std::nullopt,
                c10::SymIntArrayRef(stride_v),
                c10::SymIntArrayRef(padding_v),
                c10::SymIntArrayRef(dilation_v),
                transposed,
                c10::SymIntArrayRef(output_padding_v),
                groups_v,
                dw_mask);
            return std::get<1>(grads);
          });
      has_overlapped_dw = true;
    }
    cpu_mask[1] = false;
  }

  if (w3_dx) {
    gi = w3_slot->input_grad(grad_output, weight, *w3_dx_plan, *w3_dx,
                             w3_dw_job ? &w3_dy_packed : nullptr);
    // The dX packed dY exactly as dW needs it (same chunks: dX's input
    // channels are dW's output channels).
    if (w3_dw_job) w3_dw_job->d_buf = std::move(w3_dy_packed);
    cpu_mask[0] = false;
  } else if (small_dx) {
    gi = conv_slot->input_grad(grad_output, weight, *small_dx);
    cpu_mask[0] = false;
  } else if (full_dx) {
    TORCH_CHECK(full_dx_owner != nullptr, "full-frame dX runtime owner missing");
    gi = full_dx_owner->input_grad(grad_output, weight, *full_dx);
    cpu_mask[0] = false;
  }

  if (have_npu_dw) {
    gw = full_dw_slot->weight_grad(input, grad_output, weight);
    cpu_mask[1] = false;
  }

  if (cpu_mask[0] || cpu_mask[1] || cpu_mask[2]) {
    ScopedMappedCpu mapped_phase("aten::convolution_backward.cpu_mapped");
    ensure_host_current(grad_output);
    ensure_host_current(input);
    ensure_host_current(weight);
    auto go_cpu = cpu_alias(grad_output);
    auto input_cpu = cpu_alias(input);
    if (input.dim() == 4) {
      const bool input_cl =
          input.is_contiguous(at::MemoryFormat::ChannelsLast);
      const bool grad_cl =
          grad_output.is_contiguous(at::MemoryFormat::ChannelsLast);
      if (input_cl && !grad_cl) {
        go_cpu = go_cpu.contiguous(at::MemoryFormat::ChannelsLast);
      } else if (!input_cl && grad_cl && input.is_contiguous()) {
        go_cpu = go_cpu.contiguous(at::MemoryFormat::Contiguous);
      }
    }
    auto weight_cpu = cpu_alias(weight);

    std::vector<c10::SymInt> bias_size_vec;
    at::OptionalSymIntArrayRef bias_sizes = std::nullopt;
    if (cpu_mask[2]) {
      bias_size_vec.emplace_back(
          transposed ? weight.sym_size(1) * groups : weight.sym_size(0));
      bias_sizes = c10::SymIntArrayRef(bias_size_vec);
    }

    // convolution_backward has no real out= kernel, so compute exactly the
    // requested gradients functionally and adopt them (no dBias for
    // bias-free convs, no temporary-to-output copies).
    std::tuple<at::Tensor, at::Tensor, at::Tensor> grads;
    {
      AdoptHostAllocations adopt;
      grads = at::_ops::convolution_backward::call(
          go_cpu,
          input_cpu,
          weight_cpu,
          bias_sizes,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
          cpu_mask);
    }
    if (cpu_mask[0]) gi = adopt_or_copy(std::get<0>(grads));
    if (cpu_mask[1]) gw = adopt_or_copy(std::get<1>(grads));
    if (cpu_mask[2]) gb = adopt_or_copy(std::get<2>(grads));
  }

  if (has_overlapped_dw) {
    gw = xdna_from_cpu(overlapped_dw.get());
  }

  return {gi, gw, gb};
}


// Fused RNN cells.  ATen's LSTM/GRU send every non-CPU device
// (is_privateuseone()) to these CUDA-only fused cells, which have no CPU
// kernel to fall back to.  Same math and workspace layout as the CUDA kernels
// (activated gates for LSTM; r, z, n, hx, hn for GRU), computed in FP32.
at::Tensor xdna_host_f32(const at::Tensor& t) {
  ensure_host_current(t);
  return cpu_alias(t).to(at::kFloat);
}

at::Tensor xdna_result(const at::Tensor& f32, at::ScalarType dtype) {
  at::Tensor cpu;
  {
    AdoptHostAllocations adopt;
    cpu = f32.to(dtype).contiguous();
  }
  return adopt_or_copy(cpu);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_lstm_cell_xdna(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& cx,
    const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
  ScopedMappedCpu mapped_phase("aten::_thnn_fused_lstm_cell");
  const auto dtype = input_gates.scalar_type();
  auto gates = xdna_host_f32(input_gates) + xdna_host_f32(hidden_gates);
  if (input_bias.has_value() && input_bias->defined()) {
    gates = gates + xdna_host_f32(*input_bias) + xdna_host_f32(*hidden_bias);
  }
  auto chunks = gates.chunk(4, 1);
  auto i = chunks[0].sigmoid();
  auto f = chunks[1].sigmoid();
  auto g = chunks[2].tanh();
  auto o = chunks[3].sigmoid();
  auto cy = f * xdna_host_f32(cx) + i * g;
  auto hy = o * cy.tanh();
  auto workspace = at::cat({i, f, g, o}, 1);
  return {xdna_result(hy, dtype), xdna_result(cy, dtype), xdna_result(workspace, dtype)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_lstm_cell_backward_xdna(
    const std::optional<at::Tensor>& grad_hy,
    const std::optional<at::Tensor>& grad_cy,
    const at::Tensor& cx,
    const at::Tensor& cy,
    const at::Tensor& workspace,
    bool has_bias) {
  ScopedMappedCpu mapped_phase("aten::_thnn_fused_lstm_cell_backward_impl");
  const auto dtype = workspace.scalar_type();
  auto ws = xdna_host_f32(workspace).chunk(4, 1);
  const auto& i = ws[0];
  const auto& f = ws[1];
  const auto& g = ws[2];
  const auto& o = ws[3];
  auto cy_f = xdna_host_f32(cy);
  auto tanh_cy = cy_f.tanh();
  const bool has_ghy = grad_hy.has_value() && grad_hy->defined();
  const bool has_gcy = grad_cy.has_value() && grad_cy->defined();
  auto ghy = has_ghy ? xdna_host_f32(*grad_hy) : at::zeros_like(cy_f);
  auto gcy = has_gcy ? xdna_host_f32(*grad_cy) : at::zeros_like(cy_f);
  auto gc = gcy + ghy * o * (1 - tanh_cy * tanh_cy);
  auto gi = gc * g * i * (1 - i);
  auto gf = gc * xdna_host_f32(cx) * f * (1 - f);
  auto gg = gc * i * (1 - g * g);
  auto go = ghy * tanh_cy * o * (1 - o);
  auto grad_gates = at::cat({gi, gf, gg, go}, 1);
  auto grad_cx = gc * f;
  at::Tensor grad_bias;
  if (has_bias) grad_bias = xdna_result(grad_gates.sum(0), dtype);
  return {xdna_result(grad_gates, dtype), xdna_result(grad_cx, dtype), grad_bias};
}

std::tuple<at::Tensor, at::Tensor> fused_gru_cell_xdna(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
  ScopedMappedCpu mapped_phase("aten::_thnn_fused_gru_cell");
  const auto dtype = input_gates.scalar_type();
  auto ig = xdna_host_f32(input_gates);
  auto hg = xdna_host_f32(hidden_gates);
  if (input_bias.has_value() && input_bias->defined()) {
    ig = ig + xdna_host_f32(*input_bias);
    hg = hg + xdna_host_f32(*hidden_bias);
  }
  auto ic = ig.chunk(3, 1);
  auto hc = hg.chunk(3, 1);
  auto r = (ic[0] + hc[0]).sigmoid();
  auto z = (ic[1] + hc[1]).sigmoid();
  auto n = (ic[2] + r * hc[2]).tanh();
  auto hx_f = xdna_host_f32(hx);
  auto hy = n + z * (hx_f - n);
  auto workspace = at::cat({r, z, n, hx_f, hc[2]}, 1);
  return {xdna_result(hy, dtype), xdna_result(workspace, dtype)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_gru_cell_backward_xdna(
    const at::Tensor& grad_hy,
    const at::Tensor& workspace,
    bool has_bias) {
  ScopedMappedCpu mapped_phase("aten::_thnn_fused_gru_cell_backward");
  const auto dtype = workspace.scalar_type();
  auto ws = xdna_host_f32(workspace).chunk(5, 1);
  const auto& r = ws[0];
  const auto& z = ws[1];
  const auto& n = ws[2];
  const auto& hx = ws[3];
  const auto& hn = ws[4];
  auto ghy = xdna_host_f32(grad_hy);
  auto gz = ghy * (hx - n) * z * (1 - z);
  auto ghx = ghy * z;
  auto gn = ghy * (1 - z) * (1 - n * n);
  auto ghn = gn * r;
  auto gr = gn * hn * r * (1 - r);
  auto grad_input_gates = at::cat({gr, gz, gn}, 1);
  auto grad_hidden_gates = at::cat({gr, gz, ghn}, 1);
  at::Tensor grad_input_bias, grad_hidden_bias;
  if (has_bias) {
    grad_input_bias = xdna_result(grad_input_gates.sum(0), dtype);
    grad_hidden_bias = xdna_result(grad_hidden_gates.sum(0), dtype);
  }
  return {xdna_result(grad_input_gates, dtype),
          xdna_result(grad_hidden_gates, dtype),
          xdna_result(ghx, dtype),
          grad_input_bias,
          grad_hidden_bias};
}

} // namespace

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_xdna_allocator)
C10_REGISTER_GUARD_IMPL(PrivateUse1, XdnaGuardImpl)

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
  m.impl("empty_strided", TORCH_FN(empty_strided));
  m.impl("as_strided", TORCH_FN(as_strided_xdna));
  m.impl("_reshape_alias", TORCH_FN(reshape_alias_xdna));
  m.impl("view", TORCH_FN(view_xdna));
  m.impl("resize_", TORCH_FN(resize_xdna_));
  m.impl("mm", TORCH_FN(mm_xdna));
  m.impl("leaky_relu", TORCH_FN(leaky_relu_xdna));
  m.impl("leaky_relu_backward", TORCH_FN(leaky_relu_backward_xdna));
  m.impl("add.Tensor", TORCH_FN(add_tensor_xdna));
  m.impl("add.out", TORCH_FN(add_out_xdna));
  m.impl("mul.out", TORCH_FN(mul_out_xdna));
  m.impl("div.out", TORCH_FN(div_out_xdna));
  m.impl("sqrt.out", TORCH_FN(sqrt_out_xdna));
  m.impl("addcmul.out", TORCH_FN(addcmul_out_xdna));
  m.impl("addcdiv.out", TORCH_FN(addcdiv_out_xdna));
  m.impl("lerp.Scalar_out", TORCH_FN(lerp_scalar_out_xdna));
  m.impl("sigmoid.out", TORCH_FN(sigmoid_out_xdna));
  m.impl("sub.out", TORCH_FN(sub_out_xdna));
  m.impl("addmm.out", TORCH_FN(addmm_out_xdna));
  m.impl("bmm.out", TORCH_FN(bmm_out_xdna));
  m.impl("cat", TORCH_FN(cat_xdna));
  m.impl("upsample_nearest2d", TORCH_FN(upsample_nearest2d_xdna));
  m.impl("native_batch_norm", TORCH_FN(native_batch_norm_xdna));
  m.impl(
      "native_batch_norm_backward",
      TORCH_FN(native_batch_norm_backward_xdna));
  m.impl(
      "upsample_nearest2d_backward",
      TORCH_FN(upsample_nearest2d_backward_xdna));
  m.impl("_copy_from", TORCH_FN(copy_from));
  m.impl("_copy_from_and_resize", TORCH_FN(copy_from_and_resize));
  m.impl("convolution_overrideable", TORCH_FN(convolution_overrideable_xdna));
  m.impl(
      "convolution_backward_overrideable",
      TORCH_FN(convolution_backward_overrideable_xdna));
  m.impl("_thnn_fused_lstm_cell", TORCH_FN(fused_lstm_cell_xdna));
  m.impl(
      "_thnn_fused_lstm_cell_backward_impl",
      TORCH_FN(fused_lstm_cell_backward_xdna));
  m.impl("_thnn_fused_gru_cell", TORCH_FN(fused_gru_cell_xdna));
  m.impl("_thnn_fused_gru_cell_backward", TORCH_FN(fused_gru_cell_backward_xdna));
}

struct FallbackStat {
  uint64_t calls = 0;
  uint64_t total_ns = 0;
};

std::mutex& fallback_stats_mutex() {
  static auto* value = new std::mutex();
  return *value;
}

std::unordered_map<std::string, FallbackStat>& fallback_stats_map() {
  static auto* value =
      new std::unordered_map<std::string, FallbackStat>();
  return *value;
}

bool profile_fallbacks() {
  static const bool enabled = []() {
    const char* raw = std::getenv("XDNA_PROFILE_FALLBACK");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

bool schema_type_has_tensor(const c10::TypePtr& type) {
  if (type->kind() == c10::TypeKind::TensorType) return true;
  for (const auto& contained : type->containedTypes()) {
    if (schema_type_has_tensor(contained)) return true;
  }
  return false;
}

// XDNA storage is host-mapped XRT memory, so an operator without an XDNA
// kernel can run its CPU kernel directly on CPU aliases of the XDNA
// arguments.  at::native::cpu_fallback instead copies every input to CPU and
// every output back (about half of the VGG-loss time in NIS style
// transfer).  Returns false, with the stack untouched, for anything this
// cannot express: view returns, device/factory arguments, non-Tensor
// containers of tensors.
bool zero_copy_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto& args = schema.arguments();
  const auto& rets = schema.returns();
  for (const auto& ret : rets) {
    if (ret.alias_info() && !ret.alias_info()->isWrite()) return false;
    if (schema_type_has_tensor(ret.type()) &&
        ret.type()->kind() != c10::TypeKind::TensorType) {
      return false;
    }
  }

  const size_t n = args.size();
  TORCH_INTERNAL_ASSERT(stack->size() >= n);
  const size_t base = stack->size() - n;
  bool any_xdna = false;
  for (size_t i = 0; i < n; ++i) {
    const auto& iv = (*stack)[base + i];
    if (iv.isDevice()) return false;
    if (iv.isTensor()) {
      const auto& t = iv.toTensor();
      if (!t.defined() || t.device().type() != kXdnaType) continue;
      any_xdna = true;
      const auto* alias = args[i].alias_info();
      // A CPU kernel cannot resize a non-owning alias; let the copying
      // fallback handle out= tensors that still need their real shape.
      if (alias && alias->isWrite() && t.numel() == 0) return false;
    } else if (iv.isTensorList()) {
      for (const at::Tensor t : iv.toTensorList()) {
        if (t.defined() && t.device().type() == kXdnaType) any_xdna = true;
      }
    } else if (iv.isList()) {
      for (const auto& e : iv.toListRef()) {
        if (e.isTensor()) return false;
      }
    }
  }
  if (!any_xdna) return false;

  // Keep the original arguments: they own the XRT storage behind the
  // non-owning aliases, and they are restored if the CPU kernel throws.
  std::vector<c10::IValue> originals(stack->begin() + base, stack->end());
  std::vector<std::pair<at::Tensor, at::Tensor>> written;  // {cpu, xdna}
  for (size_t i = 0; i < n; ++i) {
    auto& iv = (*stack)[base + i];
    const auto* alias = args[i].alias_info();
    const bool is_write = alias && alias->isWrite();
    if (iv.isTensor()) {
      const auto t = iv.toTensor();
      if (!t.defined() || t.device().type() != kXdnaType) continue;
      auto cpu = cpu_alias(t);
      if (is_write) written.emplace_back(cpu, t);
      iv = c10::IValue(cpu);
    } else if (iv.isTensorList()) {
      auto list = iv.toTensorList();
      c10::List<at::Tensor> cpu_list;
      cpu_list.reserve(list.size());
      for (const at::Tensor t : list) {
        if (t.defined() && t.device().type() == kXdnaType) {
          auto cpu = cpu_alias(t);
          if (is_write) written.emplace_back(cpu, t);
          cpu_list.push_back(cpu);
        } else {
          cpu_list.push_back(t);
        }
      }
      iv = c10::IValue(cpu_list);
    }
  }

  try {
    AdoptHostAllocations adopt;
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
  } catch (const c10::Error&) {
    stack->resize(base);
    stack->insert(stack->end(), originals.begin(), originals.end());
    return false;
  }

  for (const auto& [cpu, xdna] : written) mark_host_dirty(xdna);

  const size_t r = rets.size();
  TORCH_INTERNAL_ASSERT(stack->size() >= r);
  for (size_t j = 0; j < r; ++j) {
    auto& iv = (*stack)[stack->size() - r + j];
    if (!iv.isTensor()) continue;
    const auto t = iv.toTensor();
    if (!t.defined() || t.device().type() != c10::DeviceType::CPU) continue;
    bool mapped = false;
    for (const auto& [cpu, xdna] : written) {
      if (t.is_same(cpu)) {
        iv = c10::IValue(xdna);
        mapped = true;
        break;
      }
    }
    if (mapped) continue;
    if (auto adopted = adopt_host_tensor(t)) {
      iv = c10::IValue(*adopted);
      continue;
    }
    const bool dense = t.is_non_overlapping_and_dense();
    auto out = empty_strided(
        t.sym_sizes(),
        dense ? t.sym_strides()
              : c10::SymIntArrayRef(c10::fromIntArrayRefSlow(
                    c10::contiguous_strides(t.sizes()))),
        t.scalar_type(),
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false);
    cpu_alias_unchecked(out).copy_(t);
    mark_host_dirty(out);
    iv = c10::IValue(out);
  }
  return true;
}

void run_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  static const bool zero_copy = []() {
    const char* raw = std::getenv("XDNA_ZERO_COPY_FALLBACK");
    return raw == nullptr || std::strcmp(raw, "0") != 0;
  }();
  if (zero_copy && zero_copy_cpu_fallback(op, stack)) return;
  at::native::cpu_fallback(op, stack, false, c10::DispatchKey::CPU);
}

void xdna_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  sync_host_threads();
  if (!profile_fallbacks()) {
    run_cpu_fallback(op, stack);
    return;
  }
  std::string name = op.schema().name();
  if (!op.schema().overload_name().empty()) {
    name += ".";
    name += op.schema().overload_name();
  }
  const auto t0 = std::chrono::steady_clock::now();
  run_cpu_fallback(op, stack);
  const auto t1 = std::chrono::steady_clock::now();
  const uint64_t ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  std::lock_guard<std::mutex> guard(fallback_stats_mutex());
  auto& stat = fallback_stats_map()[name];
  ++stat.calls;
  stat.total_ns += ns;
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&xdna_cpu_fallback>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_host_threads();
  if (const char* raw = std::getenv("XDNA_ADOPT_HOST_OUTPUTS");
      raw == nullptr || std::strcmp(raw, "0") != 0) {
    install_host_allocator();
  }
  c10::SetStorageImplCreate(
      c10::DeviceType::PrivateUse1, &make_xdna_storage_impl);
  if (!at::isPrivateUse1HooksRegistered()) {
    at::RegisterPrivateUse1HooksInterface(&g_xdna_hooks);
  }
  m.def("device_count", []() { return 1; });
  m.def("current_device", []() { return static_cast<int>(g_current_device); });
  m.def("set_device", [](int index) {
    TORCH_CHECK(index == 0, "only xdna:0 is currently supported");
    g_current_device = 0;
  });
  m.def("synchronize", []() { cpu_dw_queue().drain(); });
  m.def("is_available", []() { return true; });
  m.def("alignment", []() { return kPageAlignment; });
  m.def("cpu_mm", &cpu_mm_xdna);
  m.def(
      "configure_gemm_family",
      [](const std::string& xclbin,
         const std::string& fwd,
         const std::string& dx,
         const std::string& dw) {
        auto runtime = std::make_unique<XdnaGemmRuntime>();
        runtime->open(xclbin, {fwd, dx, dw});
        gemm_runtime_slot() = std::move(runtime);
      });
  m.def("gemm_family_configured", []() {
    return static_cast<bool>(gemm_runtime_slot());
  });
  m.def(
      "configure_conv_family",
      [](const std::string& xclbin,
         const std::string& fwd18,
         const std::string& dx18,
         const std::string& fwd36,
         const std::string& dx36,
         int64_t m_sched18,
         int64_t m_sched36) {
        auto runtime = std::make_unique<XdnaConvRuntime>();
        runtime->open(xclbin, fwd18, dx18, fwd36, dx36, m_sched18, m_sched36);
        conv_runtime_slot() = std::move(runtime);
      },
      py::arg("xclbin"), py::arg("fwd18"), py::arg("dx18"), py::arg("fwd36"),
      py::arg("dx36"), py::arg("m_sched18") = 12960, py::arg("m_sched36") = 51840);
  m.def(
      "configure_conv3x3w",
      [](const std::string& dir,
         const std::string& build_cmd,
         int64_t min_macs,
         bool sync_build) {
        auto runtime = std::make_unique<XdnaConv3x3wRuntime>();
        runtime->builder.dir = dir;
        runtime->builder.cmd = build_cmd;
        runtime->builder.sync = sync_build;
        runtime->min_macs = min_macs;
        conv3x3w_runtime_slot() = std::move(runtime);
      },
      py::arg("dir"), py::arg("build_cmd"), py::arg("min_macs") = 0,
      py::arg("sync_build") = false);
  m.def(
      "configure_conv3x3w_dw",
      [](const std::string& dir,
         const std::string& build_cmd,
         int64_t min_macs,
         bool sync_build) {
        auto runtime = std::make_unique<XdnaConv3x3wDwRuntime>();
        runtime->builder.dir = dir;
        runtime->builder.cmd = build_cmd;
        runtime->builder.sync = sync_build;
        runtime->min_macs = min_macs;
        conv3x3w_dw_runtime_slot() = std::move(runtime);
      },
      py::arg("dir"), py::arg("build_cmd"), py::arg("min_macs") = 0,
      py::arg("sync_build") = false);
  m.def("conv3x3w_configured", []() {
    return static_cast<bool>(conv3x3w_runtime_slot());
  });
  m.def("conv_family_configured", []() {
    return static_cast<bool>(conv_runtime_slot());
  });
  m.def(
      "configure_fullframe_conv_family",
      [](const std::string& xclbin,
         const std::string& res146,
         const std::string& fwd292,
         const std::string& fwd584,
         const std::string& dx292,
         const std::string& dx584) {
        auto runtime = std::make_unique<XdnaFullFrameConvRuntime>();
        runtime->open(xclbin, res146, fwd292, fwd584, dx292, dx584);
        fullframe_conv_runtime_slot() = std::move(runtime);
      });
  m.def("fullframe_conv_family_configured", []() {
    return static_cast<bool>(fullframe_conv_runtime_slot());
  });
  m.def(
      "configure_fast_conv11_dx",
      [](const std::string& xclbin,
         const std::string& insts,
         int64_t n_phys) {
        auto runtime = std::make_unique<XdnaFullFrameConvRuntime>();
        runtime->open_conv11_only(xclbin, insts, n_phys);
        fast_conv11_runtime_slot() = std::move(runtime);
      });
  m.def("fast_conv11_dx_configured", []() {
    return static_cast<bool>(fast_conv11_runtime_slot());
  });
  m.def(
      "configure_fast_fullframe_family",
      [](const std::string& xclbin,
         const std::string& res146,
         const std::string& dx292,
         const std::string& dx584) {
        auto runtime = std::make_unique<XdnaFullFrameConvRuntime>();
        runtime->open(xclbin, res146, "", "", dx292, dx584);
        fast_dx584_runtime_slot() = std::move(runtime);
      });
  m.def(
      "configure_fast_dx584",
      [](const std::string& xclbin, const std::string& dx584) {
        auto runtime = std::make_unique<XdnaFullFrameConvRuntime>();
        runtime->open(xclbin, dx584, "", "", "", dx584);
        fast_dx584_runtime_slot() = std::move(runtime);
      });
  m.def("fast_dx584_configured", []() {
    return static_cast<bool>(fast_dx584_runtime_slot());
  });
  m.def(
      "fullframe_cat2_upsample2_forward",
      [](const at::Tensor& src0,
         const at::Tensor& src1,
         const at::Tensor& weight) {
        auto& slot = fullframe_conv_runtime_slot();
        TORCH_CHECK(slot, "full-frame XDNA Conv family is not configured");
        auto* stream = slot->find_fused_cat2_upsample2(src0, src1, weight);
        TORCH_CHECK(
            stream != nullptr,
            "no compatible full-frame XDNA fused cat+upsample2 forward stream");
        return slot->forward_cat2_upsample2(src0, src1, weight, *stream);
      });
  m.def(
      "configure_fullframe_dw584",
      [](const std::string& xclbin, const std::string& insts) {
        auto runtime = std::make_unique<XdnaFullFrameDwRuntime>();
        runtime->open(xclbin, insts);
        fullframe_dw_runtime_slot() = std::move(runtime);
      });
  m.def("fullframe_dw584_configured", []() {
    return static_cast<bool>(fullframe_dw_runtime_slot());
  });
  m.def("reset_mapped_cpu_stats", []() {
    std::lock_guard<std::mutex> guard(mapped_cpu_stats_mutex());
    mapped_cpu_stats_map().clear();
  });
  m.def("mapped_cpu_stats", []() {
    std::lock_guard<std::mutex> guard(mapped_cpu_stats_mutex());
    py::dict out;
    for (const auto& [name, stat] : mapped_cpu_stats_map()) {
      py::dict item;
      item["calls"] = py::int_(stat.calls);
      item["total_ms"] = py::float_(double(stat.total_ns) / 1.0e6);
      item["avg_ms"] = py::float_(
          stat.calls ? double(stat.total_ns) / 1.0e6 / stat.calls : 0.0);
      out[py::str(name)] = std::move(item);
    }
    return out;
  });

  m.def("reset_fallback_stats", []() {
    std::lock_guard<std::mutex> guard(fallback_stats_mutex());
    fallback_stats_map().clear();
  });
  m.def("fallback_stats", []() {
    std::lock_guard<std::mutex> guard(fallback_stats_mutex());
    pybind11::dict out;
    for (const auto& [name, stat] : fallback_stats_map()) {
      pybind11::dict item;
      item["calls"] = pybind11::int_(stat.calls);
      item["total_ns"] = pybind11::int_(stat.total_ns);
      item["total_ms"] = pybind11::float_(
          static_cast<double>(stat.total_ns) / 1.0e6);
      out[pybind11::str(name)] = item;
    }
    return out;
  });
  m.def("empty_cache", []() { bo_cache().empty(); });
  m.def("allocator_stats", []() {
    auto& cache = bo_cache();
    std::lock_guard<std::mutex> guard(cache.mutex);
    pybind11::dict out;
    out["cached_bytes"] = pybind11::int_(cache.cached_bytes);
    out["max_cached_bytes"] = pybind11::int_(cache.max_cached_bytes);
    out["tensor_bo_flag"] = pybind11::int_(tensor_bo_flag());
    out["arena_chunk_bytes"] = pybind11::int_(cache.arena_chunk_bytes);
    out["arena_parent_allocations"] =
        pybind11::int_(cache.arena_parent_allocations);
    out["arena_subbuffer_allocations"] =
        pybind11::int_(cache.arena_subbuffer_allocations);
    out["arena_reserved_bytes"] =
        pybind11::int_(cache.arena_reserved_bytes);
    out["fresh_allocations"] = pybind11::int_(cache.fresh_allocations);
    out["reuses"] = pybind11::int_(cache.reuses);
    return out;
  });
  m.def("is_xrt_backed", [](const at::Tensor& tensor) {
    if (!tensor.defined() || tensor.device().type() != kXdnaType) {
      return false;
    }
    auto* ctx = tensor.storage().data_ptr().get_context();
    auto* allocation = static_cast<XdnaAllocation*>(ctx);
    return allocation != nullptr &&
        allocation->magic == kXdnaAllocationMagic &&
        allocation->bo != nullptr;
  });
  m.def("bo_handle", [](const at::Tensor& tensor) {
    auto* allocation = allocation_from_tensor(tensor);
    return reinterpret_cast<uintptr_t>(allocation->bo);
  });
  m.def("bo_size", [](const at::Tensor& tensor) {
    auto* allocation = allocation_from_tensor(tensor);
    return static_cast<uint64_t>(shim_bo_size(allocation->bo));
  });
  m.def("sync_to_device", [](const at::Tensor& tensor) {
    auto* allocation = allocation_from_tensor(tensor);
    TORCH_CHECK(
        shim_bo_sync_to_device(allocation->bo) == 0,
        "XDNA sync-to-device failed: ",
        shim_last_error());
    allocation->coherency = XdnaCoherency::Clean;
  });
  m.def("sync_from_device", [](const at::Tensor& tensor) {
    auto* allocation = allocation_from_tensor(tensor);
    TORCH_CHECK(
        shim_bo_sync_from_device(allocation->bo) == 0,
        "XDNA sync-from-device failed: ",
        shim_last_error());
    allocation->coherency = XdnaCoherency::Clean;
  });
  m.def("prepare_device_read", [](const at::Tensor& tensor) {
    ensure_device_current(tensor);
  });
  m.def("prepare_host_read", [](const at::Tensor& tensor) {
    ensure_host_current(tensor);
  });
  m.def("mark_device_dirty", [](const at::Tensor& tensor) {
    mark_device_dirty(tensor);
  });
  m.def("mark_host_dirty", [](const at::Tensor& tensor) {
    mark_host_dirty(tensor);
  });
  m.def("coherency_state", [](const at::Tensor& tensor) {
    return static_cast<int>(allocation_from_tensor(tensor)->coherency);
  });
}
