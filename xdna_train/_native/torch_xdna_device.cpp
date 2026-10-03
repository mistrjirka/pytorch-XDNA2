#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/convolution_ops.h>
#include <ATen/ops/convolution_backward_ops.h>
#include <ATen/ops/mm_ops.h>
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
#include <c10/core/DeviceGuard.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/extension.h>
#include "xrt_shim.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <array>
#include <fstream>
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
extern "C" void xdna_pack_conv3x3_dy_channels_last_bf16(
    const uint16_t*, uint16_t*, int, int, int, int);
extern "C" void xdna_pack_conv3x3_dy_bf16(
    const uint16_t*, uint16_t*, int, int, int, int, int);

namespace {

constexpr c10::DeviceType kXdnaType = c10::DeviceType::PrivateUse1;
constexpr size_t kPageAlignment = 4096;
constexpr uint64_t kXdnaAllocationMagic = 0x58444e41424f3031ULL; // "XDNABO01"
thread_local c10::DeviceIndex g_current_device = 0;

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
  TORCH_CHECK(
      self.is_contiguous(),
      "XDNA view currently requires contiguous input; use contiguous() first");
  auto size = at::infer_size_dv(requested_size, self.sym_numel());
  at::SymDimVector stride(size.size());
  c10::SymInt running = 1;
  for (int64_t i = static_cast<int64_t>(size.size()) - 1; i >= 0; --i) {
    stride[static_cast<size_t>(i)] = running;
    running = running * size[static_cast<size_t>(i)];
  }
  return make_alias(self, size, stride, self.sym_storage_offset());
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
      const std::string& dx36) {
    kernel = shim_kernel_load(
        xdna_device(), xclbin.c_str(), nullptr, QOS_PRIORITY_NONE);
    TORCH_CHECK(kernel != nullptr, "failed to load XDNA Conv family: ", shim_last_error());
    int pack_threads = 12;
    if (const char* raw = std::getenv("XDNA_CONV_PACK_THREADS")) {
      const int parsed = std::atoi(raw);
      if (parsed > 0) pack_threads = parsed;
    }
    xdna_conv_set_threads(pack_threads);
    open_stream(
        forward_streams[0], "fwd18", fwd18, 256, 18, 12960, 2304);
    open_stream(
        dx_streams[0], "dx18", dx18, 128, 18, 12960, 1152);
    open_stream(
        forward_streams[1], "fwd36", fwd36, 256, 36, 51840, 2304);
    open_stream(
        dx_streams[1], "dx36", dx36, 128, 36, 51840, 1152);
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
    const int64_t real_m = kBatch * h * h;
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

      auto raw_cpu = cpu_alias(raw);
      auto view = raw_cpu
          .narrow(0, 0, real_m)
          .view({h, h, kBatch, kCout})
          .permute({2, 3, 0, 1});
      out_cpu
          .narrow(1, start, kCout)
          .copy_(view);
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

    auto raw = empty_memory_format(
        {c10::SymInt(stream.m_sched), c10::SymInt(kCout)},
        at::ScalarType::BFloat16,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::Contiguous);
    auto* out = allocation_from_tensor(raw);
    {
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

struct XdnaFullFrameConvStream {
  std::string name;
  int64_t cin = 0;
  int64_t h = 0;
  int64_t w = 0;
  int64_t w_sched = 0;
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

struct XdnaFullFrameConvRuntime {
  static constexpr int64_t kBatch = 4;
  static constexpr int64_t kCout = 128;
  static constexpr int64_t kCBlock = 32;

  ShimKernel* kernel = nullptr;
  std::array<XdnaFullFrameConvStream, 3> forward_streams;
  std::array<XdnaFullFrameConvStream, 2> dx_streams;
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
      int64_t k_gemm) {
    auto blob = read_file(path);
    TORCH_CHECK(blob.size() % 4 == 0, "full-frame Conv stream is not word aligned: ", path);
    stream.name = name;
    stream.cin = cin;
    stream.h = h;
    stream.w = w;
    stream.w_sched = w_sched;
    stream.m_sched = m_sched;
    stream.k_gemm = k_gemm;

    const int gid_instr = shim_kernel_group_id(kernel, 1);
    const int gid_tmp = shim_kernel_group_id(kernel, 6);
    const int gid_trace = shim_kernel_group_id(kernel, 7);
    stream.instr = shim_bo_alloc(
        xdna_device(), kernel, blob.size(), 1, gid_instr);
    stream.a = shim_bo_alloc(
        xdna_device(), kernel, a_elems(cin, h, w_sched) * 2, 2, 0);
    stream.b = shim_bo_alloc(
        xdna_device(), kernel, static_cast<size_t>(k_gemm * kCout * 2), 2, 0);
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
        stream.b_ptr, 0, static_cast<size_t>(k_gemm * kCout * 2));
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
        grad_output.size(1) != kCout ||
        weight.size(0) != kCout ||
        (weight.size(1) != 128 && weight.size(1) != 256) ||
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
    if (weight.size(1) == 256) {
      for (auto& stream : dx_streams) {
        if (!stream.instr || grad_output.size(3) != stream.w)
          continue;
        if (grad_output.size(2) == stream.h)
          return &stream;
        if (stream.name == "dx584s73" &&
            grad_output.size(2) == 584 &&
            584 % stream.h == 0) {
          return &stream;
        }
      }
    }
    return nullptr;
  }

  at::Tensor dispatch_raw(
      XdnaFullFrameConvStream& stream,
      at::ScalarType dtype) {
    auto raw = empty_memory_format(
        {c10::SymInt(stream.m_sched), c10::SymInt(kCout)},
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
        c10::SymInt(kCout),
        c10::SymInt(1),
        c10::SymInt(stream.w_sched * kBatch * kCout),
        c10::SymInt(kBatch * kCout)};
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
    TORCH_CHECK(
        cin % kCout == 0,
        "full-frame dX requires input channels divisible by ", kCout,
        "; got ", cin);
    const int nslices = static_cast<int>(cin / kCout);

    auto out = empty_memory_format(
        {
            c10::SymInt(kBatch),
            c10::SymInt(cin),
            c10::SymInt(full_h),
            c10::SymInt(full_w),
        },
        at::ScalarType::BFloat16,
        at::Layout::Strided,
        c10::Device(kXdnaType, 0),
        false,
        at::MemoryFormat::ChannelsLast);

    // CPU aliases are zero-copy views of the host-mapped XDNA BO. Convert a
    // non-channels-last dY once, outside the stripe loop; the normal compiled
    // path already preserves channels-last and takes no conversion here.
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
          static_cast<int>(kCout),
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
        const int start_channel = slice * static_cast<int>(kCout);
        xdna_pack_conv3x3_weight_dx_slice_bf16(
            static_cast<const uint16_t*>(weight_cpu.const_data_ptr()),
            static_cast<uint16_t*>(stream.b_ptr),
            static_cast<int>(kCout),
            static_cast<int>(cin),
            start_channel,
            static_cast<int>(kCout));
        TORCH_CHECK(
            shim_bo_sync_to_device(stream.b) == 0,
            "failed to sync full-frame decoder dX weight slice: ",
            shim_last_error());

        auto raw = dispatch_raw(stream, at::ScalarType::BFloat16);
        ensure_host_current(raw);
        xdna_yxbc_slice_stripe_to_channels_last_bf16(
            static_cast<const uint16_t*>(raw.const_data_ptr()),
            static_cast<uint16_t*>(out.mutable_data_ptr()),
            static_cast<int>(kBatch),
            static_cast<int>(cin),
            static_cast<int>(kCout),
            start_channel,
            static_cast<int>(full_h),
            static_cast<int>(y0),
            static_cast<int>(stream.h),
            static_cast<int>(full_w),
            static_cast<int>(stream.w_sched));
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


at::Tensor xdna_from_cpu(const at::Tensor& cpu);

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

bool lazy_decoder_values_enabled() {
  const char* raw = std::getenv("XDNA_LAZY_DECODER_VALUES");
  return raw != nullptr && raw[0] != 0 && std::strcmp(raw, "0") != 0;
}

at::Tensor cat_xdna(
    const at::ITensorListRef& tensors,
    int64_t dim) {
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

  ensure_host_current(input);
  ensure_host_current(weight);
  auto input_cpu = cpu_alias(input);
  auto weight_cpu = cpu_alias(weight);
  std::optional<at::Tensor> bias_cpu = std::nullopt;
  if (bias.has_value() && bias->defined()) {
    ensure_host_current(*bias);
    bias_cpu = cpu_alias(*bias);
  }

  const auto shape = conv_output_shape(
      input,
      weight,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups);
  const auto output_format =
      input.dim() == 4 &&
          input.is_contiguous(at::MemoryFormat::ChannelsLast)
      ? at::MemoryFormat::ChannelsLast
      : at::MemoryFormat::Contiguous;
  auto out = empty_memory_format(
      shape,
      input.scalar_type(),
      at::Layout::Strided,
      c10::Device(kXdnaType, 0),
      false,
      output_format);
  auto out_cpu = cpu_alias(out);

  at::_ops::convolution_out::call(
      input_cpu,
      weight_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      out_cpu);
  mark_host_dirty(out);
  return out;
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
  XdnaConvStream* small_dx = nullptr;
  XdnaFullFrameConvStream* full_dx = nullptr;
  auto& conv_slot = conv_runtime_slot();
  auto& full_conv_slot = fullframe_conv_runtime_slot();
  if (output_mask[0] && conv_slot) {
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
  if (output_mask[0] && !small_dx && full_conv_slot) {
    full_dx = full_conv_slot->find_dx(
        grad_output,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups);
  }
  const bool have_npu_dx = small_dx != nullptr || full_dx != nullptr;

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
  if (overlap_cpu_dw && have_npu_dx && output_mask[1] && !have_npu_dw) {
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
            return std::get<1>(grads);
          });
      has_overlapped_dw = true;
    }
    cpu_mask[1] = false;
  }

  if (small_dx) {
    gi = conv_slot->input_grad(grad_output, weight, *small_dx);
    cpu_mask[0] = false;
  } else if (full_dx) {
    gi = full_conv_slot->input_grad(grad_output, weight, *full_dx);
    cpu_mask[0] = false;
  }

  if (have_npu_dw) {
    gw = full_dw_slot->weight_grad(input, grad_output, weight);
    cpu_mask[1] = false;
  }

  if (cpu_mask[0] || cpu_mask[1] || cpu_mask[2]) {
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

    if (cpu_mask[0] && cpu_mask[1]) {
      gi = xdna_empty_like_layout(input);
      gw = xdna_empty_like_shape(weight.sym_sizes(), weight.scalar_type());
      std::vector<c10::SymInt> gb_shape = {
          transposed ? weight.sym_size(1) * groups : weight.sym_size(0)};
      auto gb_work = xdna_empty_like_shape(gb_shape, grad_output.scalar_type());
      auto gi_cpu = cpu_alias(gi);
      auto gw_cpu = cpu_alias(gw);
      auto gb_cpu = cpu_alias(gb_work);
      std::vector<c10::SymInt> full_bias_vec = {gb_shape[0]};
      at::OptionalSymIntArrayRef full_bias_sizes =
          c10::SymIntArrayRef(full_bias_vec);
      const std::array<bool, 3> full_mask = {true, true, true};

      at::_ops::convolution_backward_out::call(
          go_cpu,
          input_cpu,
          weight_cpu,
          full_bias_sizes,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
          full_mask,
          gi_cpu,
          gw_cpu,
          gb_cpu);

      mark_host_dirty(gi);
      mark_host_dirty(gw);
      mark_host_dirty(gb_work);
      if (cpu_mask[2]) {
        gb = gb_work;
      }
    } else {
      auto grads = at::_ops::convolution_backward::call(
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
      if (cpu_mask[0]) gi = xdna_from_cpu(std::get<0>(grads));
      if (cpu_mask[1]) gw = xdna_from_cpu(std::get<1>(grads));
      if (cpu_mask[2]) gb = xdna_from_cpu(std::get<2>(grads));
    }
  }

  if (has_overlapped_dw) {
    gw = xdna_from_cpu(overlapped_dw.get());
  }

  return {gi, gw, gb};
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

void xdna_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  if (!profile_fallbacks()) {
    at::native::cpu_fallback(op, stack, false, c10::DispatchKey::CPU);
    return;
  }
  const std::string name = op.schema().name();
  const auto t0 = std::chrono::steady_clock::now();
  at::native::cpu_fallback(op, stack, false, c10::DispatchKey::CPU);
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
         const std::string& dx36) {
        auto runtime = std::make_unique<XdnaConvRuntime>();
        runtime->open(xclbin, fwd18, dx18, fwd36, dx36);
        conv_runtime_slot() = std::move(runtime);
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
