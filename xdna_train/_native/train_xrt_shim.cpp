// Extend the shared XRT shim without modifying xdna-engine.
// Including the implementation keeps all opaque Shim* definitions in this TU.
#include "xrt_shim.cpp"

extern "C" void* shim_bo_map(ShimBo* b) {
  GUARD_PTR(return b->bo.map<void*>();)
}

extern "C" size_t shim_bo_size(ShimBo* b) {
  try { return b->bo.size(); }
  catch (const std::exception& e) { set_err(e.what()); return 0; }
  catch (...) { set_err("unknown C++ exception"); return 0; }
}

// Async variant of shim_run_kernel: submits and returns; wait with shim_run_wait, free with shim_run_free.
extern "C" ShimRun* shim_run_kernel_start(ShimKernel* k, unsigned int opcode, ShimBo* instr,
                                          size_t instr_count, ShimBo* const* data, size_t n_data) {
  GUARD_PTR(
    xrt::run run(k->kern);
    run.set_arg(0, opcode);
    run.set_arg(1, instr->bo);
    run.set_arg(2, instr_count);
    std::vector<xrt::bo> args{instr->bo};
    for (size_t i = 0; i < n_data; ++i) {
      run.set_arg(static_cast<int>(3 + i), data[i]->bo);
      args.push_back(data[i]->bo);
    }
    run.start();
    return new ShimRun{ std::move(run), std::move(args), "kernel_start" };
  )
}
