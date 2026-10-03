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
