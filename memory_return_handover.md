# Handover: InferenceSession Memory Return to OS

## Goal
Make `~InferenceSession()` return freed memory to the OS so RSS doesn't accumulate across create→run→destroy cycles.

## What's Done
- `ReleaseFreedMemoryToOS()` helper in `onnxruntime/core/framework/allocator.{h,cc}` — dispatches to `malloc_trim(0)` (glibc) or `mi_collect(true)` (mimalloc via dlsym).
- `ExecutionProviders::Clear()` in `onnxruntime/core/framework/execution_providers.h` for deterministic EP teardown.
- `~InferenceSession()` (`onnxruntime/core/session/inference_session.cc:928-937`) — resets members in reverse-declaration order then calls `ReleaseFreedMemoryToOS()`.
- `OrtApis::ReleaseSession` (`onnxruntime/core/session/onnxruntime_c_api.cc`) — calls `ReleaseFreedMemoryToOS()` after `delete`.
- Python `InferenceSession.__del__` (`onnxruntime/python/onnxruntime_inference_collection.py`) — drops metadata refs, calls `release_freed_memory_to_os`.
- `release_freed_memory_to_os` exposed via pybind (`onnxruntime/python/onnxruntime_pybind_state.cc`).
- `~PyInferenceSession` (`onnxruntime/python/onnxruntime_pybind_state_common.h`) — resets `sess_`, calls `ReleaseFreedMemoryToOS()`.
- Test file in `onnxruntime/test/framework/inference_session_test.cc`:
  - `ReleaseFreedMemoryToOSDoesNotCrash` — no-platform-guard, just calls the fn.
  - `DestroyDoesNotAccumulateMemory` — `#if defined(__linux__) && !defined(__ANDROID__)`, compares RSS after warmup vs after cycle 1 vs after cycle 2.

## Test Implementation (Completed)

The test was updated to use `bart_tiny.onnx` (28.7 MB) and simplified to compare RSS across consecutive cycles rather than checking a plateau over many cycles:

**Test approach:**
1. Warmup: Load + Initialize + Run `mul_1.onnx` to page in shared libraries and trigger one-time allocations
2. Measure RSS after warmup (`rss_before`)
3. Cycle 1: Load + Initialize + Destroy `bart_tiny.onnx` (no Run, memory allocation happens during Load+Initialize)
4. Measure RSS after cycle 1 (`rss_after_cycle1`)
5. Cycle 2: Load + Initialize + Destroy `bart_tiny.onnx`
6. Measure RSS after cycle 2 (`rss_after_cycle2`)

**Verification:**
- With fix re-enabled: Test PASSES (RSS is flat across destroy cycles)
- Without fix (`ReleaseFreedMemoryToOS()` commented out): Test FAILS (RSS grew 34 MB across cycles)

**Key parameters:**
- Model: `testdata/bart_tiny.onnx` (28,712,022 bytes)
- `kMaxRssDifferenceKB = 10 * 1024` — 10 MB threshold per cycle difference
- RSS measured via `/proc/self/status` (linux only)

## Files Changed
```
M include/onnxruntime/core/framework/allocator.h
M onnxruntime/core/framework/allocator.cc
M onnxruntime/core/framework/execution_providers.h
M onnxruntime/core/session/inference_session.cc
M onnxruntime/core/session/onnxruntime_c_api.cc
M onnxruntime/python/onnxruntime_pybind_state_common.h
M onnxruntime/python/onnxruntime_pybind_state.cc
M onnxruntime/python/onnxruntime_inference_collection.py
M onnxruntime/test/framework/inference_session_test.cc
```

## Lint Status
- Ran `lintrunner -a` on modified C++ files — no lint issues found.

## Build Commands (for speed)
```bash
# Reconfigure CMake if needed:
cmake -B build/cpu/Release -S cmake -DCMAKE_BUILD_TYPE=Release

# Rebuild test binary:
cmake --build build/cpu/Release --target onnxruntime_test_all -- -j$(nproc)

# Run single test:
./build/cpu/Release/onnxruntime_test_all \
  --gtest_filter="InferenceSessionTests.DestroyDoesNotAccumulateMemory" \
  --gtest_color=no
```
