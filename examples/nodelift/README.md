# NodeLift community research wrapper

This is a research wrapper and reproducibility bundle from [NodeLift](https://github.com/sev7enITA/NodeLift), maintained separately on an experimental branch of a llama.cpp fork. It links to llama.cpp through its public API. The prefix reuse measured here already exists in llama.cpp and in NodeLift's native wrapper. This branch does not claim an upstream llama.cpp optimization or that caching is active in the NodeLift application controller.

## Status and limits

The wrapper is experimental. It supports greedy-only generation in this interface, finite contexts, selected model architectures and opt-in serial verification. Batched and serial numerical results can differ. The deferred draft cache mode is disabled by default. This package does not pool device RAM, offer a production server, or attest acceleration across devices. See the [prefix reuse experiment](research/prefix-cache-2026-11-02/README.md) for its synthetic single-Mac scope, precise timings, methodology and model/license caveats.

The upstream source was pinned to `ddf48f85c933c1f438212a02eb21e7b001d2d37d`; see `native/llama.lock`. The CMake build requires a C/C++ compiler and CMake 3.22 or newer. The test scripts use Python 3.11 or newer. The compiled library itself has no Python dependency. Do not distribute model weights without their separate licence.

## Build on macOS

From the repository root at the pinned branch:

```sh
cmake -S examples/nodelift/native -B build-nodelift \
  -DLLAMA_SOURCE_DIR="$PWD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DGGML_METAL=ON
cmake --build build-nodelift --target nodelift -j 4
export NODELIFT_LIBRARY="$PWD/build-nodelift/lib/libnodelift.dylib"
python3 examples/nodelift/scripts/check-native.py --model /path/to/model.gguf \
  --gpu-layers 999 --strict --output /tmp/nodelift-correctness.json
python3 examples/nodelift/research/prefix-cache-2026-11-02/bench.py \
  --model /path/to/model.gguf --out /tmp/nodelift-prefix-results.json
```

Set GPU layers to 0 to request CPU execution. CUDA, Windows, Linux, iOS and Android builds from this branch have not been verified as part of this publication. The optional `NODELIFT_EXPERIMENTAL_DEFERRED_DRAFT` flag is unqualified and remains off.

## Build evidence

A macOS arm64 library was compiled and tested with real model weights. The [build report](validation/BUILD.md) records its exact scope, source verification and 31 strict checks.

## Attribution

NodeLift code is copyright Fabrizio Degni under [MIT](LICENSE.nodelift). llama.cpp is copyright its contributors under the repository's [MIT licence](../../LICENSE). See [third-party notices](native/licenses/) for bundled dependencies. The benchmark and wrapper were developed with AI assistance; their limitations and test evidence are recorded with the results. This branch is an independent experiment and has not been submitted or accepted upstream.
