# NodeLift community research wrapper

This is a research wrapper and reproducibility bundle from [NodeLift](https://github.com/sev7enITA/NodeLift), maintained separately on an experimental branch of a llama.cpp fork. It links to llama.cpp through its public API. The prefix reuse measured here already exists in llama.cpp and in NodeLift's native wrapper. This branch does not claim an upstream llama.cpp optimization or that caching is active in the NodeLift application controller.

## Status and limits

The wrapper is experimental. It supports greedy-only generation in this interface, finite contexts, selected model architectures and opt-in serial verification. Batched and serial numerical results can differ. The deferred draft cache mode is disabled by default. This package does not pool device RAM, offer a production server, or attest acceleration across devices. See the [prefix reuse experiment](research/prefix-cache-2026-11-02/README.md) for its synthetic single-Mac scope, precise timings, methodology and model/license caveats.

The native API accepts context sizes from 256 to 8,192 tokens, 1 to 16 CPU threads, and requested GPU layer counts from 0 to 999. A draft call generates at most 512 tokens; a verification call accepts at most 16 proposal tokens with a remaining budget of 1 to 512. Chat input accepts at most 32 messages. Prefix and output must also fit the context. These are wrapper limits, not guarantees that a particular model or device can handle the maximum.

The published 31-check result covers the opt-in serial verifier. Calls without `strict: true` use the different batched path, which is not certified by that result. Serial results include `verification_mode: serial`; the current batch response does not include that field. Absence of the field must not be interpreted as serial validation.

The upstream source was pinned to `ddf48f85c933c1f438212a02eb21e7b001d2d37d`; see `native/llama.lock`. The CMake build requires a C/C++ compiler and CMake 3.22 or newer. The test scripts use Python 3.11 or newer. The compiled library itself has no Python dependency. Do not distribute model weights without their separate licence.

## Input trust boundary

The current tokenizer parses special-token spellings in formatted chat text. Local tests found that untrusted message content containing a literal delimiter can add a special token to the resulting sequence. Model response to such input was not tested. Do not rely on this experimental wrapper to isolate untrusted user or retrieved text from template control tokens. See the [tokenizer check](research/prefix-cache-2026-11-02/tokenizer-review.json) and research report.

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

The 3 November source update adds `core.h` to the CMake target sources so an iOS framework can package its public header. It does not replace the existing macOS prerelease binary or certify an iOS build from this fork. See the [follow-up assessment](https://github.com/sev7enITA/NodeLift/blob/main/research/reviewer-resumption-2026-11-03/REVIEW.md). The existing macOS binary exports upstream and SHA-256 symbols as well as the NodeLift ABI; coexistence with other native runtimes remains unqualified. An actual symbol collision has not been demonstrated.

## Attribution

NodeLift code is copyright Fabrizio Degni under [MIT](LICENSE.nodelift). llama.cpp is copyright its contributors under the repository's [MIT licence](../../LICENSE). See [third-party notices](native/licenses/) for bundled dependencies. The benchmark and wrapper were developed with AI assistance; their limitations and test evidence are recorded with the results. This branch is an independent experiment and has not been submitted or accepted upstream.
