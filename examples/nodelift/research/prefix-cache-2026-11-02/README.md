# Prefix reuse: exploratory native microbenchmark

On 2 November 2026, a single Mac ran the fixed [plan](PLAN.md) against an already installed Llama 3.1 8B Q4_K_M model and NodeLift's existing native library. The test exercises `Engine::sync` directly, rather than changing the desktop controller, which currently resets its context at the beginning of each request.

| Base prefix length | Cold prefill, median | Reused prefill, median | Median paired prefill ratio | Median paired ratio including 16 generated tokens | Identical token and EOS pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 tokens | 606.2 ms | 52.4 ms | 11.25x | 2.13x | 6/6 |
| 1024 tokens | 2240.1 ms | 58.6 ms | 38.99x | 5.14x | 6/6 |

Correction after review: the tokenizer inserted a common beginning-of-sequence token at the start of both separately tokenized suffixes. The exact common prefixes in this recorded run were therefore 257 and 1,025 tokens, respectively, rather than 256 and 1,024. The table labels refer to the synthetic base lengths; the retained results and timings are unchanged. The native synchronization may replay the final shared token to restore logits, so common-prefix length is not the same as the number of tokens actually skipped. The runner now records the complete shared length and the supplied model filename instead of hard-coding a model name.

Ratios are medians of the six paired ratios, not ratios of the two marginal medians. Priming the retained context took place outside the measured request and was done for both arms. The cold reset itself was also outside the timed window. The prompt is synthetic and deliberately repeats the exact prefix; it does not establish the frequency of cache hits in real conversations. Timings exclude initial model loading and do not isolate first-token latency. The generated output is a short greedy native draft call, not an application or network benchmark. There was no independent GPU placement attestation, load, energy or thermal measurement, and no statistical confidence interval. Twelve equal token/EOS pairs cannot prove equivalence for other models, backends, batching shapes or inputs.

The data supports pursuing an opt-in, session-isolated prefix cache with explicit identity, limits, invalidation, observability and cancellation tests. Before using it, verify model hash, context/backend settings, adapters, exact token prefix and the intended session boundary. Rollback and re-prefill safely after model changes, errors or cancellation. Prompt or KV data may contain private information. Keep caching in memory by default; do not share or persist it across users without an explicit design. Numerical equality needs separate qualification, especially because the llama.cpp server [documents potential backend-dependent differences](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) when caching changes batching.

An independent static review raised a separate tokenizer trust-boundary concern. In [two local tokenization checks](tokenizer-review.json), a literal special-token spelling embedded in user content became an additional special token in the formatted chat sequence for both tested model families. Whether any particular model follows injected instructions was not tested. Do not assume that untrusted text is isolated from chat-template delimiters. Changing special-token parsing globally may also break template formatting, so a model-aware remedy and targeted tests are required. The same checks did not reproduce a duplicated initial BOS for the tested Llama and Qwen chat prompts.

Other avenues are model residency and carefully bounded prefetch. Measure cold load separately from warm requests and account for RAM pressure. Parallel draft prefill can contend with target work, while speculative proposals cannot be trusted after their authoritative prefix changes. Measure and cancel stale work. Remote KV offload is a separate idea whose transfer and recovery cost must be measured against local recomputation. See [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) and [LMCache](https://docs.lmcache.ai/) for existing approaches. No production feature is enabled by this experiment.

Reproduce with an existing compatible GGUF and native library:

```sh
python3 research/prefix-cache-2026-11-02/bench.py --model /path/to/llama3.1-8b.gguf --out /tmp/nodelift-prefix-results.json
```

[Raw observations](results.json) contain the model and library hashes, requested settings, tokens, order and all measurements. The model is not included. The Python script is research tooling and does not impose Python on users of the standalone application.
