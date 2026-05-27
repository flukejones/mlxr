# mlx-lm decode bench

## Running

```
cargo bench -p mlx-lm --bench lm_decode
```

Single cell:

```
MLX_LM_BENCH_ONLY=qwen3_decode_large_bf16 cargo bench -p mlx-lm --bench lm_decode
```

## Environment knobs

- `MLX_LM_BENCH_CACHE` — checkpoint cache root (default `~/.cache/mlx-rs-bench`).
- `MLX_LM_BENCH_NO_DOWNLOAD=1` — skip cells whose checkpoint isn't cached.
- `MLX_LM_BENCH_SET={trimmed,full}` — `trimmed` (default) runs llama 1B + qwen3 1.7B; `full` adds llama 3B + qwen3 0.6B.
- `MLX_LM_BENCH_ONLY=<substr>` — substring filter on per-cell group prefix.

Checkpoints download via `hf` CLI on first use; cells skip silently if `hf` is unavailable or download fails.

## Cells

- `llama_decode_small_{bf16,q8,q4}` — `mlx-community/Llama-3.2-1B-Instruct-{bf16,8bit,4bit}`
- `qwen3_decode_large_{bf16,q8,q4}` — `mlx-community/Qwen3-1.7B-{bf16,8bit,4bit}`

Each cell runs `prefill_short` (13-token prompt), `prefill_long` (1024), `decode_short` (99 tokens after short prompt), `decode_long` (99 after long prompt).

Methodology: criterion 10-sample × 20 s window. `WARMUP_TOKENS = 4` decode steps outside timing; `DECODE_TOKENS = 100` timed.


## Results

Run the harness to populate. Median times in milliseconds; each cell is an
isolated process (`MLX_LM_BENCH_ONLY=<cell>`) for a fresh model load, kernel
cache, and mlx-c state per measurement.
