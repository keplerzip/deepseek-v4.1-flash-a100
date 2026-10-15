"""Run the real sparse-prefill combiner against an independent scalar reference.

This requires CPU PyTorch, but neither a GPU nor vLLM imports. Passing these
checks does not measure CUDA synchronization counts or end-to-end performance.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import torch


DEPLOY = Path(__file__).resolve().parents[1]
SOURCE = (
    DEPLOY / "overrides/performance/patched"
    / "vllm/models/deepseek_v4_1/amd/rocm.py"
)


def load_combiner():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "combine_topk_swa_indices"
    )
    namespace = {"torch": torch, "_SPARSE_PREFILL_TOPK_ALIGNMENT": 128}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name]


def scalar_reference(topk_rows, starts, seq_lens, gather_lens, window, ratio, topk, m, n):
    width = ((topk + window + 127) // 128) * 128
    indices, lengths = [], []
    row_index = 0
    for req, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        query_len = end - start
        for offset in range(query_len):
            position = seq_lens[req] - query_len + offset
            topk_len = max(0, min((position + 1) // ratio, topk, len(topk_rows[row_index])))
            swa_len = max(0, min(position + 1, window))
            row = [-1] * width
            for column in range(topk_len):
                value = topk_rows[row_index][column]
                if 0 <= value < n:
                    row[column] = m * req + value
            gather_start = seq_lens[req] - gather_lens[req]
            for column in range(swa_len):
                row[topk_len + column] = (
                    m * req + n + column + position - swa_len + 1 - gather_start
                )
            indices.append(row)
            lengths.append(topk_len + swa_len)
            row_index += 1
    return torch.tensor(indices, dtype=torch.int32), torch.tensor(lengths, dtype=torch.int32)


def main():
    combiner = load_combiner()
    cases = [
        # Narrowed mixed batches retain a non-zero query_start_loc base.
        dict(name="mixed-nonzero-base", starts=[17, 20, 22], seq=[21, 40], gather=[21, 32],
             window=8, ratio=4, topk=6, width=8),
        dict(name="single-token", starts=[0, 1], seq=[1], gather=[1],
             window=128, ratio=4, topk=8, width=8),
        dict(name="zero-query-slot", starts=[29, 31, 31, 34], seq=[30, 50, 11],
             gather=[20, 30, 11], window=8, ratio=4, topk=8, width=8),
        dict(name="long-prefill", starts=[256, 16640], seq=[32768], gather=[16512],
             window=128, ratio=4, topk=8, width=8),
        dict(name="swa-only", starts=[7, 10, 12], seq=[3, 100], gather=[3, 16],
             window=8, ratio=1, topk=0, width=8),
        dict(name="logical-topk-wider-than-storage", starts=[0, 3], seq=[100], gather=[32],
             window=8, ratio=4, topk=12, width=8),
    ]
    results = []
    for case in cases:
        count = case["starts"][-1] - case["starts"][0]
        for dtype in (torch.int32, torch.int64):
            starts = torch.tensor(case["starts"], dtype=dtype)
            seq = torch.tensor(case["seq"], dtype=dtype)
            gather = torch.tensor(case["gather"], dtype=dtype)
            # Exercise noncontiguous rows, negative sentinels and out-of-range candidates.
            storage = torch.arange(count * case["width"] * 2).view(count, case["width"] * 2)
            candidates = (storage % 37 - 3)[:, ::2]
            assert not candidates.is_contiguous() or count == 1
            args = (candidates, starts, seq, gather, case["window"], case["ratio"],
                    case["topk"], 1024, 32)
            original_repeat = torch.repeat_interleave
            with patch.object(torch, "repeat_interleave", wraps=original_repeat) as calls:
                actual = combiner(*args)
                assert calls.call_count == 2
                assert all(call.kwargs.get("output_size") == count for call in calls.call_args_list)
            expected = scalar_reference(
                candidates.tolist(), case["starts"], case["seq"], case["gather"],
                case["window"], case["ratio"], case["topk"], 1024, 32,
            )
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
            results.append(dict(name=case["name"], query_dtype=str(dtype), tokens=count, status="PASS"))
    print(json.dumps(dict(status="PASS", gpu_executed=False,
                         scope="actual sparse-prefill combiner; exact scalar reference; known output lengths",
                         cases=results)))


if __name__ == "__main__":
    main()
