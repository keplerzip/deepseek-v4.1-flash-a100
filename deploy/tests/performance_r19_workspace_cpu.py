"""Exercise the production indexer allocator against the real chunk builder.

Standard-library-only host check: no CUDA import, tensor allocation or fabricated
GPU runtime. The integer scalars below expose only the .item() interface used by
the production chunk planner. This verifies allocation bounds, not GPU results.
"""
import __future__
import ast
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sysconfig
from types import SimpleNamespace as NS


def load_function(path, name, namespace, cls=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if cls:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class Scalar(int):
    def item(self):
        return int(self)


def main():
    overlay = Path(__file__).resolve().parents[1] / "overrides/performance/patched"
    site = overlay if overlay.exists() else Path(sysconfig.get_paths()["purelib"])
    indexer_path = site / "vllm/v1/attention/backends/mla/indexer.py"
    env = NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512)
    namespace = dict(envs=env, math=math)
    get_size = load_function(indexer_path, "get_max_prefill_buffer_size", namespace)
    size = load_function(site / "vllm/models/deepseek_v4_1/attention.py",
                         "_indexer_prefill_buffer_size", namespace)
    split = load_function(indexer_path, "_split_indexer_prefill_chunks", namespace,
                          "DeepseekV32IndexerMetadataBuilder")
    rng = random.Random(20261015)
    chunk_count = 0
    workloads = 0

    for context in (128, 4096, 131072, 512000, 1048576):
        cfg = NS(model_config=NS(max_model_len=context))
        for ratio in (1, 2):
            compressed_max = context // ratio
            for logits_mb in (1, 16, 512, 8192):
                env.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = logits_mb
                reserved = size(cfg, ratio)
                assert compressed_max <= reserved <= get_size(cfg)
                cases = [([compressed_max] * 256, [1] * 256),
                         ([compressed_max], [min(context, 1024)]),
                         ([1] * 64, [1] * 64), ([], [])]
                for _ in range(32):
                    count = rng.randrange(1, 65)
                    keys = [rng.randrange(1, compressed_max + 1) for _ in range(count)]
                    queries = [rng.randrange(1, min(1024, k * ratio) + 1) for k in keys]
                    cases.append((keys, queries))
                for keys, queries in cases:
                    offset = 3
                    chunks = split(list(map(Scalar, keys)), list(map(Scalar, queries)),
                                   get_size(cfg), logits_mb * 1024 * 1024,
                                   request_offset=offset)
                    grouped = defaultdict(list)
                    for requests, query_rows in chunks:
                        request_range = slice(requests.start-offset, requests.stop-offset)
                        needed = sum(keys[request_range])
                        assert needed <= reserved, (context, ratio, logits_mb, needed, reserved)
                        assert query_rows.start < query_rows.stop
                        grouped[(request_range.start, request_range.stop)].append(query_rows)
                    # Query sub-chunking must retain a complete, non-overlapping
                    # range, including the single-long-request fallback.
                    consumed_requests = 0
                    for (start, stop), parts in grouped.items():
                        assert start == consumed_requests
                        consumed_requests = stop
                        assert parts[0].start == 0
                        assert all(a.stop == b.start for a, b in zip(parts, parts[1:]))
                        assert parts[-1].stop == sum(queries[start:stop])
                    assert consumed_requests == len(keys)
                    chunk_count += len(chunks)
                    workloads += 1

    # A ratio-2 dense chunk can legally exceed the old compressed cap in this
    # pinned builder. Matching its actual cap prevents a hidden overflow when
    # users raise the logits budget or shorten max_model_len.
    cfg = NS(model_config=NS(max_model_len=128))
    env.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = 8192
    assert size(cfg, 2) == get_size(cfg) > get_size(cfg) // 2

    cfg = NS(model_config=NS(max_model_len=512000))
    env.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = 512
    rows = {ratio: size(cfg, ratio) for ratio in (1, 2)}
    assert rows[1] < get_size(cfg)
    assert rows[2] < get_size(cfg) // 2
    for ratio in (0, -1):
        try:
            size(cfg, ratio)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive compression ratio accepted")
    print(json.dumps(dict(status="PASS", gpu_executed=False,
                          production_chunk_builder=True, workloads=workloads,
                          chunks=chunk_count, context=512000, logits_mb=512,
                          reserved_rows=rows,
                          scope="integer allocation bounds; not GPU accuracy or measured KV capacity")))


if __name__ == "__main__":
    main()
