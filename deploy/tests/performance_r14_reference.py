"""Load production metadata code with no model; share CPU/GPU input fixtures."""
import __future__
import ast
from pathlib import Path
import sys
import sysconfig
from types import ModuleType

import torch
import triton
import triton.language as tl

SITE = Path(sysconfig.get_paths()['purelib'])
ROCM = 'vllm/models/deepseek_v4_1/amd/rocm.py'


def load_pack(path=None, name='r14_production_pack'):
    path = Path(path) if path else SITE / ROCM
    names = {'_build_indptr_from_lengths', '_compute_topk_lens_kernel',
             '_pack_global_topk_ragged_kernel', '_pack_global_topk_ragged_fused_kernel',
             'compute_global_topk_ragged_indices_and_indptr'}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, triton=triton, tl=tl)
    sys.modules[name] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), module.__dict__)
    return module


def ragged_inputs(rows, width=512, device='cpu', block_size=32, all_padding=False):
    # Noncontiguous top-k input exercises flatten/contiguous normalization;
    # block-table row stride stays larger than the visible width.
    raw = torch.full((rows, width * 2), -1, dtype=torch.int32, device=device)
    indices = raw[:, ::2]
    r = torch.arange(rows, device=device)[:, None]
    c = torch.arange(width, device=device)[None, :]
    lengths = (r * 71 + 13) % (width + 1)
    if rows:
        lengths[-1] = width
        if rows > 1:
            lengths[0] = 0
    indices.copy_(torch.where(c < lengths, (c * 7 + r * 3) % (32 * block_size), -1).int())
    mapping = torch.arange(rows, dtype=torch.int32, device=device) % 3
    table = torch.arange(192, dtype=torch.int32, device=device).view(3, 64)[:, :32] + 100
    # Adding above makes a compact result; restore a genuinely strided table.
    storage = torch.empty((3, 64), dtype=torch.int32, device=device)
    storage[:, :32].copy_(table)
    table = storage[:, :32]
    valid = torch.arange(rows, device=device) % 4 != 1
    if all_padding:
        valid.zero_()
        mapping.fill_(1000000)  # A padded row must not dereference its request.
    return indices, mapping, table, block_size, valid


def mutate_inputs(args):
    indices, mapping, table, block_size, valid = args
    indices.copy_(torch.where(indices >= 0, (indices + 1) % (32 * block_size), indices))
    table.add_(7)
    valid.logical_not_()
    mapping.copy_(torch.arange(mapping.numel(), dtype=torch.int32, device=mapping.device) % 3)


def metadata_prefix(path, cls, kind):
    """Execute the actual changed builder prefix, stopping before unrelated kernels.

    Indexer stops at the first prefill assertion. SWA stops after validity is
    written. This tests the production branch and persistent output addresses;
    it does not claim to execute an entire attention builder on a CPU.
    """
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
    prefix = []
    for statement in method.body:
        if kind == 'indexer' and isinstance(statement, ast.If) and ast.unparse(statement.test) == 'num_prefills > 0':
            statement.body = statement.body[:next(i for i, s in enumerate(statement.body) if isinstance(s, ast.Assert))]
            # The original variant has no statements before its assertion.
            if not statement.body:
                statement.body = [ast.Pass()]
            prefix.append(statement)
            break
        if kind == 'swa' and isinstance(statement, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'non_causal' for t in statement.targets):
            break
        prefix.append(statement)
    else:
        raise ValueError('Builder prefix boundary changed')
    method.body = prefix + [ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]))]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    return module
