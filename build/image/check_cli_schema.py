"""Parse deployment CLI on a GPU-less build host. Does not create an engine."""
import argparse
import json
from pathlib import Path
import sys

# CLI defaults require a platform even for --help. CPU selects defaults only;
# the deployment arguments remain the GPU arguments and no engine is created.
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform = CpuPlatform()
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

p = argparse.ArgumentParser()
p.add_argument('--argv', type=Path)
p.add_argument('--help-output', action='store_true')
options = p.parse_args()
parser = make_arg_parser(FlexibleArgumentParser())
if options.help_output:
    parser.parse_args(['--help=all'])
else:
    arguments = json.loads(options.argv.read_text())
    parsed = parser.parse_args(arguments)
    selected_model = parsed.model_tag or parsed.model
    assert selected_model == '/models/DeepSeek-V4.1-Flash', selected_model
    assert parsed.served_model_name == ['DeepSeek-V4.1-Flash'], parsed.served_model_name
    assert parsed.max_model_len == 262144 and parsed.max_num_seqs >= 32
    print(json.dumps({'status': 'PASS', 'scope': 'CLI parse only with CPU platform defaults; no engine or GPU validation',
                      'argument_count': len(arguments), 'model': selected_model,
                      'max_model_len': parsed.max_model_len, 'max_num_seqs': parsed.max_num_seqs}))
