"""Require eight actual rank successes and fail silently skipped CUDA operator tests."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def basic(directory):
    files = sorted(directory.glob('rank-*.json'))
    assert {p.name for p in files} == {f'rank-{i}.json' for i in range(8)}, 'Need exactly eight rank results'
    for rank in range(8):
        row = json.loads((directory/f'rank-{rank}.json').read_text())
        assert row['status'] == 'PASS' and row['rank'] == rank, row
        assert row['world_size'] == 8 and row['device_index'] == rank and row['sm'] == [8, 0], row
        for key in ['uva_cpu_mutation', 'bf16_matmul', 'nccl_eager_all_reduce',
                    'nccl_graph_all_reduce', 'graph_output_reset_before_replay']:
            assert row.get(key) is True, (rank, key)
    supervisor = json.loads((directory/'supervisor.json').read_text())
    assert supervisor['status'] == 'PROCESS_EXITED' and supervisor['exit_code'] == 0, supervisor
    return {'status': 'PASS', 'ranks': 8, 'scope': 'actual SM80/UVA/BF16/NCCL eager and graph checks on all ranks'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basic-only', action='store_true')
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = basic(args.directory)
    (args.directory/'basic-operators.json').write_text(json.dumps(report, indent=2)+'\n')
    if not args.basic_only:
        counts = {}
        for name in ['engram', 'fp8-sm80']:
            root = ET.parse(args.directory/(name+'.xml')).getroot()
            values = {key: sum(int(s.attrib.get(key, 0)) for s in root.iter('testsuite'))
                      for key in ['tests', 'failures', 'errors', 'skipped']}
            assert values['tests'] > 0 and all(values[k] == 0 for k in ['failures', 'errors', 'skipped']), values
            counts[name] = values
        report.update(scope='eight-rank basic probe plus selected Engram/FP8 SM80 operator tests', counts=counts)
        (args.directory/'selected-operators.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
