#!/usr/bin/env python3
"""Audit per-case MAP output grids and fields; Python standard library only."""
import argparse
import array
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def record(handle, endian):
    marker = handle.read(4)
    if len(marker) != 4:
        raise ValueError('Missing record marker')
    size = struct.unpack(endian + 'i', marker)[0]
    if size < 0 or size > 2**31 - 1:
        raise ValueError('Unsupported record size')
    data = handle.read(size)
    if len(data) != size or handle.read(4) != marker:
        raise ValueError('Truncated or inconsistent Fortran record')
    return data


def numbers(data, kind, endian):
    values = array.array(kind)
    values.frombytes(data)
    if (sys.byteorder == 'little') != (endian == '<'):
        values.byteswap()
    return values


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect(path, grid):
    result = {'path': str(path), 'bytes': path.stat().st_size, 'blocks': []}
    with path.open('rb') as handle:
        marker = handle.read(4)
        endian = '<' if marker == b'\x04\x00\x00\x00' else '>'
        handle.seek(0)
        header = record(handle, endian)
        if len(header) != 4:
            raise ValueError('Expected a four-byte block-count record')
        count = struct.unpack(endian + 'i', header)[0]
        if not 1 <= count <= 10000:
            raise ValueError('Invalid block count')
        dims = record(handle, endian)
        width = 3 if grid else 4
        if len(dims) != count * width * 4:
            raise ValueError('Unexpected dimensions record')
        dims = struct.unpack(endian + 'i' * (count * width), dims)
        for index in range(count):
            shape = dims[index * width:(index + 1) * width]
            if any(x <= 0 for x in shape):
                raise ValueError('Nonpositive dimension')
            n = math.prod(shape[:3])
            data = record(handle, endian)
            channels = 3 if grid else shape[3]
            expected = n * channels * 8
            if len(data) != expected and not (grid and len(data) == expected + n * 4):
                raise ValueError(f'Block {index + 1}: unsupported data size {len(data)}')
            values = numbers(data[:expected], 'd', endian)
            if any(not math.isfinite(x) for x in values):
                raise ValueError(f'Block {index + 1}: NaN/Inf')
            block = {'shape': list(shape[:3]), 'points': n, 'channels': channels,
                     'ranges': [[min(values[c*n:(c+1)*n]), max(values[c*n:(c+1)*n])]
                                for c in range(channels)]}
            if grid and len(data) > expected:
                flags = numbers(data[expected:], 'i', endian)
                counts = {}
                for flag in flags:
                    counts[str(flag)] = counts.get(str(flag), 0) + 1
                block['integer_flags'] = counts
            result['blocks'].append(block)
        if handle.read(1):
            raise ValueError('Unexpected trailing records/data')
    result['sha256'] = digest(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Directory containing dataset_manifest.csv')
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    with (root / 'dataset_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r['sample'] for r in rows}) != len(rows):
        parser.error('Empty manifest or duplicate sample IDs')
    args.output.mkdir(parents=True, exist_ok=False)
    results, eligible = [], []
    for row in rows:
        sample = row['sample']
        result = {'sample': sample, 'level': row['level'], 'split': row['split'],
                  'status': 'failed', 'warnings': []}
        try:
            solution = root / row['solution_path'].replace('\\', '/')
            case = solution.parent.parent
            candidates = sorted(case.rglob('ofmeshplt.xyz'))
            # Some deliveries put output grids in the per-case mesh folder.
            mesh_folder = root / sample.split('_')[0] / 'mesh' / sample
            candidates = sorted(set(candidates + list(mesh_folder.rglob('ofmeshplt.xyz'))))
            if len(candidates) != 1:
                raise ValueError(f'Expected one per-case ofmeshplt.xyz; found {len(candidates)}')
            grid = inspect(candidates[0], True)
            field = inspect(solution, False)
            result.update(grid=grid, solution=field)
            if [b['shape'] for b in grid['blocks']] != [b['shape'] for b in field['blocks']]:
                raise ValueError('Grid/field block dimensions differ')
            if any(b['channels'] != 8 for b in field['blocks']):
                raise ValueError('Expected eight field channels')
            # Field order follows the supplied MAP field manifest. No data are modified.
            for block in field['blocks']:
                if block['ranges'][0][0] <= 0 or block['ranges'][4][0] <= 0:
                    raise ValueError('Nonpositive pressure/temperature under declared field order')
            visual = case / 'flow_visualization' / 'flow_field_manifest.json'
            if visual.exists():
                try:
                    metadata = json.loads(visual.read_text(encoding='utf-8-sig'))
                    for key, actual in [('grid_file', grid), ('solution_file', field)]:
                        old = metadata.get(key, {}).get('sha256')
                        if old and old != actual['sha256']:
                            result['warnings'].append(key + ': visualization SHA256 differs')
                except (ValueError, TypeError, AttributeError) as error:
                    result['warnings'].append('Cannot read visualization metadata: ' + str(error))
            result['status'] = 'passed'
            if row['level'] == 'A':
                eligible.append({'sample': sample, 'split': row['split'],
                                 'grid_path': grid['path'], 'solution_path': field['path'],
                                 'grid_sha256': grid['sha256'], 'solution_sha256': field['sha256']})
        except Exception as error:
            result['error'] = str(error)
        results.append(result)
        print(f"[{len(results)}/{len(rows)}] {sample}: {result['status']} {result.get('error', '')}", flush=True)
    (args.output / 'details.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (args.output / 'audit.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample', 'level', 'split', 'status', 'error', 'warnings'])
        writer.writeheader()
        for r in results:
            writer.writerow({k: '; '.join(r[k]) if k == 'warnings' else r.get(k, '') for k in writer.fieldnames})
    with (args.output / 'eligible_A.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample', 'split', 'grid_path', 'solution_path', 'grid_sha256', 'solution_sha256'])
        writer.writeheader()
        writer.writerows(eligible)
    summary = {'total': len(rows), 'passed': sum(r['status'] == 'passed' for r in results),
               'failed': sum(r['status'] == 'failed' for r in results), 'eligible_A': len(eligible),
               'note': 'Structural/finite-value audit only; not convergence certification. Integer flags retained without filtering. Baselines retain their original split.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
