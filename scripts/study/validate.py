#!/usr/bin/env python3
"""Fail the numerical gate on missing/failed/nonfinite/different cases."""
import csv
import math
from pathlib import Path
import sys

KEYS = ('label','backend','batch','simd','omp','mpi','ranks','threads','nx','ny','nz','steps')


def validate(csv_path, expected_path, tolerance, phase='15_matrix_check'):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid tolerance')
    expected = {}
    for line in Path(expected_path).read_text().splitlines():
        key, shape = line.split('\t',1)
        expected[key] = shape.split()
    if not expected:
        raise ValueError('no expected cases')
    latest = {}
    with open(csv_path,newline='') as source:
        for row in csv.DictReader(source):
            if row['phase'] == phase:
                latest['|'.join(row[k] for k in KEYS)] = row
    selected = []
    for key,shape in expected.items():
        row = latest.get(key)
        if row is None or row['status'] != 'ok':
            raise ValueError(f'missing or failed case: {key}')
        if shape and shape != [row['px'],row['py'],row['pz']]:
            raise ValueError(f'wrong process grid: {key}')
        values = [float(row[k]) for k in ('l2_ux','l2_p')]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError(f'nonfinite/negative norm: {key}')
        selected.append((row,values))
    reference = next((v for row,v in selected if row['backend'] == 'schur' and row['mpi'] == '0' and row['omp'] == '0' and row['simd'] == '0' and row['ranks'] == '1'),None)
    if reference is None:
        raise ValueError('missing serial reference')
    for row,values in selected:
        if any(abs(a-b) > tolerance*max(1,abs(a),abs(b)) for a,b in zip(reference,values)):
            raise ValueError(f'norms differ: {row["label"]}: {values} vs {reference}')
    return len(selected)


if __name__ == '__main__':
    try:
        count = validate(sys.argv[1],sys.argv[2],float(sys.argv[3]))
        print(f'PASSED: all {count} expected cases succeeded with equivalent finite norms')
    except (ValueError,KeyError,OSError) as error:
        print(f'FAILED: {error}',file=sys.stderr)
        sys.exit(3)
