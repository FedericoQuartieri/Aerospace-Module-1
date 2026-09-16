#!/usr/bin/env python3
"""Backend checks: full fields vs serial Schur, manufactured cases, residuals.

Environment: MPI/OMP/SIMD=0|1, RANKS=4, PROCESS_GRID="px py pz",
GRID=16 or "nx ny nz", GRIDS="16;17 19 13", PIPELINE_BATCHES="auto 1 3 64",
THREADS="1 4", PRECISION=double|float, STEPS=4, T_END=1, TIMEOUT=120,
TOLERANCE=1e-10 (double) or 1e-4 (float), MPIRUN, MPIEXEC_ARGS, CC/MPICC.
Results are isolated per invocation; binaries are isolated per configuration.
"""
import array
import json
import math
import os
from pathlib import Path
import shlex
import signal
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, log, env=None, timeout=120):
    with log.open('w') as output:
        output.write(shlex.join(map(str, command)) + '\n')
        output.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f'timeout: {log}') from None
    if code:
        raise RuntimeError(f'exit {code}: {log}\n' + log.read_text()[-4000:])


def build(backend, batch, mpi, omp, simd, flags, out):
    options = [f'TRIDIAG={backend}', f'PIPELINE_BATCH_LINES={"" if batch == "auto" else batch}',
               f'MPI={mpi}', f'OMP={omp}', f'SIMD={simd}', f'EXTRA_CPPFLAGS={flags}']
    compiler = os.getenv('MPICC') if mpi else os.getenv('CC')
    if compiler:
        options.append(f'CC={compiler}')
    directory = subprocess.check_output(['make', '-s', *options, 'print-test-dir'],
                                        cwd=ROOT, text=True).strip()
    targets = ['solver_snapshot', 'paper_man', 'zero_pressure', 'constant_forcing_man']
    if backend == 'pipeline':
        targets.append('pipeline_residual')
    path = Path(directory)
    run(['make', '-s', '-j'+os.getenv('BUILD_JOBS','2'), *options, *[str(path / name) for name in targets]],
        out / f'build-{backend}-{batch}-{mpi}-{omp}-{simd}.log', timeout=float(os.getenv('BUILD_TIMEOUT','600')))
    return ROOT / path


def load_snapshot(prefix, ranks, shape):
    nx, ny, nz = shape
    cells = nx * ny * nz
    values = array.array('d', [0]) * (11 * cells)
    occupied = bytearray(cells)
    for rank in range(ranks):
        data = Path(f'{prefix}.{rank}.bin').read_bytes()
        header = struct.unpack('=9i', data[:36])
        if tuple(header[:3]) != tuple(shape):
            raise AssertionError('wrong global dimensions')
        x, y, z, lx, ly, lz = header[3:]
        if min(x,y,z) < 0 or min(lx,ly,lz) < 1 or x+lx > nx or y+ly > ny or z+lz > nz:
            raise AssertionError('invalid block extent')
        local = array.array('d'); local.frombytes(data[36:])
        if len(local) != 11 * lx * ly * lz:
            raise AssertionError('wrong snapshot length')
        for f in range(11):
            for k in range(lz):
                for j in range(ly):
                    src = ((f*lz+k)*ly+j)*lx
                    dst = ((z+k)*ny+y+j)*nx+x
                    if f == 0:
                        if any(occupied[dst:dst+lx]):
                            raise AssertionError('overlapping blocks')
                        occupied[dst:dst+lx] = b'\1'*lx
                    values[f*cells+dst:f*cells+dst+lx] = local[src:src+lx]
    if not all(occupied) or not all(map(math.isfinite, values)):
        raise AssertionError('missing cells or nonfinite values')
    return values


def compare(reference, candidate, tolerance):
    if len(reference) != len(candidate):
        raise AssertionError('different field sizes')
    worst = 0.0
    for i, (a,b) in enumerate(zip(reference,candidate)):
        if not math.isfinite(a) or not math.isfinite(b):
            raise AssertionError(f'nonfinite field value {i}')
        error = abs(a-b) / max(1.0,abs(a),abs(b))
        worst = max(worst,error)
        if error > tolerance:
            raise AssertionError(f'field value {i}: {a:.17g} vs {b:.17g}, scaled error {error:.3e}')
    return worst


def main():
    mpi, omp, simd = [int(os.getenv(k, '0')) for k in ('MPI','OMP','SIMD')]
    if any(v not in (0,1) for v in (mpi,omp,simd)):
        raise ValueError('MPI, OMP and SIMD must be 0 or 1')
    precision = os.getenv('PRECISION','double')
    if precision not in ('double','float'):
        raise ValueError('PRECISION must be double or float')
    tolerance = float(os.getenv('TOLERANCE','1e-4' if precision == 'float' else '1e-10'))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('TOLERANCE must be finite and positive')
    batches = os.getenv('PIPELINE_BATCHES','auto 1 3 64').split()
    threads = list(map(int,os.getenv('THREADS', '1 ' + os.getenv('OMP_NUM_THREADS','4') if omp else '1').split()))
    threads = sorted(set(threads))
    if not batches or any(b != 'auto' and (not b.isdigit() or not 1 <= int(b) <= 1073741823) for b in batches) or not threads or min(threads) < 1:
        raise ValueError('invalid batch or thread count')
    if not omp and threads != [1]:
        raise ValueError('multiple threads require OMP=1')
    ranks = int(os.getenv('RANKS','4' if mpi else '1'))
    if ranks < 1 or (not mpi and ranks != 1):
        raise ValueError('RANKS > 1 requires MPI=1')
    shapes = [(ranks,1,1),(1,ranks,1),(1,1,ranks)] if mpi else [(1,1,1)]
    if mpi and ranks == 4: shapes += [(2,2,1),(1,2,2)]
    if mpi and ranks == 8: shapes += [(2,2,2)]
    if os.getenv('PROCESS_GRID'):
        shape = tuple(map(int,os.environ['PROCESS_GRID'].split()))
        if len(shape) != 3 or min(shape) < 1 or math.prod(shape) != ranks:
            raise ValueError('PROCESS_GRID must contain 3 positive integers with product RANKS')
        shapes = [shape]
    shapes = list(dict.fromkeys(shapes))
    grids = []
    for text in os.getenv('GRIDS',os.getenv('GRID','16')+';17 19 13').split(';'):
        grid = tuple(map(int,text.replace('x',' ').split()))
        if len(grid) == 1: grid *= 3
        if len(grid) != 3 or min(grid) < 2:
            raise ValueError('each grid needs three dimensions >= 2')
        grids.append(grid)
    steps = int(os.getenv('STEPS','4')); end = float(os.getenv('T_END','1.0'))
    timeout = float(os.getenv('TIMEOUT','120'))
    if steps < 1 or not math.isfinite(end) or end <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('STEPS, T_END and TIMEOUT must be positive')
    launcher = shlex.split(os.getenv('MPIRUN','mpirun')) + shlex.split(os.getenv('MPIEXEC_ARGS',''))
    base = ROOT / 'build/pipeline-check'; base.mkdir(parents=True,exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='run-',dir=base))
    print(f'Checks: {out}',flush=True)
    flags = os.getenv('EXTRA_CPPFLAGS','') + (' -DUSE_FLOAT' if precision == 'float' else '')
    # Manufactured executables retain a small compile-time grid; snapshots
    # and residual tests read every rectangular grid from the config file.
    flags += ' -DDEFAULT_WIDTH=16 -DDEFAULT_HEIGHT=16 -DDEFAULT_DEPTH=16'
    flags += f' -DDEFAULT_STEPS={steps} -DDEFAULT_T={end}'
    reference_bin = build('schur',64,0,0,0,flags,out)
    variants = [('schur',64,build('schur',64,mpi,omp,simd,flags,out))]
    variants += [('pipeline',b,build('pipeline',b,mpi,omp,simd,flags,out)) for b in batches]
    serial_env = os.environ | {'OMP_NUM_THREADS':'1','OMP_DYNAMIC':'FALSE'}
    checks = []
    sequence = 0
    for backend,batch,exe in variants:
        env = os.environ | {'OMP_NUM_THREADS':str(threads[-1]),'OMP_DYNAMIC':'FALSE'}
        for name in ('paper_man','zero_pressure','constant_forcing_man'):
            cmd = (launcher+['-n',str(ranks)] if mpi else [])+[str(exe/name)]
            run(cmd,out/f'manufactured-{backend}-{batch}-{name}.log',env,timeout)
    for grid in grids:
        config = out / ('grid-'+'x'.join(map(str,grid))+'.txt')
        config.write_text(f'width = {grid[0]}\nheight = {grid[1]}\ndepth = {grid[2]}\nsteps = {steps}\nt_end = {end}\n')
        for scenario in ('paper_variable','zero_pressure','constant_forcing'):
            sequence += 1; ref = out/f'ref-{sequence}'
            run([str(reference_bin/'solver_snapshot'),str(config),scenario,str(ref)],
                out/f'ref-{sequence}.log',serial_env,timeout)
            reference = load_snapshot(ref,1,grid)
            for backend,batch,exe in variants:
                for shape in shapes:
                    if any(p > n for p,n in zip(shape,grid)):
                        raise ValueError('process grid creates empty blocks')
                    # Schur needs an internal point before each interface.
                    # Pipeline supports one-cell blocks and is checked against
                    # the serial reference AND its own equation residual.
                    if backend == 'schur' and any(p > 1 and n//p < 2 for p,n in zip(shape,grid)):
                        continue
                    for thread in threads:
                        sequence += 1; got = out/f'candidate-{sequence}'
                        env = os.environ | {'OMP_NUM_THREADS':str(thread),'OMP_DYNAMIC':'FALSE'}
                        cmd = (launcher+['-n',str(ranks)] if mpi else [])
                        run(cmd+[str(exe/'solver_snapshot'),str(config),scenario,str(got),*map(str,shape)],
                            out/f'candidate-{sequence}.log',env,timeout)
                        error = compare(reference,load_snapshot(got,ranks,grid),tolerance)
                        checks.append(dict(grid=grid,scenario=scenario,backend=backend,batch=batch,
                                           ranks=ranks,shape=shape,threads=thread,error=error))
                        if backend == 'pipeline' and scenario == 'paper_variable':
                            run(cmd+[str(exe/'pipeline_residual'),str(config),*map(str,shape)],
                                out/f'residual-{sequence}.log',env,timeout)
        print(f'Passed grid {grid}; {len(checks)} field comparisons so far',flush=True)
    (out/'results.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(f'PASSED: {len(checks)} full-field comparisons, manufactured cases and pipeline residuals; '
          f'max scaled difference {max(c["error"] for c in checks):.3e}',flush=True)


if __name__ == '__main__':
    main()
