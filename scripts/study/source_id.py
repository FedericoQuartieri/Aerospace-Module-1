#!/usr/bin/env python3
"""Content/toolchain identity for study binaries and resumable results."""
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess

root = Path(__file__).resolve().parents[2]
hash_ = hashlib.sha256()
files = [root/'Makefile']
for directory in ('src','include','test','scripts'):
    files += [p for p in (root/directory).rglob('*') if p.suffix in ('.c','.h','.sh','.py')]
for path in sorted(files):
    hash_.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes()+b'\0')
revision = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True)
hash_.update(revision.stdout)
hash_.update((platform.system()+' '+platform.machine()).encode())
for name in ('CC','MPICC','CFLAGS','CPPFLAGS','EXTRA_CFLAGS','EXTRA_CPPFLAGS',
             'ZETA_SIMD_VECTORS','U_SIMD_VECTORS','OMP_SPLIT','MAKEFLAGS','BENCH_SCENARIO'):
    hash_.update((name+'='+os.getenv(name,'')).encode()+b'\0')
for command in (os.getenv('CC','cc'),os.getenv('MPICC','mpicc')):
    argv = shlex.split(command)
    hash_.update((shutil.which(argv[0]) or command).encode())
    try:
        version = subprocess.run(argv+['--version'],capture_output=True,timeout=10)
        hash_.update(version.stdout+version.stderr)
    except (OSError,subprocess.TimeoutExpired):
        hash_.update(b'unavailable')
print(hash_.hexdigest()[:20])
