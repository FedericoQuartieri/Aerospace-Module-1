#!/usr/bin/env python3
"""Regression tests for the numerical gate and snapshot comparison reader."""
import csv
import os
import shlex
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


gate = load('gate',ROOT/'scripts/study/validate.py')
check = load('check',ROOT/'scripts/check_pipeline.py')


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.csv = self.directory/'results.csv'; self.expected = self.directory/'expected.tsv'
        self.reference = dict(zip(gate.KEYS,('serial reference','schur','64','0','0','0','1','1','8','8','8','4')))
        self.reference.update(phase='15_matrix_check',status='ok',px='1',py='1',pz='1',l2_ux='0.01',l2_p='0.1')
        self.candidate = self.reference | dict(label='pipeline',backend='pipeline',mpi='1',ranks='2',px='2')
        self.expected.write_text(''.join('|'.join(r[k] for k in gate.KEYS)+'\t'+
            ' '.join(r[k] for k in ('px','py','pz'))+'\n' for r in (self.reference,self.candidate)))

    def write(self,*rows):
        with self.csv.open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(self.reference));writer.writeheader();writer.writerows(rows)

    def test_small_roundoff_is_accepted(self):
        self.write(self.reference,self.candidate | dict(l2_p='0.10000000000001'))
        self.assertEqual(gate.validate(self.csv,self.expected,1e-10),2)

    def test_missing_failed_nonfinite_and_wrong_shape_are_rejected(self):
        candidates = [None,dict(status='timeout'),dict(l2_p='nan'),dict(l2_p='inf'),
                      dict(l2_p='0.3'),dict(px='1',py='2')]
        for change in candidates:
            with self.subTest(change=change):
                self.write(self.reference,*([] if change is None else [self.candidate | change]))
                with self.assertRaises(ValueError): gate.validate(self.csv,self.expected,1e-10)

    def test_retry_replaces_failed_result(self):
        self.write(self.reference,self.candidate | dict(status='timeout'),self.candidate)
        self.assertEqual(gate.validate(self.csv,self.expected,1e-10),2)

    def test_no_expected_cases_is_failure(self):
        self.write(self.reference); self.expected.write_text('')
        with self.assertRaises(ValueError): gate.validate(self.csv,self.expected,1e-10)

    def test_nonfinite_fields_are_rejected(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(AssertionError): check.compare([0],[value],1e-10)

    def test_equal_norm_does_not_hide_different_fields(self):
        with self.assertRaises(AssertionError): check.compare([1,0],[0,1],1e-10)


class BuildTests(unittest.TestCase):
    def test_switching_configuration_and_returning_does_not_reuse_alias(self):
        # Exercise the real Makefile with a tiny compiler fixture: the output
        # records the requested backend/flags instead of compiling the solver.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(ROOT/'Makefile',root/'Makefile')
            for path in ('src/main.c','src/tridiag/schur/backend.c',
                         'src/tridiag/pipeline/backend.c','test/bench.c','include/config.h'):
                file = root/path; file.parent.mkdir(parents=True,exist_ok=True); file.write_text('')
            compiler = root/'compiler'
            compiler.write_text("#!"+sys.executable+"\n"+
                "import pathlib,sys,shlex\n"
                "if '--version' in sys.argv: print('fixture compiler 1'); sys.exit()\n"
                "out=pathlib.Path(sys.argv[sys.argv.index('-o')+1])\n"
                "flags=' '.join(v for v in sys.argv if v.startswith('-D'))\n"
                "out.write_text('#!/bin/sh\\nprintf %s '+shlex.quote(flags)+'\\n')\n")
            compiler.chmod(0o755)
            def build(backend,batch):
                subprocess.run(['make','-s','-j2',f'CC={compiler}',f'TRIDIAG={backend}',
                    f'PIPELINE_BATCH_LINES={batch}','solver','build/tests/bench'],
                    cwd=root,check=True,capture_output=True)
                outputs = [subprocess.check_output([str(root/name)],text=True)
                           for name in ('solver','build/tests/bench')]
                self.assertTrue(all(f'-DTRIDIAG_{backend.upper()}' in value for value in outputs))
                return outputs
            pipeline = build('pipeline',3)
            build('schur',64)
            self.assertEqual(build('pipeline',3),pipeline)
            self.assertNotEqual(build('pipeline',7),pipeline)

    def test_study_initialization_is_safe_across_phases(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = 'source '+shlex.quote(str(ROOT/'scripts/study/lib.sh'))+'; study_begin "$PHASE"'
            processes = [subprocess.Popen(['bash','-c',script],stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,env=os.environ | {'STUDY_BASE':temporary,
                'PHASE':f'phase-{i}','DRY_RUN':'1'}) for i in range(4)]
            outputs = [process.communicate(timeout=60) for process in processes]
            for process,(stdout,stderr) in zip(processes,outputs):
                self.assertEqual(process.returncode,0,stdout+stderr)
            self.assertEqual(len((Path(temporary)/'source-toolchain.id').read_text().strip()),20)

    def test_source_identity_detects_uncommitted_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ('scripts/study','src','include','test'):
                (root/folder).mkdir(parents=True)
            script = root/'scripts/study/source_id.py'
            shutil.copy2(ROOT/'scripts/study/source_id.py',script)
            (root/'Makefile').write_text('original')
            def identity(): return subprocess.check_output([sys.executable,str(script)],text=True)
            before = identity()
            self.assertEqual(identity(),before)
            (root/'src/new.c').write_text('uncommitted source')
            self.assertNotEqual(identity(),before)


if __name__ == '__main__': unittest.main()
