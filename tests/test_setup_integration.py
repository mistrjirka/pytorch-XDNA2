"""Check compiler selection from real device.py definitions without importing torch."""
import ast
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock,patch
from xdna_tools import toolchain


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'pkg/xdna_train'
        designs=self.root/'_designs';designs.mkdir(parents=True)
        for name in ('conv3x3w.py','conv3x3w_kernel.cc'): (designs/name).write_text('source')
        env=patch.dict(os.environ,{'TORCH_XDNA2_CACHE':str(Path(self.temp.name)/'cache')},clear=True)
        env.start();self.addCleanup(env.stop)
        tree=ast.parse((Path(__file__).resolve().parents[1]/'xdna_train/device.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_configure_iron_family']
        self.state={'__name__':'test_backend.device','__package__':'test_backend','_ROOT':self.root,
                    'Path':Path,'os':os,'sys':sys,'hashlib':hashlib,'_iron_python':Mock(return_value='')}
        fake=types.ModuleType('test_backend._build');fake._cache_root=toolchain.cache_root
        modules=patch.dict(sys.modules,{'test_backend._build':fake});modules.start();self.addCleanup(modules.stop)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<device-selection>','exec'),self.state)
        self.fn=self.state['_configure_iron_family'];self.configure=Mock()

    def test_empty_explicit_disables_auto(self):
        os.environ['XDNA_IRON_PYTHON']=''
        with patch.object(toolchain,'automatic_python') as auto,patch.object(toolchain,'managed_python') as managed:
            self.fn('conv3x3w',self.configure,20_000_000)
        auto.assert_not_called();managed.assert_not_called();self.configure.assert_not_called()

    def test_explicit_external_precedes_managed(self):
        os.environ['XDNA_IRON_PYTHON']='/custom/python';self.state['_iron_python'].return_value='/custom/python'
        with patch.object(toolchain,'automatic_python') as auto,patch.object(toolchain,'managed_python') as managed:
            self.fn('conv3x3w',self.configure,20_000_000)
        auto.assert_not_called();managed.assert_not_called()
        self.assertIn('/custom/python',self.configure.call_args.args[1]);self.assertNotIn('_run_iron',self.configure.call_args.args[1])

    def test_existing_external_no_download(self):
        self.state['_iron_python'].return_value='/legacy/python'
        with patch.object(toolchain,'automatic_python') as auto,patch.object(toolchain,'managed_python',return_value=None):
            self.fn('conv3x3w',self.configure,20_000_000)
        auto.assert_not_called();self.assertIn('/legacy/python',self.configure.call_args.args[1])

    def test_managed_compiler_wrapper(self):
        with patch.object(toolchain,'automatic_python') as auto,patch.object(toolchain,'managed_python',return_value=Path('/managed dir/env/bin/python')):
            self.fn('conv3x3w',self.configure,20_000_000)
        auto.assert_not_called();cmd=shlex.split(self.configure.call_args.args[1])
        self.assertIn('/managed dir/env/bin/python',cmd);self.assertTrue(any(p.endswith('_run_iron.py') for p in cmd))

    def test_cold_install_automatically(self):
        with patch.object(toolchain,'managed_python',return_value=None),patch.object(toolchain,'automatic_python',return_value=Path('/managed/env/bin/python')) as auto:
            self.fn('conv3x3w',self.configure,20_000_000)
        auto.assert_called_once();self.configure.assert_called_once()

    def test_failed_setup_keeps_cpu(self):
        with patch.object(toolchain,'managed_python',return_value=None),patch.object(toolchain,'automatic_python',return_value=None):
            self.fn('conv3x3w',self.configure,20_000_000)
        self.configure.assert_not_called()

    def test_managed_and_external_artifacts_separate(self):
        self.state['_iron_python'].return_value='/legacy/python'
        with patch.object(toolchain,'managed_python',return_value=None):self.fn('conv3x3w',self.configure,1)
        external=self.configure.call_args.args[0]
        with patch.object(toolchain,'managed_python',return_value=Path('/managed/python')):self.fn('conv3x3w',self.configure,1)
        self.assertNotEqual(self.configure.call_args.args[0],external)

    def test_cached_program_without_compiler(self):
        cache=Path(self.temp.name)/'artifact';cache.mkdir();(cache/'conv3x3w.xclbin').write_bytes(b'x')
        os.environ.update(XDNA_IRON_PYTHON='',XDNA_CONV3X3W_DIR=str(cache))
        self.fn('conv3x3w',self.configure,20_000_000)
        self.assertEqual(self.configure.call_args.args[1],'')


if __name__=='__main__':unittest.main()
