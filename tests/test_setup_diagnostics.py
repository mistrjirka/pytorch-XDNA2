"""No NPU, package installation or PyTorch import required."""
from contextlib import redirect_stdout
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from xdna_tools import diagnostics as d


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = patch.dict(os.environ, {}, clear=True); env.start(); self.addCleanup(env.stop)
        for key, value in {'_SYSTEM_INCLUDE_DIRS':(str(self.root/'include'),),
                           '_SYSTEM_LIB_DIRS':(str(self.root/'lib'), str(self.root/'lib64')),
                           '_OPT_XRT':str(self.root/'opt/xilinx/xrt'),
                           '_DEVICE_DIR':str(self.root/'dev/accel')}.items():
            p=patch.object(d,key,value);p.start();self.addCleanup(p.stop)

    def make(self, name):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch();return p

    def test_usr_paths(self):
        self.make('include/xrt/xrt_device.h');self.make('lib/libxrt_coreutil.so')
        self.assertEqual(d.find_xrt_paths(),(self.root/'include',self.root/'lib'))

    def test_lib64(self):
        self.make('include/xrt/xrt_device.h');self.make('lib64/libxrt_coreutil.so')
        self.assertEqual(d.find_xrt_paths()[1],self.root/'lib64')

    def test_opt(self):
        self.make('opt/xilinx/xrt/include/xrt/xrt_device.h');self.make('opt/xilinx/xrt/lib/libxrt_coreutil.so')
        self.assertEqual(d.find_xrt_paths(),(self.root/'opt/xilinx/xrt/include',self.root/'opt/xilinx/xrt/lib'))

    def test_explicit_invalid_does_not_fallback(self):
        self.make('include/xrt/xrt_device.h');self.make('lib/libxrt_coreutil.so')
        os.environ['XRT_INCLUDE_DIR']=str(self.root/'missing');os.environ['XRT_LIB_DIR']=str(self.root/'missing')
        self.assertEqual(d.find_xrt_paths(),(None,None))
        self.assertIn('not overridden',d._check_xrt_headers()['hint'])

    def test_custom_root(self):
        self.make('custom/include/xrt/xrt_device.h');self.make('custom/lib64/libxrt_coreutil.so')
        os.environ['XRT_ROOT']=str(self.root/'custom')
        self.assertEqual(d.find_xrt_paths(),(self.root/'custom/include',self.root/'custom/lib64'))

    def test_versioned_library_insufficient(self):
        self.make('lib/libxrt_coreutil.so.2')
        self.assertIsNone(d.find_xrt_paths()[1]);self.assertIn('-dev/devel',d._check_xrt_library()['detail'])

    def test_directory_not_library(self):
        (self.root/'lib/libxrt_coreutil.so').mkdir(parents=True)
        self.assertIsNone(d.find_xrt_paths()[1])

    def test_missing_torch(self):
        with patch.object(d.importlib.metadata,'version',side_effect=importlib.metadata.PackageNotFoundError):
            self.assertEqual(d._check_torch()['status'],'error')

    def test_torch_versions(self):
        for version,good in [('2.14.1+rocm7.2',True),('2.14.0',True),('2.140.0',False),('2.14.0rc1',False),('2.13.1',False)]:
            with self.subTest(version=version),patch.object(d.importlib.metadata,'version',return_value=version):
                self.assertEqual(d._check_torch()['status']=='ok',good)

    def test_no_device(self):
        r=d._check_device();self.assertEqual(r['name'],'npu-device');self.assertTrue(d._blocking([r]))

    def test_device_permission(self):
        self.make('dev/accel/accel0')
        with patch.object(d.os,'access',return_value=False):
            self.assertEqual(d._check_device()['status'],'error')

    def test_accessible_device(self):
        self.make('dev/accel/accel0')
        self.assertEqual(d._check_device()['status'],'ok')

    def test_compiler_wrapper(self):
        os.environ['CXX']='ccache g++'
        with patch.object(d.shutil,'which',side_effect=lambda s:'/bin/'+s if s=='ccache' else None):
            self.assertEqual(d._check_compiler()['status'],'error')
        with patch.object(d.shutil,'which',side_effect=lambda s:'/bin/'+s):
            self.assertEqual(d._check_compiler()['status'],'ok')

    def test_aggregates_missing(self):
        values=[d._entry('compiler','error','missing','install compiler'),d._entry('torch','error','absent','install torch')]
        with patch.object(d,'probe_dependencies',return_value=values),self.assertRaisesRegex(RuntimeError,'compiler.*'):
            d.require_runtime_dependencies()

    def test_doctor_json_exit(self):
        for status,exit_code in [('ok',0),('error',1)]:
            output=io.StringIO()
            with patch.object(d,'probe_dependencies',return_value=[d._entry('thing',status,'detail')]),redirect_stdout(output):
                self.assertEqual(d.main(['--json']),exit_code)
            self.assertEqual(json.loads(output.getvalue())[0]['status'],status)

    def test_import_is_inert(self):
        code="import sys;import xdna_tools.diagnostics,xdna_tools.toolchain;assert 'torch' not in sys.modules;assert 'xdna_train' not in sys.modules"
        env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]);env['TORCH_XDNA2_CACHE']=str(self.root/'emptycache')
        subprocess.run([sys.executable,'-c',code],env=env,check=True)
        self.assertFalse((self.root/'emptycache').exists())


if __name__=='__main__':unittest.main()
