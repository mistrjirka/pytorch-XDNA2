"""Exercise provisioning with fake wheel installation in private temporary dirs."""
from contextlib import redirect_stdout,redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from xdna_tools import toolchain as t


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'cache'
        p=patch.dict(os.environ,{'TORCH_XDNA2_CACHE':str(self.root)},clear=True);p.start();self.addCleanup(p.stop)
        for name,value in [('_attempted',False),('_uv_command',['uv','--no-config'])]:
            p=patch.object(t,name,return_value=value) if name=='_uv_command' else patch.object(t,name,value)
            p.start();self.addCleanup(p.stop)
        p=patch.object(t,'_supported_host');p.start();self.addCleanup(p.stop)
        self.commands=[]
        self.err=io.StringIO();p=redirect_stderr(self.err);p.__enter__();self.addCleanup(p.__exit__,None,None,None)

    def fake_run(self,command,env,log,timeout=900):
        self.commands.append((command,env))
        v=t.toolchain_dir()/'env'
        for p in [v/'bin/python',t._site(v)/'mlir_aie/python/aie/__init__.py',t._site(v)/'llvm-aie/bin/clang']:
            p.parent.mkdir(parents=True,exist_ok=True);p.touch();p.chmod(0o755)

    def install(self):
        with patch.object(t,'_run',side_effect=self.fake_run),patch.object(t,'_probe'):
            return t.ensure_toolchain()

    def test_success_and_isolation(self):
        os.environ.update(VIRTUAL_ENV='/user/application',PYTHONPATH='/user/lib',UV_PROJECT_ENVIRONMENT='/user/env')
        before=os.environ.copy();python=self.install()
        self.assertEqual(python,t.toolchain_dir()/'env/bin/python');self.assertEqual(os.environ.copy(),before)
        install=[(c,e) for c,e in self.commands if 'pip' in c][0]
        self.assertIn('--require-hashes',install[0]);self.assertIn('--only-binary',install[0]);self.assertIn(str(python),install[0])
        self.assertNotIn('VIRTUAL_ENV',install[1]);self.assertNotIn('PYTHONPATH',install[1]);self.assertNotIn('UV_PROJECT_ENVIRONMENT',install[1])

    def test_repeat_uses_cache(self):
        python=self.install()
        with patch.object(t,'_run',side_effect=AssertionError('unexpected install')):
            self.assertEqual(t.ensure_toolchain(),python)

    def test_offline_cache_works(self):
        python=self.install();os.environ['UV_OFFLINE']='1'
        self.assertEqual(t.ensure_toolchain(),python)

    def test_offline_cold_no_write(self):
        os.environ['UV_OFFLINE']='true'
        with self.assertRaisesRegex(t.SetupError,'OFFLINE'):t.ensure_toolchain()
        self.assertFalse(self.root.exists())

    def test_auto_disabled_no_write(self):
        os.environ['XDNA_AUTO_SETUP']='0'
        with patch.object(t,'ensure_toolchain',side_effect=AssertionError('download')):
            self.assertIsNone(t.automatic_python())
        self.assertFalse(self.root.exists())

    def test_setup_check_no_write(self):
        with redirect_stdout(io.StringIO()):self.assertEqual(t.main(['--check']),1)
        self.assertFalse(self.root.exists())

    def test_invalid_marker(self):
        python=self.install();marker=t.toolchain_dir()/'ready.json';marker.write_text('badjson')
        self.assertIsNone(t.managed_python());self.assertTrue(python.exists())

    def test_missing_binary_not_ready(self):
        self.install();(t._site(t.toolchain_dir()/'env')/'llvm-aie/bin/clang').unlink()
        self.assertIsNone(t.managed_python())

    def test_pins_change_cache_key(self):
        self.install();old=t.toolchain_dir()
        new=Path(self.tmp.name)/'new-lock.txt';new.write_text(t.REQUIREMENTS.read_text()+'\n# updated pin\n')
        with patch.object(t,'REQUIREMENTS',new):
            self.assertNotEqual(t.toolchain_dir(),old);self.assertIsNone(t.managed_python())

    def test_unknown_environment_untouched(self):
        v=t.toolchain_dir()/'env';v.mkdir(parents=True);(v/'user-file').write_text('keep')
        with self.assertRaisesRegex(t.SetupError,'unrecognized'):t.ensure_toolchain()
        self.assertEqual((v/'user-file').read_text(),'keep')

    def test_failure_and_cooldown(self):
        with patch.object(t,'_run',side_effect=t.SetupError('simulated failed wheel')):
            with self.assertRaisesRegex(t.SetupError,'failed'):t.ensure_toolchain()
        self.assertFalse((t.toolchain_dir()/'ready.json').exists())
        with patch.object(t,'_run',side_effect=AssertionError('unwanted retry')):
            with self.assertRaisesRegex(t.SetupError,'Recent'):t.ensure_toolchain()
        with patch.object(t,'_run',side_effect=self.fake_run),patch.object(t,'_probe'):
            self.assertTrue(t.ensure_toolchain(retry=True).exists())

    def test_automatic_warn_once(self):
        with patch.object(t,'ensure_toolchain',side_effect=t.SetupError('network down')) as fn:
            with self.assertWarnsRegex(RuntimeWarning,'CPU fallback'):self.assertIsNone(t.automatic_python())
            self.assertIsNone(t.automatic_python());self.assertEqual(fn.call_count,1)

    def test_corrupt_failure_timestamp(self):
        base=t.toolchain_dir();base.mkdir(parents=True);t._atomic_json(base/'failed.json',{'time':'oops'})
        self.assertTrue(self.install().exists())

    def test_lock_timeout(self):
        with t._setup_lock(t.toolchain_dir()):
            with self.assertRaisesRegex(t.SetupError,'Another'):
                with t._setup_lock(t.toolchain_dir(),timeout=0.01):self.fail('lock acquired twice')

    def test_repair_removes_stale_ready_before_install(self):
        self.install();(t._site(t.toolchain_dir()/'env')/'llvm-aie/bin/clang').unlink()
        def run(c,e,l,timeout=900):
            self.assertFalse((t.toolchain_dir()/'ready.json').exists());self.fake_run(c,e,l)
        with patch.object(t,'_run',side_effect=run),patch.object(t,'_probe'):
            t.ensure_toolchain()

    def test_compiler_environment_has_no_empty_library_entry(self):
        from xdna_tools import diagnostics as d
        os.environ['LD_LIBRARY_PATH']=':/custom/lib:'
        with patch.object(d,'find_xrt_paths',return_value=(None,None)):
            env=t.compiler_environment(t.toolchain_dir()/'env')
        self.assertEqual(env['LD_LIBRARY_PATH'],'/custom/lib')

    def test_bundled_llvm_precedes_system_libraries(self):
        from xdna_tools import diagnostics as d
        v=t.toolchain_dir()/'env'
        llvm=t._site(v)/'llvm-aie/lib';mlir=t._site(v)/'mlir_aie/lib';xrt=Path(self.tmp.name)/'xrt/lib'
        for directory in (llvm,mlir,xrt):directory.mkdir(parents=True)
        os.environ['LD_LIBRARY_PATH']='/usr/lib'
        with patch.object(d,'find_xrt_paths',return_value=(None,xrt)):
            result=t.compiler_environment(v)
        self.assertEqual(result['LD_LIBRARY_PATH'].split(os.pathsep),[str(llvm),str(mlir),str(xrt),'/usr/lib'])


class PlatformTests(unittest.TestCase):
    def test_reject_other_os(self):
        with patch.object(t.sys,'platform','win32'),self.assertRaises(t.SetupError):t._supported_host()
    def test_reject_old_glibc(self):
        with patch.object(t.sys,'platform','linux'),patch.object(t.platform,'machine',return_value='x86_64'),patch.object(t.platform,'libc_ver',return_value=('glibc','2.31')),self.assertRaises(t.SetupError):t._supported_host()
    def test_reject_musl(self):
        with patch.object(t.sys,'platform','linux'),patch.object(t.platform,'machine',return_value='x86_64'),patch.object(t.platform,'libc_ver',return_value=('musl','1.2')),self.assertRaises(t.SetupError):t._supported_host()
    def test_supported(self):
        with patch.object(t.sys,'platform','linux'),patch.object(t.platform,'machine',return_value='x86_64'),patch.object(t.platform,'libc_ver',return_value=('glibc','2.35')):t._supported_host()


if __name__=='__main__':unittest.main()
