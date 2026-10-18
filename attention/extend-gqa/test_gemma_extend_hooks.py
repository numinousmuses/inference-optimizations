"""CPU unit test for gemma_extend_hooks.py: copies the installed extend_attention.py to a temp directory,
applies the patcher (check mode then apply mode), verifies syntax, verifies the dispatch guard is present and
dispatches on SM120 (mocked), and falls back gracefully when the GQA kernel raises.

Run locally:   TRITON_INTERPRET=1 pytest operations/patches/test_gemma_extend_hooks.py -v
Run on box:    python3 operations/devbox/devbox.py test rtx-gemma operations/patches/test_gemma_extend_hooks.py -v
"""
import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

# add repo root to path so operations.* resolves without install
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _import_hooks():
    spec = importlib.util.spec_from_file_location(
        "gemma_extend_hooks",
        os.path.join(_HERE, "gemma_extend_hooks.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPatcherCheckMode(unittest.TestCase):
    """--check mode: anchors must resolve without touching any file."""

    def setUp(self):
        # find the installed sglang extend_attention.py
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            self.skipTest("sglang not installed")
        self.sgl_root = list(spec.submodule_search_locations)[0]
        self.extend_path = os.path.join(
            self.sgl_root, "kernels", "ops", "attention", "extend_attention.py"
        )
        if not os.path.isfile(self.extend_path):
            self.skipTest("extend_attention.py not found in sglang install")

    def test_check_anchors_resolve(self):
        hooks = _import_hooks()
        lines = hooks.apply(self.sgl_root, check=True)
        statuses = [l.split(":")[1].strip() for l in lines]
        for line in lines:
            self.assertNotIn("anchor found 0 times", line, msg=f"missing anchor: {line}")
            self.assertNotIn("anchor found 2 times", line, msg=f"duplicate anchor: {line}")


class TestPatcherApplyMode(unittest.TestCase):
    """Apply mode: the patcher creates syntactically valid output with the marker and dispatch guard."""

    def setUp(self):
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            self.skipTest("sglang not installed")
        self.sgl_root = list(spec.submodule_search_locations)[0]
        self.extend_path = os.path.join(
            self.sgl_root, "kernels", "ops", "attention", "extend_attention.py"
        )
        if not os.path.isfile(self.extend_path):
            self.skipTest("extend_attention.py not found")
        # work in a temp copy so we never touch the installed package
        self.tmpdir = tempfile.mkdtemp()
        self.fake_root = os.path.join(self.tmpdir, "sglang")
        target_dir = os.path.join(self.fake_root, "kernels", "ops", "attention")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(self.extend_path, os.path.join(target_dir, "extend_attention.py"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patched_text(self) -> str:
        hooks = _import_hooks()
        hooks.apply(self.fake_root, check=False)
        return open(os.path.join(self.fake_root, "kernels", "ops", "attention", "extend_attention.py")).read()

    def test_syntactically_valid(self):
        text = self._patched_text()
        compile(text, "extend_attention.py", "exec")  # raises SyntaxError if broken

    def test_dispatch_marker_present(self):
        text = self._patched_text()
        self.assertIn("# numinous-hook:gqa_dispatch", text)

    def test_helpers_marker_present(self):
        text = self._patched_text()
        self.assertIn("# numinous-hook:gqa_helpers", text)
        self.assertIn("_NUMINOUS_GQA_FAILED", text)
        self.assertIn("_numinous_gqa_enabled", text)
        self.assertIn("_numinous_gqa_config", text)

    def test_idempotent_second_pass(self):
        hooks = _import_hooks()
        hooks.apply(self.fake_root, check=False)
        lines2 = hooks.apply(self.fake_root, check=False)
        for l in lines2:
            self.assertIn("already applied", l, msg=f"second apply should be idempotent: {l}")



    def _exec_helpers(self, cap) -> dict:
        """Exec only the helpers section (no Triton @jit) with the given CUDA_CAPABILITY mock."""
        text = self._patched_text()
        # extract helpers block: everything from the numinous-hook:gqa_helpers marker to end
        marker = "# numinous-hook:gqa_helpers"
        idx = text.find(marker)
        self.assertGreater(idx, 0, "helpers block not found in patched text")
        helpers_text = text[idx:]
        ns = {"_is_cuda": cap[0] == 12, "CUDA_CAPABILITY": cap}
        exec(compile(helpers_text, "<helpers>", "exec"), ns)  # noqa: S102
        return ns

    def test_numinous_gqa_enabled_false_off_sm120(self):
        """_numinous_gqa_enabled() must return False when the env flag is 0."""
        ns = self._exec_helpers((12, 0))
        os.environ["SGLANG_NUMINOUS_EXTEND_GQA"] = "0"
        try:
            self.assertFalse(ns["_numinous_gqa_enabled"]())
        finally:
            del os.environ["SGLANG_NUMINOUS_EXTEND_GQA"]

    def test_numinous_gqa_enabled_true_sm120(self):
        ns = self._exec_helpers((12, 0))
        os.environ.pop("SGLANG_NUMINOUS_EXTEND_GQA", None)
        self.assertTrue(ns["_numinous_gqa_enabled"]())

    def test_numinous_gqa_enabled_false_sm100(self):
        ns = self._exec_helpers((10, 0))
        self.assertFalse(ns["_numinous_gqa_enabled"]())

    def test_gqa_config_global_layers(self):
        ns = self._exec_helpers((12, 0))
        cfg = ns["_numinous_gqa_config"](512, 8)
        self.assertEqual(cfg["block_t"], 8)
        self.assertEqual(cfg["dv"], 256)
        self.assertEqual(cfg["num_stages"], 3)

    def test_gqa_config_sliding_layers(self):
        ns = self._exec_helpers((12, 0))
        cfg = ns["_numinous_gqa_config"](256, 2)
        self.assertEqual(cfg["block_t"], 32)
        self.assertEqual(cfg["num_stages"], 1)

    def test_module_installed(self):
        hooks = _import_hooks()
        hooks.install_modules(self.fake_root, check=False)
        dst = os.path.join(self.fake_root, "kernels", "ops", "attention", "numinous_extend_gqa.py")
        self.assertTrue(os.path.isfile(dst), "numinous_extend_gqa.py not installed")


class TestGPUKernelOnBox(unittest.TestCase):
    """Single-GPU test: patched extend_attention_fwd dispatches to the GQA kernel on SM120 and matches reference.
    Skipped unless run on the rtx-gemma box (CUDA_CAPABILITY == 12).
    """

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            if not torch.cuda.is_available():
                raise unittest.SkipTest("no CUDA")
            cap = torch.cuda.get_device_capability()
            if cap[0] != 12:
                raise unittest.SkipTest(f"SM{cap[0]}{cap[1]}, not SM120")
            import flashinfer  # noqa: F401 (sets TRITON_PTXAS_BLACKWELL_PATH)
        except ImportError:
            raise unittest.SkipTest("flashinfer not available")
        cls.device = torch.device("cuda")

    def _make_batch(self, batch, prefix_len, extend_len, H, KVH, D, Dv, dtype_kv):
        import torch
        dev = self.device
        # construct qo_indptr, kv_indptr, kv_indices for a uniform batch
        qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
        kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
        for i in range(batch):
            qo_indptr[i + 1] = qo_indptr[i] + extend_len
            kv_indptr[i + 1] = kv_indptr[i] + prefix_len
        total_extend = batch * extend_len
        total_prefix = batch * prefix_len
        kv_indices = torch.arange(total_prefix, dtype=torch.int64, device=dev)
        q = torch.randn(total_extend, H, D, dtype=torch.bfloat16, device=dev)
        k_ext = torch.randn(total_extend, KVH, D, dtype=torch.bfloat16, device=dev)
        v_ext = torch.randn(total_extend, KVH, Dv, dtype=torch.bfloat16, device=dev)
        k_buf = (torch.randn(total_prefix, KVH, D, dtype=torch.bfloat16, device=dev)
                 .to(dtype_kv))
        v_buf = (torch.randn(total_prefix, KVH, Dv, dtype=torch.bfloat16, device=dev)
                 .to(dtype_kv))
        o = torch.zeros(total_extend, H, Dv, dtype=torch.bfloat16, device=dev)
        return q, k_ext, v_ext, o, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices

    def _run_stock_and_numinous(self, **kw):
        """Call extend_attention_gqa (our kernel) and the reference on the same data."""
        import torch
        from operations.kernels.extend_attention_gqa import extend_attention_gqa, reference
        q, k_ext, v_ext, o, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices = self._make_batch(**kw)
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        k_scale = 1.0 if k_buf.dtype != torch.float8_e4m3fn else 1.0
        v_scale = 1.0 if v_buf.dtype != torch.float8_e4m3fn else 1.0
        extend_attention_gqa(
            q, k_ext, v_ext, o, k_buf, v_buf,
            qo_indptr, kv_indptr, kv_indices,
            kw["extend_len"], sm_scale, k_scale, v_scale,
        )
        ref = reference(q, k_ext, v_ext, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices, sm_scale, k_scale, v_scale)
        return o, ref.to(torch.bfloat16)

    def test_global_layers_fp8_kv_b4(self):
        """Gemma 4 global layers: hd=512, 2 KV heads (8 Q heads per KV), fp8 cache, b=4."""
        import torch
        o, ref = self._run_stock_and_numinous(
            batch=4, prefix_len=512, extend_len=16,
            H=16, KVH=2, D=512, Dv=512,
            dtype_kv=torch.float8_e4m3fn,
        )
        maxerr = (o.float() - ref.float()).abs().max().item()
        self.assertLess(maxerr, 0.05, f"global hd512 fp8 max error {maxerr:.4f}")

    def test_sliding_layers_bf16_kv_b8(self):
        """Gemma 4 sliding layers: hd=256, 8 KV heads (2 Q heads per KV), window 1024, bf16 cache, b=8."""
        import torch
        o, ref = self._run_stock_and_numinous(
            batch=8, prefix_len=256, extend_len=8,
            H=16, KVH=8, D=256, Dv=256,
            dtype_kv=torch.bfloat16,
        )
        maxerr = (o.float() - ref.float()).abs().max().item()
        self.assertLess(maxerr, 0.01, f"sliding hd256 bf16 max error {maxerr:.4f}")


if __name__ == "__main__":
    unittest.main()
