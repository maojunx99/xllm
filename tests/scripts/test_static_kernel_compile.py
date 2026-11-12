# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for static compilation, cache validity and concurrent ranks."""

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("static_kernel_compile", _ROOT / "tools/static_kernel_compile.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_COMPILER = r'''#!/usr/bin/env python3
import json
import pathlib
import sys
args = sys.argv[1:]
root = pathlib.Path(__file__).parent
with (root / "invocations").open("a") as output:
    output.write("compile\n")
if (root / "fail").exists():
    sys.exit(3)
out = pathlib.Path(args[args.index("-o") + 1])
assert args[args.index("-f") + 1] == "true"
(out / "kernels.run").write_text("""#!/bin/bash
set -eu
for argument in "$@"; do
  case "$argument" in --install-path=*) destination="${argument#*=}" ;; esac
done
mkdir -p "$destination/static_kernel/ai_core/config/ascend910_test"
mkdir -p "$destination/static_kernel/ai_core/kernel"
echo '{"Add": [{"binPath": "kernel/Add.o"}]}' > "$destination/static_kernel/ai_core/config/ascend910_test/binary_info_config.json"
echo 'kernel' > "$destination/static_kernel/ai_core/kernel/Add.o"
""")
'''


class StaticKernelCompileTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self._root = Path(self._temporary.name)
        self._dump = self._root / "dump"
        self._inputs = self._dump / "123_opcompile"
        self._inputs.mkdir(parents=True)
        self._description = {"op_type": "Add", "shape": [1, 4096], "dtype": "float16"}
        (self._inputs / "Add.json").write_text(json.dumps(self._description), encoding="utf-8")
        self._compiler = self._root / "op_compiler"
        self._compiler.write_text(_COMPILER, encoding="utf-8")
        self._compiler.chmod(0o755)
        self._cache = self._root / "cache"
        self._environment = mock.patch.dict(
            os.environ,
            {
                "ASCEND_HOME_PATH": str(self._root),
                "ASCEND_OPP_PATH": str(self._root / "opp"),
                "ASCEND_CUSTOM_OPP_PATH": "",
            },
        )
        self._environment.start()
        self.addCleanup(self._environment.stop)

    def _compile(self, soc: str = "Ascend910_test", deterministic: bool = False) -> dict:
        return _MODULE.compile_static_kernels(self._dump, self._cache, soc, deterministic, compiler=self._compiler)

    def _compile_count(self) -> int:
        return len((self._root / "invocations").read_text().splitlines())

    def test_blacklist_excludes_exact_op_type_only(self) -> None:
        for op in ("RepeatInterleave", "RepeatInterleaveGrad"):
            (self._inputs / f"{op}.json").write_text(json.dumps({"op_type": op}))
        operators = _MODULE._operator_inputs([self._dump])
        self.assertEqual({json.loads(path.read_text())["op_type"] for path in operators.values()},
                         {"Add", "RepeatInterleaveGrad"})

    def test_blacklist_change_invalidates_compile_and_startup_cache(self) -> None:
        first = self._compile()
        before = _MODULE._startup_descriptor(self._startup_context(), "Ascend910_test", False,
                                             False, self._compiler)
        with mock.patch.object(_MODULE, "_blacklisted_ops", return_value=[]):
            second = self._compile()
            after = _MODULE._startup_descriptor(self._startup_context(), "Ascend910_test", False,
                                                False, self._compiler)
        self.assertNotEqual(first["cache_key"], second["cache_key"])
        self.assertNotEqual(before, after)
        self.assertFalse(second["cache_hit"])

    def test_duplicate_ranks_share_compilation_and_cache(self) -> None:
        other_rank = self._dump / "124_opcompile"
        other_rank.mkdir()
        (other_rank / "Add.json").write_text(json.dumps(self._description, indent=4), encoding="utf-8")
        first = self._compile()
        second = self._compile()
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["operator_count"], 1)
        self.assertEqual(first["install_path"], second["install_path"])
        self.assertEqual(self._compile_count(), 1)

    def test_shape_soc_determinism_and_toolchain_invalidate_cache(self) -> None:
        keys = [self._compile()["cache_key"]]
        keys.append(self._compile(soc="Ascend910_other")["cache_key"])
        keys.append(self._compile(deterministic=True)["cache_key"])
        (self._inputs / "Add.json").write_text(json.dumps({**self._description, "shape": [32, 4096]}))
        keys.append(self._compile()["cache_key"])
        self._compiler.write_text(_COMPILER + "\n# new compiler build\n", encoding="utf-8")
        keys.append(self._compile()["cache_key"])
        self.assertEqual(len(set(keys)), 5)
        self.assertEqual(self._compile_count(), 5)

    def test_corrupt_installation_rebuilds_without_changing_old_directory(self) -> None:
        first = self._compile()
        binary = Path(first["install_path"]) / "static_kernel/ai_core/kernel/Add.o"
        binary.write_text("truncated")
        second = self._compile()
        self.assertFalse(second["cache_hit"])
        self.assertNotEqual(first["install_path"], second["install_path"])
        self.assertEqual(binary.read_text(), "truncated")
        self.assertEqual(self._compile_count(), 2)

    def test_super_kernel_has_distinct_compilation_and_reusable_cache(self) -> None:
        with mock.patch.object(_MODULE, "_run", wraps=_MODULE._run) as run:
            plain = self._compile()
            enabled = _MODULE.compile_static_kernels(
                self._dump, self._cache, "Ascend910_test", compiler=self._compiler, super_kernel=True
            )
            cached = _MODULE.compile_static_kernels(
                self._dump, self._cache, "Ascend910_test", compiler=self._compiler, super_kernel=True
            )
        self.assertNotEqual(plain["cache_key"], enabled["cache_key"])
        self.assertEqual(enabled["cache_key"], cached["cache_key"])
        self.assertTrue(cached["cache_hit"])
        commands = [call.args[0] for call in run.call_args_list if call.args[0][0] == str(self._compiler)]
        self.assertEqual(len(commands), 2)
        self.assertNotIn("--enable_super_kernel", commands[0])
        self.assertIn("--enable_super_kernel", commands[1])
        self.assertEqual(self._compile_count(), 2)

    def test_failed_compile_is_never_published(self) -> None:
        (self._root / "fail").touch()
        with self.assertRaises(subprocess.CalledProcessError):
            self._compile()
        self.assertEqual(list(self._cache.glob("*.json")), [])
        self.assertTrue(list(self._cache.glob("build-*/compile.log")))
        (self._root / "fail").unlink()
        self.assertFalse(self._compile()["cache_hit"])
        self.assertEqual(self._compile_count(), 2)

    def test_concurrent_ranks_compile_once(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: self._compile(), range(4)))
        self.assertEqual(sum(not result["cache_hit"] for result in results), 1)
        self.assertEqual(len({result["install_path"] for result in results}), 1)
        self.assertEqual(self._compile_count(), 1)

    def test_node_merge_deduplicates_processes_and_keeps_distinct_attributes(self) -> None:
        other = self._root / "rank1" / "124_opcompile"
        other.mkdir(parents=True)
        (other / "Add.json").write_text(json.dumps(self._description, indent=4))
        (other / "OtherAdd.json").write_text(json.dumps({**self._description, "shape": [2, 4096]}))
        (self._inputs / "Address.json").write_text(json.dumps({**self._description, "attrs": {"address": 1}}))
        (other / "Address.json").write_text(json.dumps({**self._description, "attrs": {"address": 2}}))
        inputs = [self._dump, other.parent]
        result = _MODULE.compile_static_kernels(
            self._dump, self._cache, "Ascend910_test", compiler=self._compiler, dump_dirs=inputs
        )
        self.assertEqual(result["operator_count"], 4)
        descriptions = [json.loads(p.read_text()) for p in (Path(result["build_dir"]) / "inputs").glob("*.json")]
        self.assertEqual({x["attrs"]["address"] for x in descriptions if "attrs" in x}, {1, 2})
        cached = _MODULE.compile_static_kernels(
            other.parent, self._cache, "Ascend910_test", compiler=self._compiler, dump_dirs=list(reversed(inputs))
        )
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(result["cache_key"], cached["cache_key"])
        self.assertEqual(self._compile_count(), 1)

    def test_node_merge_accepts_an_empty_rank_but_rejects_missing_directory(self) -> None:
        empty = self._root / "empty-rank"
        empty.mkdir()
        result = _MODULE.compile_static_kernels(
            empty, self._cache, "Ascend910_test", compiler=self._compiler, dump_dirs=[empty, self._dump]
        )
        self.assertEqual(result["operator_count"], 1)
        with self.assertRaises(FileNotFoundError):
            _MODULE.compile_static_kernels(
                self._dump,
                self._cache,
                "Ascend910_test",
                compiler=self._compiler,
                dump_dirs=[self._dump, self._root / "missing"],
            )
        self.assertEqual(self._compile_count(), 1)

    def test_node_compilation_environment_ignores_rank_but_checks_runtime(self) -> None:
        first = self._probe({**self._startup_context(), "rank": 0})
        second = self._probe({**self._startup_context(), "rank": 1})
        self.assertNotEqual(first["startup_key"], second["startup_key"])
        self.assertEqual(first["compilation_environment"], second["compilation_environment"])
        queue = "0" if os.environ.get("TASK_QUEUE_ENABLE") == "2" else "2"
        with mock.patch.dict(os.environ, {"TASK_QUEUE_ENABLE": queue}):
            changed = self._probe()
        self.assertNotEqual(first["compilation_environment"], changed["compilation_environment"])

    def test_empty_collection_fails_before_compiling(self) -> None:
        (self._inputs / "Add.json").unlink()
        with self.assertRaisesRegex(ValueError, "No CANN operator arguments"):
            self._compile()
        self.assertFalse((self._root / "invocations").exists())

    def test_registered_vendor_change_invalidates_without_custom_path_environment(self) -> None:
        library = self._root / "opp/vendors/test_vendor/op_api/lib/custom.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"first vendor build")
        first = self._compile()
        library.write_bytes(b"second vendor build")
        second = self._compile()
        self.assertNotEqual(first["cache_key"], second["cache_key"])
        self.assertEqual(self._compile_count(), 2)

    def _startup_context(self) -> dict:
        image = self._root / "xllm"
        if not image.exists():
            image.write_bytes(b"test executable")
        return {
            "runtime_image": str(image),
            "model": {"dtype": "bf16", "tp": 2, "ep": 2},
            "cache_shape": [140, 16, 128, 128],
        }

    def _probe(self, context: dict | None = None, super_kernel: bool = False, auto_op_parallel: bool = False) -> dict:
        return _MODULE.probe_startup_cache(
            self._startup_context() if context is None else context,
            self._cache,
            "Ascend910_test",
            super_kernel=super_kernel,
            compiler=self._compiler,
            super_kernel_auto_op_parallel=auto_op_parallel,
        )

    def test_super_kernel_scheduling_options_separate_compile_and_startup_caches(self) -> None:
        results = []
        probes = []
        with mock.patch.object(_MODULE, "_run", wraps=_MODULE._run) as run:
            for enabled in (False, True):
                result = _MODULE.compile_static_kernels(
                    self._dump,
                    self._cache,
                    "Ascend910_test",
                    compiler=self._compiler,
                    super_kernel=True,
                    super_kernel_auto_op_parallel=enabled,
                )
                results.append(result)
                probe = self._probe(super_kernel=True, auto_op_parallel=enabled)
                self.assertFalse(probe["startup_cache_hit"])
                _MODULE.publish_startup_cache(self._cache, [probe["startup_key"]], result["cache_key"])
                probes.append(self._probe(super_kernel=True, auto_op_parallel=enabled))
            self.assertTrue(all(p["startup_cache_hit"] for p in probes))
        self.assertNotEqual(results[0]["cache_key"], results[1]["cache_key"])
        self.assertNotEqual(probes[0]["startup_key"], probes[1]["startup_key"])
        self.assertNotEqual(probes[0]["compilation_environment"], probes[1]["compilation_environment"])
        commands = [call.args[0] for call in run.call_args_list if call.args[0][0] == str(self._compiler)]
        self.assertEqual(len(commands), 2)
        for enabled, command in enumerate(commands):
            self.assertEqual(
                command[command.index("--super_kernel_options") + 1],
                f"dcci_after_kernel_end=.*:auto_op_parallel={enabled}",
            )

    def test_plain_static_compilation_ignores_super_kernel_scheduling(self) -> None:
        plain = self._compile()
        cached = _MODULE.compile_static_kernels(
            self._dump,
            self._cache,
            "Ascend910_test",
            compiler=self._compiler,
            super_kernel_auto_op_parallel=True,
        )
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(plain["cache_key"], cached["cache_key"])
        self.assertEqual(self._compile_count(), 1)

    def _publish(self) -> tuple[dict, dict]:
        compiled = self._compile()
        probe = self._probe()
        self.assertFalse(probe["startup_cache_hit"])
        _MODULE.publish_startup_cache(self._cache, [probe["startup_key"]], compiled["cache_key"])
        return compiled, probe

    def test_startup_hit_does_not_collect_or_compile(self) -> None:
        compiled, _ = self._publish()
        (self._inputs / "Add.json").unlink()
        with mock.patch.object(_MODULE, "_operator_inputs", side_effect=AssertionError("collection must not run")):
            hit = self._probe()
        self.assertTrue(hit["startup_cache_hit"])
        self.assertEqual(hit["result"]["install_path"], compiled["install_path"])
        self.assertEqual(self._compile_count(), 1)

    def test_startup_invalidates_model_parallelism_and_state_layout(self) -> None:
        self._publish()
        original = self._startup_context()
        variants = [
            {**original, "model": {**original["model"], "tp": 4}},
            {**original, "model": {**original["model"], "ep": 1}},
            {**original, "model": {**original["model"], "dtype": "fp16"}},
            {**original, "cache_shape": [141, 16, 128, 128]},
            {**original, "decode_batches": [1, 2, 4, 8]},
        ]
        for context in variants:
            with self.subTest(context=context):
                self.assertFalse(self._probe(context)["startup_cache_hit"])
        self.assertTrue(self._probe(original)["startup_cache_hit"])
        self.assertEqual(self._compile_count(), 1)

    def test_startup_invalidates_runtime_image_library_and_toolchain(self) -> None:
        self._publish()
        image = self._root / "xllm"
        image.write_bytes(b"new executable")
        self.assertFalse(self._probe()["startup_cache_hit"])
        image.write_bytes(b"test executable")
        self.assertTrue(self._probe()["startup_cache_hit"])
        library = self._root / "libtorch_npu.so"
        library.write_bytes(b"runtime version one")
        context = {**self._startup_context(), "runtime_libraries": [str(library)]}
        compiled = self._compile()
        probe = self._probe(context)
        _MODULE.publish_startup_cache(self._cache, [probe["startup_key"]], compiled["cache_key"])
        library.write_bytes(b"runtime version two")
        self.assertFalse(self._probe(context)["startup_cache_hit"])
        self._compiler.write_text(_COMPILER + "\n# upgraded compiler\n")
        self.assertFalse(self._probe()["startup_cache_hit"])

    def test_startup_invalidates_checkpoint_metadata_without_reading_weights(self) -> None:
        model = self._root / "model"
        model.mkdir()
        metadata = model / "config.json"
        metadata.write_text('{"hidden_size":4096}')
        weights = model / "model.safetensors"
        weights.write_bytes(b"weights are not compiler constants")
        context = {**self._startup_context(), "model_path": str(model)}
        compiled = self._compile()
        probe = self._probe(context)
        _MODULE.publish_startup_cache(self._cache, [probe["startup_key"]], compiled["cache_key"])
        weights.write_bytes(b"different weights, identical model structure")
        self.assertTrue(self._probe(context)["startup_cache_hit"])
        metadata.write_text('{"hidden_size":8192}')
        self.assertFalse(self._probe(context)["startup_cache_hit"])

    def test_startup_invalidates_math_environment_but_ignores_ports(self) -> None:
        with mock.patch.dict(os.environ, {"HCCL_DETERMINISTIC": "false", "HCCL_IF_BASE_PORT": "55000"}):
            self._publish()
            with mock.patch.dict(os.environ, {"HCCL_IF_BASE_PORT": "55016"}):
                self.assertTrue(self._probe()["startup_cache_hit"])
            with mock.patch.dict(os.environ, {"HCCL_DETERMINISTIC": "true"}):
                self.assertFalse(self._probe()["startup_cache_hit"])

    def test_startup_corruption_is_a_miss_before_loading(self) -> None:
        compiled, probe = self._publish()
        alias = self._cache / "startup" / (probe["startup_key"] + ".json")
        alias.write_text("truncated json")
        self.assertFalse(self._probe()["startup_cache_hit"])
        _MODULE.publish_startup_cache(self._cache, [probe["startup_key"]], compiled["cache_key"])
        binary = Path(compiled["install_path"]) / "static_kernel/ai_core/kernel/Add.o"
        binary.write_bytes(b"corrupt kernel")
        self.assertFalse(self._probe()["startup_cache_hit"])
        with self.assertRaisesRegex(ValueError, "invalid static installation"):
            _MODULE.publish_startup_cache(self._cache, [probe["startup_key"]], compiled["cache_key"])

    def test_startup_publishes_all_local_worker_aliases_atomically(self) -> None:
        compiled, first = self._publish()
        other = self._probe({**self._startup_context(), "rank": 1})
        keys = [first["startup_key"], other["startup_key"]]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(
                executor.map(
                    lambda _: _MODULE.publish_startup_cache(self._cache, keys, compiled["cache_key"]), range(4)
                )
            )
        self.assertTrue(self._probe()["startup_cache_hit"])
        self.assertTrue(self._probe({**self._startup_context(), "rank": 1})["startup_cache_hit"])
        self.assertEqual(self._compile_count(), 1)

    def test_startup_keeps_plain_and_super_kernel_separate(self) -> None:
        self._publish()
        self.assertFalse(self._probe(super_kernel=True)["startup_cache_hit"])
        self.assertTrue(self._probe()["startup_cache_hit"])


if __name__ == "__main__":
    unittest.main()
