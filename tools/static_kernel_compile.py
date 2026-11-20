#!/usr/bin/env python3
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

"""Compile collected CANN operator arguments into a private static kernel cache.

This helper needs neither torch nor TorchAir. It runs only during startup.
Completed installations are immutable: a lock and an atomic manifest publish
them to other ranks, while failed attempts retain logs without becoming hits.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# The CMake-installed executable has scripts/logger.py next to this helper;
# the source-tree helper lives in tools/.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR if (_SCRIPT_DIR / "scripts/logger.py").is_file() else _SCRIPT_DIR.parent))
from scripts.logger import logger  # noqa: E402

_CACHE_SCHEMA = 1
_STARTUP_SCHEMA = 1


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _super_kernel_options(auto_op_parallel: bool) -> dict[str, Any]:
    return {"dcci_after_kernel_end": [".*"], "auto_op_parallel": int(auto_op_parallel)}


def _atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as output:
        temporary = Path(output.name)
        json.dump(value, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _blacklisted_ops() -> list[str]:
    path = _SCRIPT_DIR / "static_kernel_blacklist.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(op, str) or not op for op in value):
        raise ValueError(f"Static kernel blacklist must contain nonempty OpType strings: {path}")
    return sorted(set(value))


def _operator_inputs(dump_dirs: list[Path]) -> dict[str, Path]:
    # CANN emits compiler inputs separately from debug/PID metadata.
    directories: list[Path] = []
    for dump_dir in sorted(set(path.resolve(strict=True) for path in dump_dirs)):
        selected = sorted(path for path in dump_dir.rglob("*_opcompile") if path.is_dir())
        if not selected:
            selected = sorted(path for path in dump_dir.rglob("*_debug") if path.is_dir())
        directories.extend(selected)
    blacklist = set(_blacklisted_ops())
    excluded: set[str] = set()
    operators: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            description = json.loads(path.read_text(encoding="utf-8"))
            if description.get("op_type") in blacklist:
                excluded.add(description["op_type"])
                continue
            signature = hashlib.sha256(_canonical_json(description)).hexdigest()
            operators.setdefault(signature, path)
    if excluded:
        logger.info("Excluded static kernel OpTypes: %s", sorted(excluded))
    if not operators:
        raise ValueError(f"No CANN operator arguments were collected under {dump_dirs}")
    return operators


def _toolchain_fingerprint(compiler: Path) -> dict[str, Any]:
    cann_root = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not cann_root:
        raise ValueError("Source CANN set_env.sh before enabling static kernels")
    root = Path(cann_root).resolve()
    fingerprints: dict[str, str] = {str(compiler): _digest(compiler)}
    version_files = set(root.glob("*.info")) | set(root.glob("*/version.info"))
    version_files |= set(root.glob("*/ascend_*_install.info"))
    for path in sorted(version_files):
        if path.is_file():
            fingerprints[str(path)] = _digest(path)
    for name in ("libopapi.so", "libnnopbase.so", "libascend_dump.so"):
        path = root / "lib64" / name
        if path.is_file():
            fingerprints[str(path.resolve())] = _digest(path)
    opp_root = Path(os.environ.get("ASCEND_OPP_PATH", str(root / "opp"))).resolve()
    vendor_roots = {Path(path).resolve() for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if path}
    # CANN also loads vendors registered in the OPP installation when the
    # custom-path environment variable is empty.
    installed_vendors = opp_root / "vendors"
    if installed_vendors.is_dir():
        vendor_roots.update(path.resolve() for path in installed_vendors.iterdir() if path.is_dir())
    vendor_payloads: dict[str, tuple[int, int]] = {}
    for vendor in vendor_roots:
        files = set(vendor.rglob("*.info")) | set(vendor.rglob("*.so"))
        for path in sorted(files):
            if path.is_file():
                fingerprints[str(path)] = _digest(path)
        for path in vendor.rglob("*"):
            if path.is_file() and path.suffix in (".o", ".bin", ".json", ".py", ".cpp", ".h"):
                metadata = path.stat()
                vendor_payloads[str(path)] = (metadata.st_size, metadata.st_mtime_ns)
    return {
        "cann_root": str(root),
        "opp_root": str(opp_root),
        "vendor_roots": sorted(str(path) for path in vendor_roots),
        "vendor_payloads": vendor_payloads,
        "files": fingerprints,
    }


def _installed_files(install_path: Path) -> dict[str, str]:
    root = install_path / "static_kernel/ai_core"
    configurations = sorted(root.glob("config/*/binary_info_config.json"))
    if not configurations:
        raise ValueError(f"Static installation has no SoC kernel index: {install_path}")
    for configuration in configurations:
        if not json.loads(configuration.read_text(encoding="utf-8")):
            raise ValueError(f"Empty static kernel index: {configuration}")
    binaries = [path for path in root.rglob("*") if path.suffix in (".o", ".bin")]
    if not binaries:
        raise ValueError(f"Static installation has no kernel binaries: {install_path}")
    return {
        str(path.relative_to(install_path)): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix != ".lock"
    }


def _valid_cached_result(manifest: Path, cache_key: str) -> dict[str, Any] | None:
    if not manifest.is_file():
        return None
    # Cache corruption is an expected disk/cache condition, checked before any
    # model runtime state changes. Rebuild into a fresh immutable directory.
    try:
        cached = json.loads(manifest.read_text(encoding="utf-8"))
        if cached["cache_key"] != cache_key or cached["schema"] != _CACHE_SCHEMA:
            return None
        files = _installed_files(Path(cached["install_path"]))
        if files != cached["files"]:
            return None
        return {**cached, "cache_hit": True}
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Ignoring incomplete or corrupt static kernel cache: %s", manifest)
        return None


def _run(arguments: list[str], log: Path) -> None:
    logger.info("Running static kernel tool; log: %s", log)
    with log.open("wb") as output:
        subprocess.run(arguments, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=3600)


def _startup_descriptor(
    context: dict[str, Any],
    soc: str,
    deterministic: bool,
    super_kernel: bool,
    compiler: Path | None = None,
    super_kernel_auto_op_parallel: bool = False,
) -> dict[str, Any]:
    """Identify a launch without executing model operators or reading weights."""
    if compiler is None:
        executable = shutil.which("op_compiler")
        if executable is None:
            raise ValueError("op_compiler is not on PATH; source CANN set_env.sh")
        compiler = Path(executable).resolve()
    image = Path(context["runtime_image"])
    model_files: dict[str, str] = {}
    model_path = context.get("model_path")
    if model_path:
        model_root = Path(model_path)
        for path in sorted(model_root.rglob("*.json")):
            if path.is_file():
                model_files[str(path.relative_to(model_root))] = _digest(path)
    # These options affect implementation selection. Network ports, log paths,
    # physical device numbering and process ids do not belong in a launch key.
    environment = {
        name: os.environ.get(name)
        for name in (
            "TASK_QUEUE_ENABLE",
            "HCCL_DETERMINISTIC",
            "ASCEND_COMPUTE_UNIT",
            "ACL_OP_SELECT_IMPL_MODE",
            "ACL_OPTYPELIST_FOR_IMPLMODE",
            "ALLOW_MATMUL_HF32",
            "ALLOW_CONV_HF32",
            "NPU_FUZZY_COMPILE_BLACKLIST",
        )
    }
    descriptor = {
        "schema": _STARTUP_SCHEMA,
        "blacklisted_ops": _blacklisted_ops(),
        "soc": soc,
        "deterministic": deterministic,
        "super_kernel": super_kernel,
        "context": context,
        "runtime_image": _digest(image),
        "runtime_libraries": {
            str(Path(path).resolve()): _digest(Path(path)) for path in context.get("runtime_libraries", [])
        },
        "helper": _digest(Path(__file__)),
        "model_metadata": model_files,
        "environment": environment,
        "toolchain": _toolchain_fingerprint(compiler),
    }
    if super_kernel:
        descriptor["super_kernel_options"] = _super_kernel_options(super_kernel_auto_op_parallel)
    return descriptor


def probe_startup_cache(
    context: dict[str, Any],
    cache_dir: Path,
    soc: str,
    deterministic: bool = False,
    super_kernel: bool = False,
    compiler: Path | None = None,
    super_kernel_auto_op_parallel: bool = False,
) -> dict[str, Any]:
    """Validate an existing launch alias before changing CANN selection state."""
    descriptor = _startup_descriptor(context, soc, deterministic, super_kernel, compiler, super_kernel_auto_op_parallel)
    key = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    # Different model stages may share an installation, but compiler/runtime
    # selection settings must agree. Keep process addresses in operator inputs.
    compilation_environment = {
        name: value for name, value in descriptor.items() if name not in ("context", "model_metadata")
    }
    environment_key = hashlib.sha256(_canonical_json(compilation_environment)).hexdigest()
    miss = {"startup_cache_hit": False, "startup_key": key, "compilation_environment": environment_key}
    alias = cache_dir / "startup" / f"{key}.json"
    if not alias.is_file():
        return miss
    try:
        index = json.loads(alias.read_text(encoding="utf-8"))
        if index["schema"] != _STARTUP_SCHEMA or index["startup_key"] != key:
            return miss
        compiled_key = index["cache_key"]
        if (
            not isinstance(compiled_key, str)
            or len(compiled_key) != 64
            or any(c not in "0123456789abcdef" for c in compiled_key)
        ):
            return miss
        cached = _valid_cached_result(cache_dir / f"{compiled_key}.json", compiled_key)
        if cached is None:
            return miss
        if bool(cached.get("super_kernel", False)) != super_kernel:
            return miss
        if super_kernel and bool(cached.get("super_kernel_auto_op_parallel", False)) != super_kernel_auto_op_parallel:
            return miss
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Ignoring incomplete or corrupt static startup index: %s", alias)
        return miss
    logger.info("Static kernel startup cache hit: %s", key)
    return {**miss, "startup_cache_hit": True, "result": cached}


def publish_startup_cache(cache_dir: Path, startup_keys: list[str], cache_key: str) -> dict[str, Any]:
    """Publish launch aliases only after every worker successfully loaded kernels."""
    if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
        raise ValueError("Invalid static kernel cache key")
    if _valid_cached_result(cache_dir / f"{cache_key}.json", cache_key) is None:
        raise ValueError("Cannot publish a launch index for an invalid static installation")
    directory = cache_dir / "startup"
    directory.mkdir(parents=True, exist_ok=True)
    for key in sorted(set(startup_keys)):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid static kernel startup key")
        with (directory / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _atomic_json(
                directory / f"{key}.json", {"schema": _STARTUP_SCHEMA, "startup_key": key, "cache_key": cache_key}
            )
    return {"published_startup_keys": sorted(set(startup_keys))}


def compile_static_kernels(
    dump_dir: Path,
    cache_dir: Path,
    soc: str,
    deterministic: bool = False,
    jobs: int = 1,
    compiler: Path | None = None,
    super_kernel: bool = False,
    dump_dirs: list[Path] | None = None,
    super_kernel_auto_op_parallel: bool = False,
) -> dict[str, Any]:
    """Merge participating processes' inputs, deduplicate, compile and install."""
    if compiler is None:
        executable = shutil.which("op_compiler")
        if executable is None:
            raise ValueError("op_compiler is not on PATH; source CANN set_env.sh")
        compiler = Path(executable).resolve()
    operators = _operator_inputs([dump_dir] if dump_dirs is None else dump_dirs)
    context = {
        "schema": _CACHE_SCHEMA,
        "blacklisted_ops": _blacklisted_ops(),
        "soc": soc,
        "deterministic": deterministic,
        "toolchain": _toolchain_fingerprint(compiler),
        "operators": sorted(operators),
    }
    # Preserve existing plain-static cache keys while keeping SK-enabled
    # binaries in a distinct immutable installation.
    if super_kernel:
        context["super_kernel"] = True
        context["super_kernel_options"] = _super_kernel_options(super_kernel_auto_op_parallel)
    cache_key = hashlib.sha256(_canonical_json(context)).hexdigest()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / f"{cache_key}.json"
    with (cache_dir / f"{cache_key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _valid_cached_result(manifest, cache_key)
        if cached is not None:
            logger.info("Static kernel cache hit: %s", cache_key)
            return cached
        attempt = Path(tempfile.mkdtemp(prefix=f"build-{cache_key[:12]}-", dir=cache_dir))
        input_dir = attempt / "inputs"
        input_dir.mkdir()
        # Keep CANN filenames where possible; colliding rank filenames get a
        # signature suffix. Equal descriptions compile just once.
        for signature, source in operators.items():
            destination = input_dir / source.name
            if destination.exists():
                destination = input_dir / f"{source.stem}_{signature}{source.suffix}"
            shutil.copyfile(source, destination)
        output_dir = attempt / "packages"
        output_dir.mkdir()
        _atomic_json(attempt / "context.json", context)
        command = [str(compiler), "-p", str(input_dir), "-v", soc, "-j", str(jobs), "-f", "true", "-o", str(output_dir)]
        if super_kernel:
            command.append("--enable_super_kernel")
            command.extend(
                [
                    "--super_kernel_options",
                    f"dcci_after_kernel_end=.*:auto_op_parallel={int(super_kernel_auto_op_parallel)}",
                ]
            )
        _run(command, attempt / "compile.log")
        packages = sorted(output_dir.rglob("*.run"))
        if not packages:
            raise ValueError(f"op_compiler produced no installation packages; inspect {attempt / 'compile.log'}")
        install_path = attempt / "install"
        install_path.mkdir()
        for index, package in enumerate(packages):
            _run(["bash", str(package), f"--install-path={install_path}", "--quiet"], attempt / f"install-{index}.log")
        result = {
            "schema": _CACHE_SCHEMA,
            "cache_key": cache_key,
            "cache_hit": False,
            "operator_count": len(operators),
            "super_kernel": super_kernel,
            "super_kernel_auto_op_parallel": super_kernel and super_kernel_auto_op_parallel,
            "install_path": str(install_path),
            "build_dir": str(attempt),
            "files": _installed_files(install_path),
        }
        _atomic_json(manifest, result)
        logger.info("Static kernel installation ready: %s (%d operator descriptions)", install_path, len(operators))
        return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--deterministic", type=int, choices=(0, 1), default=0)
    parser.add_argument("--super-kernel", action="store_true")
    parser.add_argument("--super-kernel-auto-op-parallel", type=int, choices=(0, 1), default=0)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 4)))
    parser.add_argument("--action", choices=("compile", "probe", "publish"), default="compile")
    parser.add_argument("--startup-context", type=Path)
    parser.add_argument("--startup-keys", type=Path)
    parser.add_argument("--compiled-key")
    parser.add_argument("--dump-dirs", type=Path, help="JSON list of node-local collection directories")
    arguments = parser.parse_args()
    if arguments.jobs < 1:
        parser.error("--jobs must be positive")
    if arguments.action == "probe":
        if arguments.startup_context is None:
            parser.error("--startup-context is required for probe")
        context = json.loads(arguments.startup_context.read_text(encoding="utf-8"))
        result = probe_startup_cache(
            context,
            arguments.cache_dir,
            arguments.soc,
            bool(arguments.deterministic),
            arguments.super_kernel,
            super_kernel_auto_op_parallel=bool(arguments.super_kernel_auto_op_parallel),
        )
        if "result" in result:
            result["result"].pop("files", None)
    elif arguments.action == "publish":
        if arguments.startup_keys is None or arguments.compiled_key is None:
            parser.error("--startup-keys and --compiled-key are required for publish")
        result = publish_startup_cache(
            arguments.cache_dir, json.loads(arguments.startup_keys.read_text(encoding="utf-8")), arguments.compiled_key
        )
    else:
        dump_dirs = None
        if arguments.dump_dirs is not None:
            collected = json.loads(arguments.dump_dirs.read_text(encoding="utf-8"))
            if not isinstance(collected, list) or not collected or not all(isinstance(path, str) for path in collected):
                parser.error("--dump-dirs must contain a non-empty JSON list of paths")
            dump_dirs = [Path(path) for path in collected]
        result = compile_static_kernels(
            arguments.dump_dir,
            arguments.cache_dir,
            arguments.soc,
            bool(arguments.deterministic),
            arguments.jobs,
            super_kernel=arguments.super_kernel,
            dump_dirs=dump_dirs,
            super_kernel_auto_op_parallel=bool(arguments.super_kernel_auto_op_parallel),
        )
    _atomic_json(arguments.result, {key: value for key, value in result.items() if key != "files"})


if __name__ == "__main__":
    _main()
