#!/usr/bin/env python3
"""Calibrate and bake only the dense DSV4 linears into EXL3.

The bake deliberately keeps the routed-expert tensors in the existing EXL3
pack.  It uses the read-only exllamav3 checkout as a runtime dependency and
does not copy or modify that checkout.  ``--dry-run`` only inspects config,
model layout, and (when ``--merge`` is supplied) local safetensors headers.
It never imports CUDA conversion kernels or writes a model pack.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

try:
    from . import layer_overlay as overlay
except ImportError:  # Running as ``python tools/dense_bake.py``.
    import layer_overlay as overlay


CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
DENSE_SHARD = "model-dense-exl3.safetensors"
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg")
EXL3_DTYPES = {"trellis": "I16", "suh": "F16", "svh": "F16", "mcg": "I32"}
MCG_MARKER = -877912083
DEFAULT_EXL3_SOURCE = "/run/media/tonoken3/DATA1/.tmp/exl3src"

_LAYER_FILE_RE = re.compile(r"^layers\.(?P<layer>[0-9]+)\.safetensors$")
_WO_A_RE = re.compile(r"^attn\.wo_a\.slice\.(?P<slice>[0-7])$")
_DENSE_TAILS = {
    "attn.wq_a",
    "attn.wq_b",
    "attn.wkv",
    "attn.wo_b",
    "attn.compressor.wkv",
    "attn.compressor.wgate",
    "attn.indexer.compressor.wkv",
    "attn.indexer.compressor.wgate",
    "attn.indexer.wq_b",
    "ffn.shared_experts.w1",
    "ffn.shared_experts.w2",
    "ffn.shared_experts.w3",
}


def log(message: str) -> None:
    print(message, flush=True)


def dense_kind(key: str) -> str | None:
    """Return ``attn`` or ``shared`` only for the ordered T4 target list."""
    match = re.match(r"^layers\.[0-9]+\.(?P<tail>.+)$", key)
    if match is None:
        return None
    tail = match.group("tail")
    if tail in _DENSE_TAILS or _WO_A_RE.fullmatch(tail):
        return "shared" if tail.startswith("ffn.shared_experts.") else "attn"
    return None


def is_dense_key(key: str) -> bool:
    return dense_kind(key) is not None


def dense_keys_for_layer(layer: int, keys: Iterable[str]) -> list[str]:
    """Filter a module's recursive Linear keys, excluding experts, MTP, and non-linears."""
    result = [key for key in keys if dense_kind(key) is not None and key.startswith(f"layers.{layer}.")]
    result.sort()
    wo_a = [key for key in result if ".attn.wo_a.slice." in key]
    if wo_a and {int(key.rsplit(".", 1)[1]) for key in wo_a} != set(range(8)):
        raise RuntimeError(f"layers.{layer}: wo_a must contain exactly slices 0..7, got {wo_a}")
    return result


def bits_for_key(key: str, bits: float, attn_bits: float | None, shared_bits: float | None) -> float:
    kind = dense_kind(key)
    if kind is None:
        raise ValueError(f"not a T4 dense key: {key}")
    return float(attn_bits if kind == "attn" and attn_bits is not None else
                 shared_bits if kind == "shared" and shared_bits is not None else bits)


def _schema_bits(value: float) -> int:
    if not float(value).is_integer() or int(value) not in (2, 3, 4, 5, 6):
        raise ValueError(f"vLLM non_routed_exl3 requires integer K in 2..6, got {value}")
    return int(value)


def _layer_number(key: str) -> int:
    match = re.match(r"^layers\.([0-9]+)(?:\.|$)", key)  # LNA-LAB: module keys are bare "layers.N"
    if match is None:
        raise ValueError(f"not a language-layer key: {key}")
    return int(match.group(1))


def _parse_devices(value: str) -> list[int]:
    try:
        devices = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid --devices value: {value!r}") from exc
    if not devices or any(device < 0 for device in devices) or len(set(devices)) != len(devices):
        raise ValueError(f"--devices must be unique non-negative integers: {value!r}")
    return devices


def _runtime_import(*, dry_run: bool = False):
    """Import the exact exllamav3 classes/functions used by the bake.

    A dry-run can construct the Python model graph without the native extension.
    The extension stub is never used by a real bake; it exists solely to make
    the requested offline layout check possible on a CPU-only host.
    """
    source_root = os.environ.get("EXL3_SOURCE", DEFAULT_EXL3_SOURCE)
    if not Path(source_root).is_dir():
        raise RuntimeError(f"exllamav3 source checkout is missing: {source_root}")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    if dry_run and "exllamav3.ext" not in sys.modules:
        class _ExtensionStub:
            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        import types
        ext_module = types.ModuleType("exllamav3.ext")
        ext_module.exllamav3_ext = _ExtensionStub()
        sys.modules["exllamav3.ext"] = ext_module

    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.modules import Linear
    from exllamav3.loader.safetensors import SafetensorsCollection, VariantSafetensorsCollection
    from exllamav3.conversion.calibration_data import get_default_calibration
    from exllamav3.conversion.convert_model import (
        advance_state_parallel,
        capture_module_parallel,
        load_parallel_calib_modules,
        quantize_linears_parallel,
        quantize_linears_single,
        load_tensor,
        save_tensor,
    )
    return SimpleNamespace(
        Config=Config,
        Model=Model,
        Tokenizer=Tokenizer,
        Linear=Linear,
        SafetensorsCollection=SafetensorsCollection,
        VariantSafetensorsCollection=VariantSafetensorsCollection,
        get_default_calibration=get_default_calibration,
        advance_state_parallel=advance_state_parallel,
        capture_module_parallel=capture_module_parallel,
        load_parallel_calib_modules=load_parallel_calib_modules,
        quantize_linears_parallel=quantize_linears_parallel,
        quantize_linears_single=quantize_linears_single,
        load_tensor=load_tensor,
        save_tensor=save_tensor,
    )


def _load_variant_model(src: Path, experts: Path, runtime):
    config = runtime.Config.from_directory(str(src))
    expert_stc = runtime.SafetensorsCollection(str(experts))
    variant = runtime.VariantSafetensorsCollection(config.stc)
    # This is the T4 contract: only routed expert keys are overridden.
    variant.add_stc(["layers.*.ffn.experts.*"], expert_stc)
    config.stc = variant
    model = runtime.Model.from_config(config)
    # DSV4's architecture exposes this flag for calibration; make the T4
    # invariant explicit in case a future architecture default changes.
    model.calibration_all_experts = True
    return config, model


def _module_plan(model, linear_type) -> list[tuple[int, Any, list[Any]]]:
    plan = []
    for idx, module in enumerate(model.modules):
        layer_match = re.fullmatch(r"layers\.([0-9]+)", getattr(module, "key", ""))
        if layer_match is None:
            continue
        layer = int(layer_match.group(1))
        linears = [m for m in module if isinstance(m, linear_type)]
        keys = dense_keys_for_layer(layer, [m.key for m in linears])
        by_key = {m.key: m for m in linears}
        targets = [by_key[key] for key in keys]
        if len(targets) == 0:
            raise RuntimeError(f"{module.key}: no dense linears found")
        plan.append((idx, module, targets))
    if not plan:
        raise RuntimeError("model has no layers.N modules")
    return plan


def dry_run_layout(src: Path, experts: Path) -> dict[str, Any]:
    if not src.is_dir() or not experts.is_dir():
        raise RuntimeError(f"dry-run inputs must be directories: {src}, {experts}")
    runtime = _runtime_import(dry_run=True)
    # Construct the real source layout with the D2 expert override, and also
    # construct Config.from_directory(D2) explicitly: the latter is the
    # complete experts-only config used for the offline pack sanity check.
    config, model = _load_variant_model(src, experts, runtime)
    d2_config = runtime.Config.from_directory(str(experts))
    plan = _module_plan(model, runtime.Linear)
    counts = {"attn": 0, "shared": 0}
    for _, _, targets in plan:
        for linear in targets:
            counts[dense_kind(linear.key)] += 1
    log(
        f"DRY_RUN_LAYOUT architecture={getattr(config, 'architecture', 'unknown')} "
        f"modules={len(model.modules)} layers={len(plan)} dense_linears={sum(counts.values())} "
        f"attn={counts['attn']} shared={counts['shared']} routed_override=layers.*.ffn.experts.* "
        f"d2_config={getattr(d2_config, 'architecture', 'unknown')}"
    )
    for _, module, targets in plan[:2]:
        log(f"DRY_RUN_LAYER key={module.key} dense_linears={len(targets)}")
    log(f"DRY_RUN_LAYER key={plan[-1][1].key} dense_linears={len(plan[-1][2])}")
    return {"config": config, "model": model, "runtime": runtime, "plan": plan}


def _load_module(module, device, *, load_slice=None, source=None, close=True):
    defer = source is None and module.can_defer_load()
    if defer:
        module.config.stc.begin_deferred_load()
    try:
        kwargs = {}
        if load_slice is not None:
            kwargs["load_slice"] = load_slice
        if source is not None:
            kwargs["source"] = source
        module.load(device, **kwargs)
    finally:
        if defer:
            module.config.stc.end_deferred_load()
    if close:
        module.config.stc.close()


def _capture_serial(model, module, state, original_ids, quant_preserves, bad_rows, num_ref_states=5):
    capture_h = {}
    ref_states = {}

    def get_preserve(i, params):
        params.update(quant_preserves[i])
        params["quant_preserve"] = quant_preserves[i]

    def put_preserve(i, params):
        quant_preserves[i] = params["quant_preserve"]

    for i in range(len(state)):
        if i in bad_rows:
            continue
        params = {
            "attn_mode": "flash_attn_nc",
            "capture": capture_h,
            "activate_all_experts": model.calibration_all_experts,
            "input_ids": original_ids[i],
        }
        get_preserve(i, params)
        result = module.forward(module.prepare_for_device(state[i], params), params)
        put_preserve(i, params)
        if i < num_ref_states:
            if model.calibration_all_experts:
                params = {"attn_mode": "flash_attn_nc", "input_ids": original_ids[i]}
                get_preserve(i, params)
                result = module.forward(module.prepare_for_device(state[i], params), params)
                put_preserve(i, params)
            if result.isfinite().all().item():
                ref_states[i] = result.cpu()
            else:
                bad_rows.add(i)
                log(f"WARNING non-finite reference state row={i}; excluded")
    for item in capture_h.values():
        item["H_swap_device"] = item["H"].device
        item["H"] = item["H"].cpu()
    return capture_h, ref_states, get_preserve, put_preserve


def _advance_serial(model, module, state, original_ids, quant_preserves, ref_states, bad_rows, have_linears):
    error = 0.0
    measured = 0

    def get_preserve(i, params):
        params.update(quant_preserves[i])
        params["quant_preserve"] = quant_preserves[i]

    def put_preserve(i, params):
        quant_preserves[i] = params["quant_preserve"]

    for i in range(len(state)):
        if i in bad_rows:
            continue
        params = {"attn_mode": "flash_attn_nc", "input_ids": original_ids[i]}
        state[i] = module.prepare_for_device(state[i], params)
        if i < 5:
            get_preserve(i, params)
            result = module.forward(state[i], params)
            put_preserve(i, params)
            if not result.isfinite().all().item():
                bad_rows.add(i)
                continue
            state[i] = result.cpu()
        else:
            # We stop after the final transformer block, so every row must be
            # advanced; the reference/error pass remains limited to five rows.
            get_preserve(i, params)
            state[i] = module.forward(state[i], params).cpu()
            put_preserve(i, params)
        ref = ref_states.get(i)
        if ref is not None and have_linears and i not in bad_rows:
            x = state[i].view(-1, state[i].shape[-1]).float()
            y = ref.view(-1, ref.shape[-1]).float()
            error += (torch_norm(x - y) / torch_norm(y)).item()
            measured += 1
            ref_states[i] = None
    return error / max(measured, 1), measured


def _set_new_tensors(stc, tensors) -> None:
    """Set the in-memory reload layer on either a plain or variant STC.

    VariantSafetensorsCollection intentionally leaves ``set_new_tensors``
    unimplemented.  Dense keys resolve to its ``main`` collection, so the
    converter's reload operation is applied to that collection only; the
    routed-expert override remains the D2 collection.
    """
    target = getattr(stc, "main", stc)
    target.set_new_tensors(tensors)


def _inject_reload_keys(stc, tensors) -> list[str]:
    """Make variant ``has_tensor_group`` see in-memory EXL3 keys.

    The audited VariantSafetensorsCollection delegates ``get_tensor`` but its
    ``has_tensor_group`` checks only ``tensor_file_map``.  Adding temporary
    sentinels lets Linear.load choose its EXL3 branch; get_tensor still takes
    the actual values from ``new_tensors``.  The sentinels are removed before
    the collection is closed.
    """
    main = getattr(stc, "main", stc)
    added = []
    for key in tensors:
        if key not in main.tensor_file_map:
            main.tensor_file_map[key] = "__dense_bake_memory__"
            added.append(key)
    return added


def _remove_reload_keys(stc, added: list[str]) -> None:
    main = getattr(stc, "main", stc)
    for key in added:
        main.tensor_file_map.pop(key, None)


def torch_norm(tensor):
    # Kept as a tiny late-bound helper so importing this tool for header-only
    # merge planning does not initialize torch/CUDA.
    import torch
    return torch.linalg.norm(tensor, "fro")


def _checkpoint(runtime, work: Path, job: dict, state, original_ids, args_dict: dict):
    ckpt = work / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    runtime.save_tensor(state, "ckpt/state.safetensors", args_dict)
    runtime.save_tensor(original_ids, "ckpt/original_input_ids.safetensors", args_dict)
    (ckpt / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")


def _read_job(work: Path) -> dict:
    path = work / "ckpt" / "job.json"
    if not path.is_file():
        return {"next_module_idx": 0, "bad_rows": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid checkpoint {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("next_module_idx"), int):
        raise RuntimeError(f"checkpoint {path} has no integer next_module_idx")
    return data


def _prepare_real_state(runtime, experts: Path, args, model, job, work: Path):
    import torch
    args_dict = vars(args).copy()
    args_dict["work_dir"] = str(work)
    args_dict["apply_out_scales"] = True
    args_dict["image_dump"] = False
    args_dict["codebook"] = "mcg"
    args_dict["verbose"] = args.verbose
    quant_preserves = []
    bad_rows = set(job.get("bad_rows") or [])
    if job["next_module_idx"] > 0:
        state = runtime.load_tensor("ckpt/state.safetensors", args_dict)
        original_ids = runtime.load_tensor("ckpt/original_input_ids.safetensors", args_dict)
        if not isinstance(state, list) or not isinstance(original_ids, list):
            raise RuntimeError("checkpoint state/original_input_ids must both be tensor lists")
        quant_preserves = [{} for _ in state]
        return state, original_ids, quant_preserves, bad_rows, args_dict

    tokenizer_config = runtime.Config.from_directory(str(experts))
    tokenizer = runtime.Tokenizer.from_config(tokenizer_config)
    original_ids = runtime.get_default_calibration(
        {"cal_rows": args.cal_rows, "cal_cols": args.cal_cols}, tokenizer
    )
    state = list(original_ids)
    quant_preserves = [{} for _ in state]
    # The converter starts with token rows and advances them through embed and
    # hc_expand before entering layers.N.  These modules are intentionally not
    # quantized or written to qtensors.
    for idx, module in enumerate(model.modules[: model.first_block_idx]):
        log(f"LOAD_PREFIX module={module.key}")
        _load_module(module, torch.device(f"cuda:{args.devices[0]}"))
        for i in range(len(state)):
            params = {"attn_mode": "flash_attn_nc", "input_ids": original_ids[i]}
            state[i] = module.forward(module.prepare_for_device(state[i], params), params).cpu()
        module.unload()
    return state, original_ids, quant_preserves, bad_rows, args_dict


def bake(args: argparse.Namespace) -> int:
    src = Path(args.src).expanduser().resolve()
    experts = Path(args.experts).expanduser().resolve()
    work = Path(args.work).expanduser().resolve()
    if not src.is_dir() or not experts.is_dir():
        raise RuntimeError(f"--src and --experts must be directories: {src}, {experts}")
    if args.bits < 1 or args.bits > 8 or any(
        value is not None and (value < 1 or value > 8)
        for value in (args.attn_bits, args.shared_bits)
    ):
        raise ValueError("bits must be in the range 1..8")
    devices = _parse_devices(args.devices)
    if not args.dry_run:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("real bake requires CUDA; use --dry-run for the offline checks")
        count = torch.cuda.device_count()
        if any(device >= count for device in devices):
            raise RuntimeError(f"requested CUDA devices {devices}, but only {count} are visible")

    runtime = _runtime_import(dry_run=args.dry_run)
    config, model = _load_variant_model(src, experts, runtime)
    plan = _module_plan(model, runtime.Linear)
    log(
        f"BAKE_PLAN layers={len(plan)} dense_linears={sum(len(t) for _, _, t in plan)} "
        f"bits={args.bits:g} attn_bits={(args.attn_bits if args.attn_bits is not None else args.bits):g} "
        f"shared_bits={(args.shared_bits if args.shared_bits is not None else args.bits):g} codebook=mcg"
    )
    if args.dry_run:
        for _, module, targets in plan[:2]:
            log(f"DRY_RUN_BAKE module={module.key} targets={len(targets)}")
        log(f"DRY_RUN_BAKE module={plan[-1][1].key} targets={len(plan[-1][2])}")
        if args.merge:
            merge_plan = plan_merge(work, experts, args)
            log(f"DRY_RUN_MERGE dropped={len(merge_plan['dropped'])} new={len(merge_plan['new_plan'])}")
        return 0

    work.mkdir(parents=True, exist_ok=True)
    (work / "qtensors").mkdir(exist_ok=True)
    (work / "debug").mkdir(exist_ok=True)
    signature = {
        "src": str(src), "experts": str(experts), "bits": args.bits,
        "attn_bits": args.attn_bits, "shared_bits": args.shared_bits,
        "cal_rows": args.cal_rows, "cal_cols": args.cal_cols, "codebook": "mcg",
    }
    signature_path = work / "args.json"
    if args.resume:
        if not signature_path.is_file():
            raise RuntimeError(f"--resume requested but {signature_path} is missing")
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError("resume settings differ from work/args.json")
    else:
        qtensors_dir = work / "qtensors"
        existing_qtensors = qtensors_dir.is_dir() and any(qtensors_dir.glob("layers.*.safetensors"))
        if signature_path.exists() or (work / "ckpt" / "job.json").exists() or existing_qtensors:
            raise RuntimeError("work directory already contains a bake; use --resume or choose a new --work path")
        signature_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")

    job = _read_job(work) if args.resume else {"next_module_idx": 0, "bad_rows": []}
    state, original_ids, quant_preserves, bad_rows, args_dict = _prepare_real_state(
        runtime, experts, args, model, job, work
    )
    job["cal_rows"] = args.cal_rows
    job["cal_cols"] = args.cal_cols
    job["codebook"] = "mcg"
    job["bad_rows"] = sorted(bad_rows)
    parallel = len(devices) > 1
    replicas = [runtime.Model.from_config(config) for _ in devices[1:]] if parallel else []
    for replica in replicas:
        replica.calibration_all_experts = True
    import torch

    for module_idx, module, targets in plan:
        if module_idx < job["next_module_idx"]:
            continue
        layer = _layer_number(module.key)
        log(f"BAKE_LAYER start={module.key} targets={len(targets)}")
        device = torch.device(f"cuda:{devices[0]}")
        _load_module(module, device)
        # Recalculate recursive linears after load; dense targets must be FP16
        # here, while the routed expert variant is already EXL3.
        targets = [m for m in module if isinstance(m, runtime.Linear) and is_dense_key(m.key)]
        if len(targets) == 0 or any(getattr(m, "inner", None).__class__.__name__ != "LinearFP16" for m in targets):
            raise RuntimeError(f"{module.key}: a dense target did not load as LinearFP16")
        if any(m.qmap is None for m in targets):
            raise RuntimeError(f"{module.key}: a dense target has no Hessian qmap")
        keys = [m.key for m in targets]
        if sorted(keys) != sorted(dense_keys_for_layer(layer, keys)):  # LNA-LAB: module order is not lexical
            raise RuntimeError(f"{module.key}: dense target ordering/filter changed after load; got={keys} expected={dense_keys_for_layer(layer, keys)}")

        capture_replicas = None
        if parallel:
            capture_replicas = runtime.load_parallel_calib_modules(
                replicas, module_idx, devices, None
            )
        if capture_replicas is not None:
            def get_preserve(i, params):
                params.update(quant_preserves[i])
                params["quant_preserve"] = quant_preserves[i]
            def put_preserve(i, params):
                quant_preserves[i] = params["quant_preserve"]
            capture_h, ref_states = runtime.capture_module_parallel(
                model, [module] + capture_replicas, devices, None, state,
                original_ids, get_preserve, put_preserve, False, 0,
                f"CAPTURE {module.key}", bad_rows,
            )
            for replica in capture_replicas:
                replica.unload()
        else:
            capture_h, ref_states, get_preserve, put_preserve = _capture_serial(
                model, module, state, original_ids, quant_preserves, bad_rows
            )

        strategy = {m.key: bits_for_key(m.key, args.bits, args.attn_bits, args.shared_bits) for m in targets}
        for linear in targets:
            linear.inner.swap_cpu()
        if len(targets) >= len(devices) and all(bits <= 8 for bits in strategy.values()):
            runtime.quantize_linears_parallel(
                args_dict, targets, config, strategy, module_idx, devices, None, capture_h, state
            )
        else:
            runtime.quantize_linears_single(
                args_dict, targets, config, strategy, module_idx, devices, None, capture_h, state
            )
        qtensors = {}
        for linear in targets:
            qtensors.update(linear.get_tensors())
        expected = {f"{m.key}.{suffix}" for m in targets for suffix in EXL3_SUFFIXES}
        if set(qtensors) != expected:
            raise RuntimeError(f"{module.key}: converter emitted unexpected tensors")
        runtime.save_tensor(qtensors, f"qtensors/layers.{layer}.safetensors", args_dict)
        module.unload()
        reload_keys = _inject_reload_keys(config.stc, qtensors)
        _set_new_tensors(config.stc, qtensors)
        _load_module(module, device, source=qtensors, close=False)
        advance_replicas = None
        if parallel:
            advance_replicas = runtime.load_parallel_calib_modules(
                replicas, module_idx, devices, None, source=qtensors
            )
        _set_new_tensors(config.stc, None)
        _remove_reload_keys(config.stc, reload_keys)
        if advance_replicas is not None:
            def get_preserve(i, params):
                params.update(quant_preserves[i])
                params["quant_preserve"] = quant_preserves[i]
            def put_preserve(i, params):
                quant_preserves[i] = params["quant_preserve"]
            runtime.advance_state_parallel(
                model, [module] + advance_replicas, devices, None, state,
                original_ids, get_preserve, put_preserve, ref_states, True,
                module_idx == plan[-1][0], f"ADVANCE {module.key}", bad_rows,
            )
            for replica in advance_replicas:
                replica.unload()
        else:
            _advance_serial(model, module, state, original_ids, quant_preserves, ref_states, bad_rows, True)
        module.unload()
        job["next_module_idx"] = module_idx + 1
        job["bad_rows"] = sorted(bad_rows)
        job["completed_layers"] = sorted(set(job.get("completed_layers", [])) | {layer})
        _checkpoint(runtime, work, job, state, original_ids, args_dict)
        log(f"BAKE_LAYER done={module.key} checkpoint=after_layer_{layer}")
    log(f"BAKE_COMPLETE work={work} layers={len(plan)}")
    return 0


def _read_work_tensors(work: Path) -> list[dict[str, Any]]:
    qtensors = work / "qtensors"
    if not qtensors.is_dir():
        raise RuntimeError(f"missing work qtensors directory: {qtensors}")
    result = []
    seen = set()
    for path in sorted(qtensors.glob("layers.*.safetensors"), key=lambda p: p.name):
        match = _LAYER_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        header_len, header = overlay.read_header(path)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not is_dense_key(name.rsplit(".", 1)[0]) or name.rsplit(".", 1)[1] not in EXL3_SUFFIXES:
                raise RuntimeError(f"work file contains non-dense/non-EXL3 tensor: {path.name}:{name}")
            base, suffix = name.rsplit(".", 1)
            if _layer_number(base) != int(match.group("layer")):
                raise RuntimeError(f"tensor {name} is in the wrong work layer file {path.name}")
            if name in seen:
                raise RuntimeError(f"duplicate work tensor: {name}")
            expected_dtype = EXL3_DTYPES[suffix]
            if meta.get("dtype") != expected_dtype:
                raise RuntimeError(f"{name}: expected dtype {expected_dtype}, got {meta.get('dtype')}")
            seen.add(name)
            result.append({"target": name, "base": base, "suffix": suffix,
                           "meta": meta, "path": path, "header_len": header_len})
    if not result:
        raise RuntimeError(f"no complete dense EXL3 tensors under {qtensors}")
    grouped = {}
    for item in result:
        grouped.setdefault(item["base"], set()).add(item["suffix"])
    incomplete = [base for base, suffixes in grouped.items() if suffixes != set(EXL3_SUFFIXES)]
    if incomplete:
        raise RuntimeError(f"work tensors have incomplete EXL3 groups, e.g. {incomplete[0]}")
    for layer in sorted({_layer_number(base) for base in grouped}):
        wo_a = {
            int(base.rsplit(".", 1)[1])
            for base in grouped
            if _layer_number(base) == layer and _WO_A_RE.fullmatch(base.split(".", 2)[2])
        }
        if wo_a and wo_a != set(range(8)):
            raise RuntimeError(f"work layer {layer} has incomplete wo_a slices: {sorted(wo_a)}")
    return sorted(result, key=lambda item: item["target"])


def _vllm_prefixes(layer: int, root: str, shared_prefix: str) -> dict[str, str]:
    base = f"{root}.layers.{layer}"
    return {
        "attn.wq_a": f"{base}.attn.fused_wqa_wkv",
        "attn.wkv": f"{base}.attn.fused_wqa_wkv",
        "attn.wq_b": f"{base}.attn.wq_b",
        "attn.wo_b": f"{base}.attn.wo_b",
        "attn.compressor.wkv": f"{base}.attn.compressor.fused_wkv_wgate",
        "attn.compressor.wgate": f"{base}.attn.compressor.fused_wkv_wgate",
        "attn.indexer.compressor.wkv": f"{base}.attn.indexer.compressor.fused_wkv_wgate",
        "attn.indexer.compressor.wgate": f"{base}.attn.indexer.compressor.fused_wkv_wgate",
        "attn.indexer.wq_b": f"{base}.attn.indexer.wq_b",
        "ffn.shared_experts.w1": f"{base}.{shared_prefix}.gate_up_proj",
        "ffn.shared_experts.w3": f"{base}.{shared_prefix}.gate_up_proj",
        "ffn.shared_experts.w2": f"{base}.{shared_prefix}.down_proj",
    }


def _config_with_vllm_block(base_config: dict, work_items: list[dict], args) -> dict:
    config = copy.deepcopy(base_config)
    q = config.setdefault("quantization_config", {})
    layers = {}
    for item in work_items:
        base = item["base"]
        tail = base.split(".", 2)[2]
        if _WO_A_RE.fullmatch(tail):
            # LNA-LAB: the plugin consumes one already-sliced wo_a per TP rank.
            vllm = f"{args.vllm_root}.layers.{_layer_number(base)}.attn.wo_a"
        else:
            vllm = _vllm_prefixes(
                _layer_number(base), args.vllm_root, args.vllm_shared_prefix
            ).get(tail)
        if vllm is None:
            raise RuntimeError(f"no vLLM prefix mapping for {base}")
        bits = args.attn_bits if dense_kind(base) == "attn" and args.attn_bits is not None else \
            args.shared_bits if dense_kind(base) == "shared" and args.shared_bits is not None else args.bits
        bits = _schema_bits(bits)
        old = layers.setdefault(vllm, {"bits": bits})
        if old["bits"] != bits:
            raise RuntimeError(f"fused vLLM module has mixed bits: {vllm}")
    # LNA-LAB: unmatched dense modules (notably indexer.weights_proj) stay BF16.
    q.setdefault("non_routed_dtype_policy", "bf16_as_stored")
    q["non_routed_exl3"] = {
        "codebook": "mcg",
        "layers": dict(sorted(layers.items())),
    }
    return config


def _write_dense_shard(out: Path, items: list[dict]) -> dict[str, Any]:
    destination = out / DENSE_SHARD
    partial = out / (DENSE_SHARD + ".partial")
    entries = [(item["target"], item["meta"]) for item in items]
    with partial.open("wb") as dst:
        header_len, _ = overlay.common.write_safetensors_header(
            dst, entries, {"format": "pt", "source": "exllamav3-dense-bake", "codebook": "mcg"}
        )
        for item in items:
            with item["path"].open("rb") as src:
                overlay.common.copy_payload(src, dst, 8 + item["header_len"], item["meta"])
    os.replace(partial, destination)
    return {"name": DENSE_SHARD, "header_len": header_len, "entries": entries}


def _link_base_files(experts: Path, out: Path, rewrite_files: set[str]) -> None:
    excluded = rewrite_files | {CONFIG_NAME, INDEX_NAME, DENSE_SHARD, DENSE_SHARD + ".partial"}
    for entry in sorted(experts.iterdir(), key=lambda p: p.name):
        if entry.name in excluded or not (entry.is_file() or entry.is_symlink()):
            continue
        destination = out / entry.name
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() and destination.resolve() == entry.resolve():
                continue
            raise RuntimeError(f"output path already exists: {destination}")
        destination.symlink_to(entry.resolve())


def plan_merge(work: Path, experts: Path, args, *, require_complete: bool = True) -> dict[str, Any]:
    work_items = _read_work_tensors(work)
    source, source_map = overlay.source_headers(experts)
    overlay.validate_source_index(experts, source_map)
    dense_bases = sorted({item["base"] for item in work_items})
    expected_bases = set()
    for name in source:
        if not name.endswith(".weight"):
            continue
        base = name[:-len(".weight")]
        if base.startswith("layers.") and base.endswith(".attn.wo_a"):
            layer = _layer_number(base)
            expected_bases.update(f"layers.{layer}.attn.wo_a.slice.{i}" for i in range(8))
        elif is_dense_key(base):
            expected_bases.add(base)
    missing_work = sorted(expected_bases - set(dense_bases))
    extra_work = sorted(set(dense_bases) - expected_bases)
    if (missing_work or extra_work) and require_complete:
        raise RuntimeError(
            "work directory is not a complete dense bake: "
            f"missing={missing_work[:3]} extra={extra_work[:3]}"
        )
    if missing_work or extra_work:
        log(
            "WARNING dry-run merge is partial: "
            f"missing={missing_work[:3]} extra={extra_work[:3]}"
        )
    dropped = {f"{base}.weight" for base in dense_bases}
    missing = sorted(name for name in dropped if name not in source)
    if missing:
        raise RuntimeError(f"dense work tensor has no BF16/fp8 source weight to replace: {missing[0]}")
    by_file: dict[str, set[str]] = {}
    for name in dropped:
        by_file.setdefault(source[name][0].name, set()).add(name)
    rewrite_files = set(by_file)
    new_plan = [{"target": item["target"], "meta": item["meta"]} for item in work_items]
    base_config_path = experts / CONFIG_NAME
    try:
        base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {base_config_path}: {exc}") from exc
    config = _config_with_vllm_block(base_config, work_items, args)
    index = overlay.output_index(source, dropped, new_plan)
    # layer_overlay's generic planner uses its historical shard name; T4 has
    # a distinct dense shard so it cannot be confused with a layer overlay.
    index["weight_map"] = {
        name: (DENSE_SHARD if filename == overlay.NEW_SHARD else filename)
        for name, filename in index["weight_map"].items()
    }
    return {
        "source": source,
        "source_map": source_map,
        "items": work_items,
        "dropped": dropped,
        "rewrite_files": rewrite_files,
        "new_plan": new_plan,
        "config": config,
        "index": index,
    }


def _verify_merge(experts: Path, out: Path, plan: dict[str, Any]) -> None:
    index_path = out / INDEX_NAME
    if json.loads(index_path.read_text(encoding="utf-8")) != plan["index"]:
        raise RuntimeError("merge index differs from plan")
    output_headers = {}
    for filename in sorted(set(plan["index"]["weight_map"].values())):
        path = out / filename
        if not path.is_file():
            raise RuntimeError(f"indexed output shard missing: {path}")
        header_len, header = overlay.read_header(path)
        expected_names = {
            name for name, mapped in plan["index"]["weight_map"].items() if mapped == filename
        }
        actual_names = {name for name in header if name != "__metadata__"}
        if actual_names != expected_names:
            raise RuntimeError(
                f"output shard {filename} has unexpected tensor set: "
                f"missing={sorted(expected_names - actual_names)[:2]} "
                f"extra={sorted(actual_names - expected_names)[:2]}"
            )
        for name, mapped in plan["index"]["weight_map"].items():
            if mapped == filename:
                if name not in header:
                    raise RuntimeError(f"indexed tensor missing from {filename}: {name}")
                output_headers[name] = (path, header_len, header)
    if any(name in plan["dropped"] for name in output_headers):
        raise RuntimeError("dropped source tensor remains in merged output")
    for filename in plan["rewrite_files"]:
        path = out / filename
        if path.is_symlink():
            raise RuntimeError(f"rewritten shard is still a symlink: {filename}")
        _, header = overlay.read_header(path)
        if any(name in plan["dropped"] for name in header):
            raise RuntimeError(f"dropped tensor remains in rewritten shard: {filename}")
    for name, (source_path, source_header_len, source_header) in plan["source"].items():
        if name in plan["dropped"] or source_path.name not in plan["rewrite_files"]:
            continue
        output_path, output_header_len, output_header = output_headers[name]
        if source_header[name].get("dtype") != output_header[name].get("dtype") or \
                source_header[name].get("shape") != output_header[name].get("shape"):
            raise RuntimeError(f"retained tensor metadata changed: {name}")
        overlay.compare_payloads(source_path, output_path, source_header[name], output_header[name],
                                 source_header_len, output_header_len)
    for item in plan["items"]:
        path, header_len, header = output_headers[item["target"]]
        meta = header[item["target"]]
        if meta.get("dtype") != item["meta"].get("dtype") or meta.get("shape") != item["meta"].get("shape"):
            raise RuntimeError(f"new tensor metadata changed: {item['target']}")
        if item["target"].endswith(".mcg"):
            start, end = meta["data_offsets"]
            with path.open("rb") as stream:
                stream.seek(8 + header_len + start)
                if struct.unpack("<i", stream.read(end - start))[0] != MCG_MARKER:
                    raise RuntimeError(f"MCG marker mismatch: {item['target']}")
    if json.loads((out / CONFIG_NAME).read_text(encoding="utf-8")) != plan["config"]:
        raise RuntimeError("merge config differs from plan")
    for entry in experts.iterdir():
        if entry.name in plan["rewrite_files"] or entry.name in {CONFIG_NAME, INDEX_NAME, DENSE_SHARD}:
            continue
        if not (entry.is_file() or entry.is_symlink()):
            continue
        linked = out / entry.name
        if not linked.is_symlink() or linked.resolve() != entry.resolve():
            raise RuntimeError(f"unchanged source file is not linked unchanged: {entry.name}")
    log(f"MERGE_VERIFY_OK indexed_tensors={len(output_headers)} dropped_absent=true retained_payloads_byte_identical=true")


def merge(args: argparse.Namespace) -> int:
    work = Path(args.work).expanduser().resolve()
    experts = Path(args.experts).expanduser().resolve()
    out = Path(args.merge).expanduser().resolve()
    overlay.safe_output_path(experts, out)
    plan = plan_merge(work, experts, args, require_complete=not args.dry_run)
    log(f"MERGE_PLAN rewrite_files={len(plan['rewrite_files'])} dropped={len(plan['dropped'])} new={len(plan['new_plan'])}")
    if args.dry_run:
        log(f"DRY_RUN_MERGE output={out} symlinks=all_unchanged_source_files")
        return 0
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to merge into non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    _link_base_files(experts, out, plan["rewrite_files"])
    for filename in sorted(plan["rewrite_files"]):
        overlay.rewrite_source_shard(experts / filename, out / filename, plan["dropped"])
    _write_dense_shard(out, plan["items"])
    overlay.write_output_metadata(out, plan["index"], plan["config"])
    _verify_merge(experts, out, plan)
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="bf16/fp8 DeepSeek-V4 source checkpoint")
    parser.add_argument("--experts", required=True, help="existing routed-expert EXL3 pack")
    parser.add_argument("--work", required=True, help="resumable qtensors/checkpoint directory")
    parser.add_argument("--merge", help="write merged overlay pack to this directory")
    parser.add_argument("--bits", type=float, default=4, help="dense default EXL3 K (default: 4)")
    parser.add_argument("--attn-bits", type=float, default=None)
    parser.add_argument("--shared-bits", type=float, default=None)
    parser.add_argument("--cal-rows", type=int, default=250)
    parser.add_argument("--cal-cols", type=int, default=2048)
    parser.add_argument("--devices", default="0,1", help="CUDA device IDs, e.g. 0,1")
    parser.add_argument("--resume", action="store_true", help="resume completed module checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="offline layout/merge planning; no writes")
    parser.add_argument("--vllm-root", default="model", help="vLLM model root prefix")
    parser.add_argument("--vllm-shared-prefix", default="mlp.shared_experts",
                        help="vLLM shared-expert child prefix under model.layers.N")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.merge:
            # A merge is independent of CUDA; --dry-run additionally performs
            # the requested source/model layout check before header planning.
            if args.dry_run:
                dry_run_layout(Path(args.src).expanduser().resolve(), Path(args.experts).expanduser().resolve())
            return merge(args)
        import torch  # LNA-LAB: the converter runs under inference_mode; loaded tensors are inference tensors
        with torch.inference_mode():
            return bake(args)
    except (RuntimeError, ValueError, OSError) as exc:
        log(f"ERROR {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
