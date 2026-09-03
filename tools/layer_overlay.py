#!/usr/bin/env python3
"""Build a selected-main-layer K2 EXL3 overlay.

The source pack is never changed.  Unchanged source files are linked into the
output, source shards containing the selected K3 routed experts are rewritten
tensor-by-tensor, and the replacement K2 tensors are written to one new shard.
Donor tensors are fetched with HTTP Range requests one tensor at a time; a
local donor-shard directory can be supplied through ``LAYER_OVERLAY_LOCAL_DIR``.

``--dry-run`` is deliberately offline: it scans the local source headers and
donor index and plans the replacement using a K2 ABI exemplar already in the
source pack.  A donor header cache, when supplied, adds header-level shape and
dtype validation without network access.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import BinaryIO, Iterable

try:
    from . import mtp_overlay as common
except ImportError:  # Running as ``python tools/layer_overlay.py``.
    import mtp_overlay as common


CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
NEW_SHARD = "model-layers-k2.safetensors"
DEFAULT_DONOR_INDEX = "/run/media/tonoken3/DATA1/.tmp/k2v1-index.json"
DEFAULT_HEADER_CACHE = "/run/media/tonoken3/DATA1/.tmp/layer-overlay-donor-headers.json"
CHUNK_SIZE = common.CHUNK_SIZE
RETRY_COUNT = common.RETRY_COUNT

DTYPE_BYTES = common.DTYPE_BYTES
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg")
EXL3_DTYPE = {"trellis": "I16", "suh": "F16", "svh": "F16", "mcg": "I32"}
PROJECTION_MAP = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
PROJECTION_ORDER = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}

SOURCE_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>[0-9]+)\.ffn\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>w[123])\."
    r"(?P<suffix>trellis|suh|svh|mcg)$"
)
DONOR_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<suffix>trellis|suh|svh|mcg)$"
)

# These helpers are intentionally re-exported for small offline tests and for
# callers that used the corresponding mtp_overlay helpers.
parse_header = common.parse_header
validate_header = common.validate_header
read_header = common.read_header
tensor_nbytes = common.tensor_nbytes


def log(message: str) -> None:
    print(message, flush=True)


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid --layers value {value!r}") from exc
    if not values or any(layer < 0 for layer in values):
        raise ValueError(f"--layers must contain non-negative layer numbers: {value!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"--layers contains duplicates: {value!r}")
    return tuple(sorted(values))


def source_layer(name: str) -> int | None:
    match = SOURCE_EXPERT_RE.fullmatch(name)
    return None if match is None else int(match.group("layer"))


def donor_layer(name: str) -> int | None:
    match = DONOR_EXPERT_RE.fullmatch(name)
    return None if match is None else int(match.group("layer"))


def map_donor_name(name: str) -> str:
    """Map one donor main-layer expert name to the target pack ABI."""
    match = DONOR_EXPERT_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"not a donor main-layer EXL3 expert tensor: {name}")
    return (
        f"layers.{match.group('layer')}.ffn.experts.{int(match.group('expert'))}."
        f"{PROJECTION_MAP[match.group('projection')]}.{match.group('suffix')}"
    )


# Short alias used by offline tests and useful to importers.
map_name = map_donor_name


def is_donor_expert_name(name: str) -> bool:
    return DONOR_EXPERT_RE.fullmatch(name) is not None


def is_source_expert_name(name: str) -> bool:
    return SOURCE_EXPERT_RE.fullmatch(name) is not None


def donor_base_url(donor: str) -> str:
    return donor.rstrip("/") if donor.startswith("http") else f"https://huggingface.co/{donor}/resolve/main"


def local_or_remote_bytes(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    """Read a range from a local donor shard or a remote URL.

    The real build calls this only for header ranges and selected tensor
    ranges.  It never requests a complete donor shard.
    """
    local_dir = os.environ.get("LAYER_OVERLAY_LOCAL_DIR")
    filename = url.split("?", 1)[0].rsplit("/", 1)[-1]
    if local_dir:
        local_path = Path(local_dir).expanduser() / filename
        if local_path.is_file():
            with local_path.open("rb") as stream:
                if byte_range is None:
                    return stream.read()
                stream.seek(byte_range[0])
                data = stream.read(byte_range[1] - byte_range[0] + 1)
            expected = byte_range[1] - byte_range[0] + 1
            if len(data) != expected:
                raise RuntimeError(
                    f"short local range read {local_path}: got {len(data)}, expected {expected}"
                )
            return data

    for attempt in range(RETRY_COUNT):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "layer_overlay/1"})
            if byte_range is not None:
                request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                if byte_range is not None and status != 206:
                    raise RuntimeError(f"expected HTTP 206 for Range request, got {status}")
                data = response.read()
            expected = None if byte_range is None else byte_range[1] - byte_range[0] + 1
            if expected is not None and len(data) != expected:
                raise RuntimeError(f"short range read: got {len(data)}, expected {expected}")
            return data
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
            if attempt == RETRY_COUNT - 1:
                raise RuntimeError(f"failed fetching {url} range={byte_range}: {exc}") from exc
            delay = 2**attempt
            log(f"RETRY attempt={attempt + 1}/{RETRY_COUNT} range={byte_range}: {exc}")
            time.sleep(delay)
    raise AssertionError("unreachable")


def donor_index_from_file(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read donor index {path}: {exc}") from exc
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        raise RuntimeError(f"donor index {path} has no valid weight_map")
    return weight_map


def source_headers(src: Path) -> tuple[dict[str, tuple[Path, int, dict]], dict[str, str]]:
    """Scan every local safetensors file, including an existing overlay shard."""
    candidates = sorted(
        (path for path in src.iterdir() if path.name.endswith(".safetensors") and path.is_file()),
        key=lambda path: path.name,
    )
    if not candidates:
        raise RuntimeError(f"no safetensors shards found under {src}")
    headers: dict[str, tuple[Path, int, dict]] = {}
    weight_map: dict[str, str] = {}
    for path in candidates:
        header_len, header = read_header(path)
        for name in header:
            if name == "__metadata__":
                continue
            if name in weight_map:
                raise RuntimeError(f"duplicate source tensor {name!r}")
            headers[name] = (path, header_len, header)
            weight_map[name] = path.name
    log(f"SOURCE_HEADERS files={len(candidates)} tensors={len(headers)}")
    return headers, weight_map


def validate_source_index(src: Path, source_map: dict[str, str]) -> None:
    """Cross-check the numbered source headers against an existing index."""
    path = src / INDEX_NAME
    if not path.is_file():
        log("SOURCE_INDEX absent=allowed; numbered headers are authoritative")
        return
    try:
        data = json.loads(path.read_text())
        index_map = data["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"invalid source index {path}: {exc}") from exc
    if not isinstance(index_map, dict):
        raise RuntimeError(f"source index {path} has no object weight_map")
    if set(index_map) != set(source_map):
        raise RuntimeError(
            f"source index tensor set differs from local headers: "
            f"missing={sorted(set(source_map) - set(index_map))[:5]} "
            f"extra={sorted(set(index_map) - set(source_map))[:5]}"
        )
    for name, filename in source_map.items():
        if index_map.get(name) != filename:
            raise RuntimeError(f"source index disagrees with header for {name}: {index_map.get(name)!r} != {filename!r}")
    log(f"SOURCE_INDEX checked=local_headers tensors={len(source_map)} indexed_total={len(index_map)}")


def _abi_meta(meta: dict) -> dict:
    return {"dtype": meta.get("dtype"), "shape": list(meta.get("shape", []))}


def validate_abi(abi: dict[str, dict[str, dict]], label: str) -> None:
    for projection in ("w1", "w2", "w3"):
        if projection not in abi:
            raise RuntimeError(f"{label}: missing projection {projection}")
        for suffix in EXL3_SUFFIXES:
            meta = abi[projection].get(suffix)
            if not isinstance(meta, dict) or meta.get("dtype") != EXL3_DTYPE[suffix]:
                raise RuntimeError(f"{label}: unexpected {projection}.{suffix} metadata: {meta}")
            shape = meta.get("shape")
            if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
                raise RuntimeError(f"{label}: invalid {projection}.{suffix} shape: {shape}")
        if abi[projection]["trellis"]["shape"][-1:] != [32]:
            raise RuntimeError(f"{label}: {projection}.trellis is not K2: {abi[projection]['trellis']}")


def find_source_k2_abi(
    source: dict[str, tuple[Path, int, dict]],
    selected_layers: Iterable[int],
) -> tuple[dict[str, dict[str, dict]], int]:
    """Use one unselected local main layer as the expected K2 ABI."""
    selected = set(selected_layers)
    available = sorted({layer for name in source if (layer := source_layer(name)) is not None})
    for layer in available:
        if layer in selected:
            continue
        abi: dict[str, dict[str, dict]] = {}
        complete = True
        for projection in ("w1", "w2", "w3"):
            abi[projection] = {}
            for suffix in EXL3_SUFFIXES:
                name = f"layers.{layer}.ffn.experts.0.{projection}.{suffix}"
                if name not in source:
                    complete = False
                    break
                abi[projection][suffix] = _abi_meta(source[name][2][name])
            if not complete:
                break
        if complete:
            try:
                validate_abi(abi, f"source layer {layer} K2 exemplar")
            except RuntimeError:
                continue
            log(f"MAIN_K2_ABI exemplar_layer={layer} validated=true")
            return abi, layer
    raise RuntimeError("could not find an unselected K2 main-layer ABI exemplar in the source pack")


def expected_names(layers: Iterable[int], *, donor: bool) -> set[str]:
    names = set()
    for layer in layers:
        for expert in range(256):
            for projection in ("gate_proj", "up_proj", "down_proj") if donor else ("w1", "w2", "w3"):
                for suffix in EXL3_SUFFIXES:
                    if donor:
                        names.add(f"model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}")
                    else:
                        names.add(f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}")
    return names


def validate_source_selection(
    source: dict[str, tuple[Path, int, dict]],
    selected_layers: tuple[int, ...],
    k2_abi: dict[str, dict[str, dict]],
) -> tuple[set[str], dict[str, list[tuple[str, dict]]], int]:
    """Validate selected source layers are complete K3 sets and plan drops."""
    selected = set(selected_layers)
    expected = expected_names(selected_layers, donor=False)
    actual = {name for name in source if source_layer(name) in selected}
    if actual != expected:
        raise RuntimeError(
            f"source selected expert set mismatch: actual={len(actual)} "
            f"expected={len(expected)} missing={sorted(expected - actual)[:5]} "
            f"extra={sorted(actual - expected)[:5]}"
        )
    for name in source:
        raw_layer = re.match(r"^layers\.(?P<layer>[0-9]+)\.ffn\.experts\.", name)
        if (
            raw_layer is not None
            and int(raw_layer.group("layer")) in selected
            and not is_source_expert_name(name)
        ):
            raise RuntimeError(f"malformed selected source expert name: {name}")

    dropped: set[str] = set()
    by_file: dict[str, list[tuple[str, dict]]] = {}
    for name in sorted(actual):
        path, _, header = source[name]
        meta = header[name]
        match = SOURCE_EXPERT_RE.fullmatch(name)
        assert match is not None
        expert = int(match.group("expert"))
        projection = match.group("projection")
        suffix = match.group("suffix")
        if expert >= 256:
            raise RuntimeError(f"selected source expert id out of range: {name}")
        if meta.get("dtype") != EXL3_DTYPE[suffix]:
            raise RuntimeError(f"selected source dtype mismatch for {name}: {meta}")
        shape = meta.get("shape")
        expected_shape = list(k2_abi[projection][suffix]["shape"])
        if suffix == "trellis":
            if not isinstance(shape, list) or shape[:-1] != expected_shape[:-1] or shape[-1:] != [48]:
                raise RuntimeError(f"selected source is not K3 for {name}: {meta}")
        elif shape != expected_shape:
            raise RuntimeError(f"selected source ABI mismatch for {name}: {meta}, expected {k2_abi[projection][suffix]}")
        dropped.add(name)
        by_file.setdefault(path.name, []).append((name, meta))

    expected_per_layer = 256 * 3 * 4
    for layer in selected_layers:
        count = sum(1 for name in actual if source_layer(name) == layer)
        if count != expected_per_layer:
            raise RuntimeError(f"source layer {layer}: found {count} expert tensors, expected {expected_per_layer}")
    for filename, items in sorted(by_file.items()):
        # A shard may contain one or more selected layers; every complete layer
        # contributes exactly 256 * 3 * 4 expert tensors.
        layers_here = {source_layer(name) for name, _ in items}
        expected_count = expected_per_layer * len(layers_here)
        if len(items) != expected_count:
            raise RuntimeError(f"{filename}: found {len(items)} selected tensors, expected {expected_count}")
    drop_bytes = sum(tensor_nbytes(meta) for items in by_file.values() for _, meta in items)
    log(
        f"SOURCE_DROP tensors={len(dropped)} bytes={drop_bytes} "
        f"files={sorted(by_file)} layers={list(selected_layers)}"
    )
    return dropped, by_file, drop_bytes


def validate_donor_selection(weight_map: dict[str, str], layers: tuple[int, ...]) -> list[str]:
    selected = set(layers)
    candidates = [
        name for name in weight_map
        if name.startswith("model.layers.") and ".mlp.experts." in name
    ]
    malformed = sorted(name for name in candidates if not is_donor_expert_name(name))
    if malformed:
        raise RuntimeError(f"malformed donor main expert names (first 5): {malformed[:5]}")
    selected_names = [name for name in candidates if donor_layer(name) in selected]
    expected = expected_names(layers, donor=True)
    actual = set(selected_names)
    if actual != expected:
        raise RuntimeError(
            f"donor selected expert set mismatch: actual={len(actual)} expected={len(expected)} "
            f"missing={sorted(expected - actual)[:5]} extra={sorted(actual - expected)[:5]}"
        )
    selected_names.sort(
        key=lambda name: (
            int(name.split(".")[2]),
            int(name.split(".")[5]),
            PROJECTION_ORDER[name.split(".")[6]],
            EXL3_SUFFIXES.index(name.split(".")[7]),
        )
    )
    log(
        f"DONOR_INDEX selected_tensors={len(selected_names)} layers={list(layers)} "
        f"files={sorted({weight_map[name] for name in selected_names})}"
    )
    return selected_names


def donor_headers_from_remote(
    donor: str,
    weight_map: dict[str, str],
    selected_names: Iterable[str],
    cache_path: Path | None,
    *,
    write_cache: bool = True,
    allow_network: bool = True,
) -> dict[str, tuple[int, dict]]:
    """Read one header per required donor shard, locally or with ranges."""
    base = donor_base_url(donor)
    files = sorted({weight_map[name] for name in selected_names})
    cached: dict = {}
    if cache_path is not None and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text()).get("headers", {})
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read donor header cache {cache_path}: {exc}") from exc

    result: dict[str, tuple[int, dict]] = {}
    for filename in files:
        item = cached.get(filename)
        if isinstance(item, list) and len(item) == 2:
            header_len, header = int(item[0]), item[1]
            validate_header(header)
            result[filename] = (header_len, header)
            log(f"DONOR_HEADER cache file={filename} tensors={len(header) - 1}")
            continue
        if not allow_network:
            raise RuntimeError(f"donor header cache has no entry for {filename}")
        url = f"{base}/{filename}"
        length_blob = local_or_remote_bytes(url, (0, 7))
        if len(length_blob) != 8:
            raise RuntimeError(f"donor header length read for {filename} was not 8 bytes")
        header_len = struct.unpack("<Q", length_blob)[0]
        header_blob = local_or_remote_bytes(url, (8, 8 + header_len - 1))
        parsed_len, header = parse_header(length_blob + header_blob)
        validate_header(header)
        if parsed_len != header_len:
            raise RuntimeError(f"donor header length changed for {filename}")
        result[filename] = (header_len, header)
        log(f"DONOR_HEADER remote_or_local file={filename} tensors={len(header) - 1}")

    if cache_path is not None and write_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"headers": {name: [hlen, header] for name, (hlen, header) in result.items()}}, indent=2)
        )
        log(f"DONOR_HEADER_CACHE wrote={cache_path}")
    return result


def expected_output_meta(
    donor_name: str,
    k2_abi: dict[str, dict[str, dict]],
) -> dict:
    match = DONOR_EXPERT_RE.fullmatch(donor_name)
    assert match is not None
    projection = PROJECTION_MAP[match.group("projection")]
    suffix = match.group("suffix")
    if suffix == "mcg":
        # Donor markers are scalar [], while the destination parameter is [1].
        return {"dtype": "I32", "shape": [1]}
    return copy.deepcopy(k2_abi[projection][suffix])


def validate_and_plan_donor(
    selected: list[str],
    weight_map: dict[str, str],
    donor_headers: dict[str, tuple[int, dict]] | None,
    k2_abi: dict[str, dict[str, dict]],
) -> list[dict]:
    plan: list[dict] = []
    seen_targets: set[str] = set()
    for donor_name in selected:
        target_name = map_donor_name(donor_name)
        if target_name in seen_targets:
            raise RuntimeError(f"donor name map collision at {target_name}")
        seen_targets.add(target_name)
        expected = expected_output_meta(donor_name, k2_abi)
        source_file = weight_map[donor_name]
        actual = copy.deepcopy(expected)
        donor_meta = None
        if donor_headers is not None:
            if source_file not in donor_headers:
                raise RuntimeError(f"missing donor header for {source_file}")
            _, header = donor_headers[source_file]
            if donor_name not in header:
                raise RuntimeError(f"donor header lacks indexed tensor {donor_name}")
            donor_meta = copy.deepcopy(header[donor_name])
            validate_header({donor_name: donor_meta})
            actual = _abi_meta(donor_meta)
            if donor_name.endswith(".mcg") and actual.get("shape") == []:
                actual["shape"] = [1]
            if actual != expected:
                raise RuntimeError(
                    f"donor shape/dtype mismatch for {donor_name}: got {actual}, expected {expected}"
                )
        plan.append(
            {
                "donor": donor_name,
                "target": target_name,
                "file": source_file,
                "meta": actual,
                "expected": expected,
                "donor_meta": donor_meta,
            }
        )
    expected_count = len(selected)
    if len(plan) != expected_count:
        raise RuntimeError(f"planned donor tensor count is {len(plan)}, expected {expected_count}")
    return plan


def tensor_range(header_len: int, meta: dict) -> tuple[int, int]:
    start, end = meta["data_offsets"]
    return 8 + header_len + start, 8 + header_len + end - 1


def build_new_header(plan: list[dict], donor: str, layers: tuple[int, ...]) -> tuple[list[tuple[str, dict]], int, dict]:
    entries = [(item["target"], item["meta"]) for item in plan]
    offset = 0
    header: dict = {
        "__metadata__": {
            "format": "pt",
            "source": donor,
            "overlay": "main-layer-k2",
            "layers": ",".join(str(layer) for layer in layers),
        }
    }
    for name, meta in entries:
        size = tensor_nbytes(meta)
        header[name] = {
            "dtype": meta["dtype"],
            "shape": list(meta["shape"]),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    return entries, offset, header


def header_without_offsets(header: dict) -> dict:
    return {
        name: {"dtype": meta["dtype"], "shape": list(meta["shape"])}
        for name, meta in header.items()
        if name != "__metadata__"
    }


def write_new_shard(
    out: Path,
    plan: list[dict],
    donor: str,
    layers: tuple[int, ...],
    donor_headers: dict[str, tuple[int, dict]],
) -> int:
    entries, payload_bytes, expected_header = build_new_header(plan, donor, layers)
    destination = out / NEW_SHARD
    partial = out / (NEW_SHARD + ".partial")
    header_len = None
    actual_header = None
    resume_at = 0

    if destination.exists():
        existing_len, existing_header = read_header(destination)
        if header_without_offsets(existing_header) != header_without_offsets(expected_header):
            raise RuntimeError(f"existing {destination} has a different tensor plan")
        if destination.stat().st_size != 8 + existing_len + payload_bytes:
            raise RuntimeError(f"existing {destination} is incomplete or has trailing bytes")
        log(f"NEW_SHARD existing_complete={destination}")
        return payload_bytes

    if partial.exists():
        existing_len, existing_header = read_header(partial, check_data=False)
        if header_without_offsets(existing_header) == header_without_offsets(expected_header):
            data_start = 8 + existing_len
            data_bytes = partial.stat().st_size - data_start
            boundaries = {0} | {
                meta["data_offsets"][1]
                for name, meta in existing_header.items()
                if name != "__metadata__"
            }
            if data_bytes in boundaries and 0 <= data_bytes <= payload_bytes:
                header_len, actual_header, resume_at = existing_len, existing_header, data_bytes
                log(f"NEW_SHARD resume={partial} completed_bytes={resume_at}")
            else:
                log(f"NEW_SHARD discard_invalid_partial={partial}")
                partial.unlink()
        else:
            log(f"NEW_SHARD discard_stale_partial={partial}")
            partial.unlink()

    if header_len is None:
        with partial.open("wb") as stream:
            header_len, actual_header = common.write_safetensors_header(
                stream,
                entries,
                {"format": "pt", "source": donor, "overlay": "main-layer-k2", "layers": ",".join(map(str, layers))},
            )
        resume_at = 0

    assert header_len is not None and actual_header is not None
    with partial.open("r+b") as stream:
        data_start = 8 + header_len
        stream.truncate(data_start + resume_at)
        stream.seek(data_start + resume_at)
        for index, item in enumerate(plan):
            name = item["target"]
            meta = actual_header[name]
            start, end = meta["data_offsets"]
            if end <= resume_at:
                log(f"FETCH skip_existing tensor={index + 1}/{len(plan)} name={name}")
                continue
            if start != resume_at:
                raise RuntimeError(f"partial new shard boundary mismatch before {name}")
            donor_header_len, donor_header = donor_headers[item["file"]]
            donor_meta = donor_header[item["donor"]]
            absolute = tensor_range(donor_header_len, donor_meta)
            log(f"FETCH tensor={index + 1}/{len(plan)} name={name} bytes={end - start}")
            data = local_or_remote_bytes(f"{donor_base_url(donor)}/{item['file']}", absolute)
            if len(data) != end - start:
                raise RuntimeError(f"{name}: fetched {len(data)} bytes, expected {end - start}")
            stream.write(data)
            resume_at = end
    if resume_at != payload_bytes:
        raise RuntimeError(f"new shard ended at {resume_at} bytes, expected {payload_bytes}")
    os.replace(partial, destination)
    log(f"NEW_SHARD wrote={destination} payload_bytes={payload_bytes}")
    return payload_bytes


def rewrite_source_shard(source_path: Path, destination_path: Path, dropped: set[str]) -> int:
    """Copy one source shard while streaming every retained tensor payload."""
    header_len, header = read_header(source_path)
    entries = [
        (name, meta)
        for name, meta in header.items()
        if name != "__metadata__" and name not in dropped
    ]
    copied = sum(tensor_nbytes(meta) for _, meta in entries)
    partial = destination_path.with_name(destination_path.name + ".partial")
    with source_path.open("rb") as source_stream, partial.open("wb") as destination_stream:
        common.write_safetensors_header(destination_stream, entries, header.get("__metadata__"))
        source_data_start = 8 + header_len
        for name, meta in entries:
            common.copy_payload(source_stream, destination_stream, source_data_start, meta)
    os.replace(partial, destination_path)
    log(
        f"REWRITE file={source_path.name} kept_tensors={len(entries)} copied_bytes={copied} "
        f"dropped_tensors={sum(name in dropped for name in header if name != '__metadata__')}"
    )
    return copied


def safe_output_path(src: Path, out: Path) -> None:
    src_real = src.resolve()
    out_real = out.resolve(strict=False)
    if out_real == src_real or src_real in out_real.parents:
        raise RuntimeError(f"refusing output inside source pack: {out}")


def link_source_files(src: Path, out: Path, rewrite_files: set[str]) -> int:
    excluded = rewrite_files | {CONFIG_NAME, INDEX_NAME, NEW_SHARD}
    linked = 0
    for entry in sorted(src.iterdir(), key=lambda path: path.name):
        if entry.name in excluded or not (entry.is_file() or entry.is_symlink()):
            continue
        destination = out / entry.name
        if os.path.lexists(destination):
            if destination.is_symlink() and destination.resolve() == entry.resolve():
                continue
            raise RuntimeError(f"output path already exists and is not the source link: {destination}")
        # Resolve source symlinks so the overlay remains usable independently
        # of a symlink chain in the source overlay.
        destination.symlink_to(entry.resolve())
        linked += 1
    log(f"LINKED source_files={linked}")
    return linked


def output_index(
    source: dict[str, tuple[Path, int, dict]],
    dropped: set[str],
    new_plan: list[dict],
) -> dict:
    weight_map: dict[str, str] = {}
    total_size = 0
    for name, (path, _, header) in sorted(source.items()):
        if name in dropped:
            continue
        weight_map[name] = path.name
        total_size += tensor_nbytes(header[name])
    for item in new_plan:
        if item["target"] in weight_map:
            raise RuntimeError(f"new tensor collides with retained source tensor: {item['target']}")
        weight_map[item["target"]] = NEW_SHARD
        total_size += tensor_nbytes(item["meta"])
    return {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}


def edited_config(src: Path, layers: tuple[int, ...]) -> dict:
    config_path = src / CONFIG_NAME
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {config_path}: {exc}") from exc
    q = config.get("quantization_config")
    if not isinstance(q, dict):
        raise RuntimeError("source config has no quantization_config object")
    if q.get("bits") != 2:
        raise RuntimeError(f"source base quantization bits must be 2, got {q.get('bits')!r}")
    layer_bits = q.get("layer_bits")
    if not isinstance(layer_bits, dict):
        raise RuntimeError("source quantization_config.layer_bits is not an object")
    layer_bits = dict(layer_bits)
    for layer in layers:
        key = str(layer)
        if layer_bits.get(key) != 3:
            raise RuntimeError(f"source config layer_bits[{key!r}] must be 3 before replacement, got {layer_bits.get(key)!r}")
        layer_bits.pop(key)
    q["layer_bits"] = layer_bits
    return config


def write_output_metadata(out: Path, index: dict, config: dict) -> None:
    (out / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    log(f"WROTE metadata files={INDEX_NAME},{CONFIG_NAME}")


def compare_payloads(
    source_path: Path,
    output_path: Path,
    source_meta: dict,
    output_meta: dict,
    source_header_len: int,
    output_header_len: int,
) -> None:
    source_start, source_end = source_meta["data_offsets"]
    output_start, output_end = output_meta["data_offsets"]
    if source_end - source_start != output_end - output_start:
        raise RuntimeError("retained tensor payload length changed")
    with source_path.open("rb") as source_stream, output_path.open("rb") as output_stream:
        source_stream.seek(8 + source_header_len + source_start)
        output_stream.seek(8 + output_header_len + output_start)
        remaining = source_end - source_start
        while remaining:
            size = min(CHUNK_SIZE, remaining)
            source_data = source_stream.read(size)
            output_data = output_stream.read(size)
            if len(source_data) != size or source_data != output_data:
                raise RuntimeError(f"retained tensor payload differs: {source_path.name}")
            remaining -= size


def verify_output(
    src: Path,
    out: Path,
    source: dict[str, tuple[Path, int, dict]],
    dropped: set[str],
    plan: list[dict],
    expected_index: dict,
    expected_config: dict,
    rewrite_files: set[str],
) -> None:
    index_path = out / INDEX_NAME
    config_path = out / CONFIG_NAME
    if not index_path.is_file() or not config_path.is_file():
        raise RuntimeError("overlay index/config is missing")
    actual_index = json.loads(index_path.read_text())
    if actual_index != expected_index:
        raise RuntimeError("overlay index differs from the planned index")

    output_headers: dict[str, tuple[Path, int, dict]] = {}
    for filename in sorted(set(actual_index["weight_map"].values())):
        path = out / filename
        if not path.is_file():
            raise RuntimeError(f"indexed shard is missing: {path}")
        header_len, header = read_header(path)
        for name, mapped_file in actual_index["weight_map"].items():
            if mapped_file == filename:
                if name not in header:
                    raise RuntimeError(f"indexed tensor is missing from {filename}: {name}")
                output_headers[name] = (path, header_len, header)
    if set(output_headers) != set(actual_index["weight_map"]):
        raise RuntimeError("not every indexed tensor was opened")

    for entry in src.iterdir():
        if not (entry.is_file() or entry.is_symlink()):
            continue
        if entry.name in rewrite_files or entry.name in (CONFIG_NAME, INDEX_NAME, NEW_SHARD):
            continue
        linked = out / entry.name
        if not linked.is_symlink() or linked.resolve() != entry.resolve():
            raise RuntimeError(f"source file is not linked unchanged: {entry.name}")
    for filename in rewrite_files:
        if (out / filename).is_symlink():
            raise RuntimeError(f"rewritten source shard is still a symlink: {filename}")

    for name, (source_path, source_header_len, source_header) in source.items():
        if name in dropped:
            # Replaced in place: the same name must now point at the new K2 shard.
            where = actual_index["weight_map"].get(name)
            if where is not None and where != NEW_SHARD:
                raise RuntimeError(f"dropped source tensor remains indexed in {where}: {name}")
            continue
        if source_path.name not in rewrite_files:
            continue
        output_path, output_header_len, output_header = output_headers[name]
        if output_header[name]["dtype"] != source_header[name]["dtype"] or output_header[name]["shape"] != source_header[name]["shape"]:
            raise RuntimeError(f"retained tensor metadata changed: {name}")
        compare_payloads(
            source_path,
            output_path,
            source_header[name],
            output_header[name],
            source_header_len,
            output_header_len,
        )

    for item in plan:
        path, header_len, header = output_headers[item["target"]]
        actual = header[item["target"]]
        if actual["dtype"] != item["meta"]["dtype"] or actual["shape"] != item["meta"]["shape"]:
            raise RuntimeError(f"new tensor metadata differs from donor plan: {item['target']}")
        if item["target"].endswith(".mcg"):
            with path.open("rb") as stream:
                start, end = actual["data_offsets"]
                stream.seek(8 + header_len + start)
                payload = stream.read(end - start)
            if len(payload) != 4 or struct.unpack("<i", payload)[0] != -877912083:
                raise RuntimeError(f"MCG marker mismatch for {item['target']}")

    actual_config = json.loads(config_path.read_text())
    if actual_config != expected_config:
        raise RuntimeError("overlay config differs from the planned config")
    log(f"VERIFY_OK indexed_tensors={len(output_headers)} retained_payloads_byte_identical=true")


def build(args: argparse.Namespace) -> int:
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser()
    donor = args.donor
    layers = parse_layers(args.layers)
    if not src.is_dir():
        raise RuntimeError(f"source directory does not exist: {src}")
    safe_output_path(src, out)

    source, source_map = source_headers(src)
    validate_source_index(src, source_map)
    k2_abi, exemplar_layer = find_source_k2_abi(source, layers)
    dropped, dropped_by_file, dropped_bytes = validate_source_selection(source, layers, k2_abi)
    rewrite_files = set(dropped_by_file)
    expected_config = edited_config(src, layers)
    log(f"CONFIG_PLAN remove_layer_bits={[str(layer) for layer in layers]} exemplar_layer={exemplar_layer}")

    donor_index_path = Path(args.donor_index).expanduser() if args.donor_index else None
    if donor_index_path is not None and donor_index_path.exists():
        donor_map = donor_index_from_file(donor_index_path)
        log(f"DONOR_INDEX read={donor_index_path}")
    elif args.dry_run:
        raise RuntimeError(f"--dry-run requires the local donor index; not found: {donor_index_path}")
    else:
        index_blob = local_or_remote_bytes(f"{donor_base_url(donor)}/{INDEX_NAME}")
        try:
            donor_map = json.loads(index_blob)["weight_map"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"remote donor index is invalid: {exc}") from exc
        log(f"DONOR_INDEX remote tensors={len(donor_map)}")
    selected = validate_donor_selection(donor_map, layers)

    cache_path = Path(args.donor_headers).expanduser() if args.donor_headers else None
    donor_headers: dict[str, tuple[int, dict]] | None = None
    if args.dry_run:
        if cache_path is not None and cache_path.exists():
            donor_headers = donor_headers_from_remote(
                donor,
                donor_map,
                selected,
                cache_path,
                write_cache=False,
                allow_network=False,
            )
        else:
            log("DONOR_HEADERS offline=not-read; dry-run uses local K2 ABI")
    else:
        donor_headers = donor_headers_from_remote(donor, donor_map, selected, cache_path)
    plan = validate_and_plan_donor(selected, donor_map, donor_headers, k2_abi)

    fetched_bytes = sum(tensor_nbytes(item["meta"]) for item in plan)
    rewritten_bytes = sum(
        tensor_nbytes(header[name])
        for name, (path, _, header) in source.items()
        if name not in dropped and path.name in rewrite_files
    )
    expected_index = output_index(source, dropped, plan)
    log(
        f"PLAN layers={list(layers)} donor_tensors={len(plan)} bytes_to_fetch={fetched_bytes} "
        f"({fetched_bytes / (1 << 30):.3f} GiB)"
    )
    log(
        f"PLAN rewrite_files={sorted(rewrite_files)} bytes_to_rewrite={rewritten_bytes} "
        f"bytes_removed={dropped_bytes} dropped_tensors={len(dropped)}"
    )
    log(
        f"PLAN output symlinks=source_files_except_rewritten_config_index "
        f"new_shard={NEW_SHARD} index={INDEX_NAME} config=remove_selected_layer_bits"
    )
    if args.dry_run:
        log("DRY_RUN_OK network=false writes=false")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    link_source_files(src, out, rewrite_files)
    if donor_headers is None:
        raise RuntimeError("donor headers were not loaded for a real build")
    write_new_shard(out, plan, donor, layers, donor_headers)
    for filename in sorted(rewrite_files):
        rewrite_source_shard(src / filename, out / filename, dropped_by_file[filename])
    write_output_metadata(out, expected_index, expected_config)
    log(f"BUILD_OK out={out}")
    if args.verify:
        verify_output(src, out, source, dropped, plan, expected_index, expected_config, rewrite_files)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source pack directory")
    parser.add_argument("--donor", required=True, help="Hugging Face repo id or base URL")
    parser.add_argument("--out", required=True, help="new overlay directory")
    parser.add_argument("--layers", required=True, help="comma-separated main layer numbers to replace")
    parser.add_argument(
        "--donor-index",
        default=DEFAULT_DONOR_INDEX,
        help=f"offline donor weight_map JSON (default: {DEFAULT_DONOR_INDEX})",
    )
    parser.add_argument(
        "--donor-headers",
        default=DEFAULT_HEADER_CACHE,
        help=f"donor header cache (default: {DEFAULT_HEADER_CACHE})",
    )
    parser.add_argument("--dry-run", action="store_true", help="plan only; no network and no writes")
    parser.add_argument("--verify", action="store_true", help="verify every indexed tensor after building")
    args = parser.parse_args()
    if args.dry_run and args.verify:
        parser.error("--dry-run and --verify cannot be combined")
    try:
        return build(args)
    except KeyboardInterrupt:
        log("ABORT interrupted")
        return 130
    except Exception as exc:
        log(f"ABORT {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
