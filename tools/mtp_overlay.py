#!/usr/bin/env python3
"""Build an EXL3-routed-expert overlay for the three DSpark MTP layers.

The source pack is never changed.  The output directory links every source file
except ``config.json`` and source shards 46--48, rewrites those three shards with
only their non-draft tensors, and adds ``model-mtp-exl3.safetensors`` containing
the donor's EXL3 MTP experts.  Donor tensor payloads are fetched with HTTP Range
requests one tensor at a time; whole donor shards are never downloaded.

Offline planning is intentional.  ``--dry-run`` reads the source headers and the
local donor weight map but never opens a network connection or writes an output.
The default local donor index is the one named in orders/T2-mtp-overlay.md.
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


SOURCE_REWRITE_FILES = tuple(
    f"model-{i:05d}-of-00048.safetensors" for i in (46, 47, 48)
)
NEW_SHARD = "model-mtp-exl3.safetensors"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
DEFAULT_DONOR_INDEX = "/run/media/tonoken3/DATA1/.tmp/d2-index.json"
DEFAULT_HEADER_CACHE = "/run/media/tonoken3/DATA1/.tmp/mtp-overlay-donor-headers.json"
CHUNK_SIZE = 8 << 20
RETRY_COUNT = 5

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

DONOR_EXPERT_RE = re.compile(
    r"^mtp\.(?P<layer>[0-2])\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<suffix>trellis|suh|svh|mcg)$"
)
SOURCE_EXPERT_RE = re.compile(
    r"^mtp\.(?P<layer>[0-2])\.ffn\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>w[123])\."
    r"(?P<suffix>weight|scale)$"
)
SOURCE_SHARD_RE = re.compile(r"^model-(?P<number>[0-9]+)-of-(?P<total>[0-9]+)\.safetensors$")

PROJECTION_MAP = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
PROJECTION_ORDER = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg")
EXL3_DTYPE = {"trellis": "I16", "suh": "F16", "svh": "F16", "mcg": "I32"}


def log(message: str) -> None:
    print(message, flush=True)


def tensor_nbytes(meta: dict) -> int:
    """Return the payload size and reject malformed safetensors metadata."""
    dtype = meta.get("dtype")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    shape = meta.get("shape")
    if not isinstance(shape, list) or any(not isinstance(x, int) or x < 0 for x in shape):
        raise ValueError(f"invalid safetensors shape {shape!r}")
    size = DTYPE_BYTES[dtype]
    for dim in shape:
        size *= dim
    return size


def parse_header(blob: bytes) -> tuple[int, dict]:
    """Parse a safetensors header from a blob containing at least its header."""
    if len(blob) < 8:
        raise ValueError("safetensors header is shorter than the 8-byte length")
    header_len = struct.unpack("<Q", blob[:8])[0]
    end = 8 + header_len
    if end > len(blob):
        raise ValueError(f"truncated safetensors header: need {end}, have {len(blob)}")
    try:
        header = json.loads(blob[8:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header JSON must be an object")
    return header_len, header


def validate_header(header: dict, data_bytes: int | None = None) -> None:
    """Validate the structural parts needed by the range copier."""
    for name, meta in header.items():
        if name == "__metadata__":
            if not isinstance(meta, dict):
                raise ValueError("safetensors __metadata__ must be an object")
            continue
        if not isinstance(meta, dict):
            raise ValueError(f"tensor {name!r} metadata is not an object")
        offsets = meta.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(x, int) for x in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"invalid data_offsets for {name!r}: {offsets!r}")
        expected = tensor_nbytes(meta)
        actual = offsets[1] - offsets[0]
        if expected != actual:
            raise ValueError(
                f"payload size mismatch for {name!r}: metadata={expected}, offsets={actual}"
            )
        if data_bytes is not None and offsets[1] > data_bytes:
            raise ValueError(
                f"data_offsets for {name!r} exceed data region: {offsets[1]} > {data_bytes}"
            )


def read_header(path: Path, *, check_data: bool = True) -> tuple[int, dict]:
    """Read and validate only a local file's 8-byte length and JSON header."""
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: missing 8-byte safetensors header length")
        header_len = struct.unpack("<Q", prefix)[0]
        header_blob = prefix + stream.read(header_len)
    if len(header_blob) != 8 + header_len:
        raise ValueError(f"{path}: truncated safetensors JSON header")
    parsed_len, header = parse_header(header_blob)
    if parsed_len != header_len:
        raise AssertionError("header length changed during parsing")
    validate_header(header, file_size - 8 - header_len if check_data else None)
    return header_len, header


def map_donor_name(name: str) -> str:
    """Map one donor EXL3 expert name to the target pack's expert ABI."""
    match = DONOR_EXPERT_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"not a donor MTP EXL3 expert tensor: {name}")
    projection = match.group("projection")
    return (
        f"mtp.{match.group('layer')}.ffn.experts.{int(match.group('expert'))}."
        f"{PROJECTION_MAP[projection]}.{match.group('suffix')}"
    )


# Short alias used by the offline tests and useful to callers importing this tool.
map_name = map_donor_name


def is_donor_expert_name(name: str) -> bool:
    return DONOR_EXPERT_RE.fullmatch(name) is not None


def tensor_range(header_len: int, meta: dict) -> tuple[int, int]:
    """Return the inclusive absolute byte range for one tensor."""
    start, end = meta["data_offsets"]
    return 8 + header_len + start, 8 + header_len + end - 1


def source_headers(src: Path) -> tuple[dict[str, tuple[Path, int, dict]], dict[str, str]]:
    """Scan all numbered source shards; the source intentionally has no index."""
    candidates = []
    for path in src.iterdir():
        match = SOURCE_SHARD_RE.fullmatch(path.name)
        if match and path.is_file():
            candidates.append((int(match.group("number")), path))
    if not candidates:
        raise RuntimeError(f"no numbered safetensors shards found under {src}")

    headers: dict[str, tuple[Path, int, dict]] = {}
    weight_map: dict[str, str] = {}
    for _, path in sorted(candidates):
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


def http_bytes(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    """Fetch one complete response, retrying transient/range failures."""
    for attempt in range(RETRY_COUNT):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mtp_overlay/1"})
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


def donor_headers_from_remote(
    donor: str,
    weight_map: dict[str, str],
    selected_names: Iterable[str],
    cache_path: Path | None,
) -> dict[str, tuple[int, dict]]:
    """Read one remote safetensors header per selected donor shard."""
    base = donor.rstrip("/") if donor.startswith("http") else f"https://huggingface.co/{donor}/resolve/main"
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
        url = f"{base}/{filename}"
        length_blob = http_bytes(url, (0, 7))
        if len(length_blob) != 8:
            raise RuntimeError(f"donor header length read for {filename} was not 8 bytes")
        header_len = struct.unpack("<Q", length_blob)[0]
        header_blob = http_bytes(url, (8, 8 + header_len - 1))
        parsed_len, header = parse_header(length_blob + header_blob)
        validate_header(header)
        if parsed_len != header_len:
            raise RuntimeError(f"donor header length changed for {filename}")
        result[filename] = (header_len, header)
        log(f"DONOR_HEADER remote file={filename} tensors={len(header) - 1}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"headers": {name: [hlen, header] for name, (hlen, header) in result.items()}},
                indent=2,
            )
        )
        log(f"DONOR_HEADER_CACHE wrote={cache_path}")
    return result


def expected_donor_names() -> set[str]:
    return {
        f"mtp.{layer}.mlp.experts.{expert}.{projection}.{suffix}"
        for layer in range(3)
        for expert in range(256)
        for projection in PROJECTION_MAP
        for suffix in EXL3_SUFFIXES
    }


def source_draft_shapes(
    source: dict[str, tuple[Path, int, dict]],
) -> dict[str, tuple[int, int]]:
    """Validate the measured source FP8 draft weight shapes and return out/in dims."""
    matches = [
        (name, meta[2][name])
        for name, meta in source.items()
        if SOURCE_EXPERT_RE.fullmatch(name)
    ]
    expected = {
        f"mtp.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
        for layer in range(3)
        for expert in range(256)
        for projection in ("w1", "w2", "w3")
        for suffix in ("weight", "scale")
    }
    actual = {name for name, _ in matches}
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise RuntimeError(f"source draft expert set mismatch missing={missing} extra={extra}")

    shapes: dict[str, tuple[int, int]] = {}
    for projection in ("w1", "w2", "w3"):
        name = f"mtp.0.ffn.experts.0.{projection}.weight"
        meta = source[name][2][name]
        if meta.get("dtype") != "I8" or len(meta.get("shape", [])) != 2:
            raise RuntimeError(f"unexpected source draft metadata for {name}: {meta}")
        out_dim, in_dim = meta["shape"]
        if out_dim % 16 or in_dim % 16:
            raise RuntimeError(f"source draft dimensions are not tile-aligned for {name}: {meta['shape']}")
        shapes[projection] = (out_dim, in_dim)
        scale_name = f"mtp.0.ffn.experts.0.{projection}.scale"
        scale = source[scale_name][2][scale_name]
        if scale.get("dtype") != "F8_E8M0":
            raise RuntimeError(f"unexpected source draft scale dtype for {scale_name}: {scale}")

    for name, (_, _, header) in source.items():
        match = SOURCE_EXPERT_RE.fullmatch(name)
        if match is None:
            continue
        meta = header[name]
        projection = match.group("projection")
        suffix = match.group("suffix")
        if suffix == "weight":
            if meta["dtype"] != "I8" or tuple(meta["shape"]) != shapes[projection]:
                raise RuntimeError(f"inconsistent source draft weight {name}: {meta}")
        elif meta["dtype"] != "F8_E8M0":
            raise RuntimeError(f"inconsistent source draft scale {name}: {meta}")

    log(
        "SOURCE_DRAFT_SHAPES "
        + " ".join(
            f"{projection}.weight=I8{list(shapes[projection])}"
            for projection in ("w1", "w2", "w3")
        )
        + " scale=F8_E8M0"
    )
    return shapes


def main_k2_shapes(
    source: dict[str, tuple[Path, int, dict]],
) -> dict[str, dict[str, dict]]:
    """Read the concrete K2 EXL3 ABI from the requested main-layer exemplar."""
    result: dict[str, dict[str, dict]] = {}
    for projection in ("w1", "w2", "w3"):
        result[projection] = {}
        for suffix in EXL3_SUFFIXES:
            name = f"layers.5.ffn.experts.0.{projection}.{suffix}"
            if name not in source:
                raise RuntimeError(f"required main-layer K2 exemplar is missing: {name}")
            result[projection][suffix] = copy.deepcopy(source[name][2][name])
    for projection, parts in result.items():
        trellis = parts["trellis"]
        if trellis.get("dtype") != "I16" or trellis.get("shape", [])[-1:] != [32]:
            raise RuntimeError(f"main-layer exemplar is not K2 EXL3 for {projection}: {trellis}")
        for suffix in EXL3_SUFFIXES:
            if parts[suffix].get("dtype") != EXL3_DTYPE[suffix]:
                raise RuntimeError(f"main-layer exemplar dtype mismatch for {projection}.{suffix}")
    log(
        "MAIN_K2_ABI "
        + " ".join(
            f"{projection}="
            + ",".join(
                f"{suffix}:{result[projection][suffix]['dtype']}{result[projection][suffix]['shape']}"
                for suffix in EXL3_SUFFIXES
            )
            for projection in ("w1", "w2", "w3")
        )
    )
    return result


def expected_mtp_meta(
    projection: str,
    suffix: str,
    draft_shapes: dict[str, tuple[int, int]],
    main_shapes: dict[str, dict[str, dict]],
) -> dict:
    """Build the offline expected MTP ABI from local draft dimensions and K2."""
    out_dim, in_dim = draft_shapes[projection]
    if suffix == "trellis":
        # Exl3MoEMethod stores gate/up as [in_tiles, out_tiles, K*16] and
        # down as [out_tiles, in_tiles, K*16].  MTP source weights are the
        # local draft dimensions, not the 43-layer main dimensions.
        if projection in ("w1", "w3"):
            shape = [in_dim // 16, out_dim // 16, main_shapes[projection][suffix]["shape"][-1]]
        else:
            shape = [out_dim // 16, in_dim // 16, main_shapes[projection][suffix]["shape"][-1]]
    elif suffix == "suh":
        shape = [in_dim]
    elif suffix == "svh":
        shape = [out_dim]
    else:
        shape = [1]
    return {"dtype": EXL3_DTYPE[suffix], "shape": shape}


def validate_donor_selection(
    weight_map: dict[str, str],
) -> list[str]:
    names = [name for name in weight_map if name.startswith("mtp.") and ".experts." in name]
    malformed = sorted(name for name in names if not is_donor_expert_name(name))
    if malformed:
        raise RuntimeError(f"malformed donor MTP expert names (first 5): {malformed[:5]}")
    expected = expected_donor_names()
    actual = set(names)
    if actual != expected:
        raise RuntimeError(
            f"donor MTP expert set mismatch: selected={len(actual)} "
            f"missing={sorted(expected - actual)[:5]} extra={sorted(actual - expected)[:5]}"
        )
    selected = sorted(
        names,
        key=lambda name: (
            int(name.split(".")[1]),
            int(name.split(".")[4]),
            PROJECTION_ORDER[name.split(".")[5]],
            EXL3_SUFFIXES.index(name.split(".")[6]),
        ),
    )
    log(
        f"DONOR_INDEX experts={len(selected)} non_expert_mtp={sum(name.startswith('mtp.') for name in weight_map) - len(selected)} "
        f"files={sorted({weight_map[name] for name in selected})}"
    )
    return selected


def validate_and_plan_donor(
    selected: list[str],
    weight_map: dict[str, str],
    donor_headers: dict[str, tuple[int, dict]] | None,
    draft_shapes: dict[str, tuple[int, int]],
    main_shapes: dict[str, dict[str, dict]],
) -> list[dict]:
    plan = []
    seen_targets: set[str] = set()
    for donor_name in selected:
        target_name = map_donor_name(donor_name)
        if target_name in seen_targets:
            raise RuntimeError(f"donor name map collision at {target_name}")
        seen_targets.add(target_name)
        projection = PROJECTION_MAP[donor_name.split(".")[5]]
        suffix = donor_name.split(".")[6]
        expected = expected_mtp_meta(projection, suffix, draft_shapes, main_shapes)
        source_file = weight_map[donor_name]
        actual = expected
        if donor_headers is not None:
            if source_file not in donor_headers:
                raise RuntimeError(f"missing donor header for {source_file}")
            _, header = donor_headers[source_file]
            if donor_name not in header:
                raise RuntimeError(f"donor header lacks indexed tensor {donor_name}")
            actual = copy.deepcopy(header[donor_name])
            # Dtype and K must agree with the concrete main K2 EXL3 ABI.  The
            # draft's dimensions are intentionally used for MTP shape checking.
            if actual.get("dtype") != expected["dtype"] or actual.get("shape") != expected["shape"]:
                raise RuntimeError(
                    f"donor shape/dtype mismatch for {donor_name}: got "
                    f"{actual.get('dtype')}{actual.get('shape')}, expected "
                    f"{expected['dtype']}{expected['shape']} from local MTP dimensions"
                )
            validate_header({donor_name: actual})
        plan.append(
            {
                "donor": donor_name,
                "target": target_name,
                "file": source_file,
                "meta": actual,
                "expected": expected,
            }
        )
    if len(plan) != 9216:
        raise RuntimeError(f"planned donor tensor count is {len(plan)}, expected 9216")
    return plan


def source_drop_plan(
    source: dict[str, tuple[Path, int, dict]],
) -> tuple[set[str], dict[str, list[tuple[str, dict]]], int, int]:
    dropped: set[str] = set()
    by_file: dict[str, list[tuple[str, dict]]] = {name: [] for name in SOURCE_REWRITE_FILES}
    for name, (path, _, header) in source.items():
        if name.startswith("mtp.") and ".ffn.experts." in name and SOURCE_EXPERT_RE.fullmatch(name) is None:
            raise RuntimeError(f"malformed source MTP expert tensor name: {name}")
        match = SOURCE_EXPERT_RE.fullmatch(name)
        if match is None:
            continue
        if path.name not in by_file:
            raise RuntimeError(f"draft expert tensor is outside source shards 46-48: {name} in {path.name}")
        if int(match.group("expert")) >= 256:
            raise RuntimeError(f"source draft expert id out of range: {name}")
        dropped.add(name)
        by_file[path.name].append((name, header[name]))
    expected_per_file = 256 * 3 * 2
    for filename, items in by_file.items():
        if len(items) != expected_per_file:
            raise RuntimeError(f"{filename}: found {len(items)} draft tensors, expected {expected_per_file}")
    drop_bytes = sum(tensor_nbytes(meta) for items in by_file.values() for _, meta in items)
    log(f"SOURCE_DROP tensors={len(dropped)} bytes={drop_bytes} files={list(SOURCE_REWRITE_FILES)}")
    return dropped, by_file, len(dropped), drop_bytes


def write_safetensors_header(
    stream: BinaryIO,
    entries: list[tuple[str, dict]],
    metadata: dict | None = None,
) -> tuple[int, dict]:
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    for name, meta in entries:
        size = tensor_nbytes(meta)
        header[name] = {
            "dtype": meta["dtype"],
            "shape": list(meta["shape"]),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    blob = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    stream.write(struct.pack("<Q", len(blob)))
    stream.write(blob)
    return len(blob), header


def copy_payload(
    source: BinaryIO,
    destination: BinaryIO,
    source_data_start: int,
    meta: dict,
) -> None:
    start, end = meta["data_offsets"]
    source.seek(source_data_start + start)
    remaining = end - start
    while remaining:
        block = source.read(min(CHUNK_SIZE, remaining))
        if not block:
            raise RuntimeError("short source tensor payload while rewriting shard")
        destination.write(block)
        remaining -= len(block)


def rewrite_source_shard(
    source_path: Path,
    destination_path: Path,
    dropped: set[str],
) -> int:
    header_len, header = read_header(source_path)
    entries = [(name, meta) for name, meta in header.items() if name != "__metadata__" and name not in dropped]
    copied = sum(tensor_nbytes(meta) for _, meta in entries)
    partial = destination_path.with_name(destination_path.name + ".partial")
    with source_path.open("rb") as source_stream, partial.open("wb") as destination_stream:
        _, new_header = write_safetensors_header(
            destination_stream,
            entries,
            header.get("__metadata__"),
        )
        del new_header
        source_data_start = 8 + header_len
        for name, meta in entries:
            copy_payload(source_stream, destination_stream, source_data_start, meta)
    os.replace(partial, destination_path)
    log(
        f"REWRITE file={source_path.name} kept_tensors={len(entries)} "
        f"copied_bytes={copied} dropped_tensors={sum(name in dropped for name, _ in header.items())}"
    )
    return copied


def safe_output_path(src: Path, out: Path) -> None:
    src_real = src.resolve()
    out_real = out.resolve(strict=False)
    if out_real == src_real or src_real in out_real.parents:
        raise RuntimeError(f"refusing output inside source pack: {out}")


def link_source_files(src: Path, out: Path) -> int:
    excluded = set(SOURCE_REWRITE_FILES) | {CONFIG_NAME, INDEX_NAME}
    linked = 0
    for entry in sorted(src.iterdir(), key=lambda p: p.name):
        if entry.name in excluded or not (entry.is_file() or entry.is_symlink()):
            continue
        destination = out / entry.name
        if os.path.lexists(destination):
            if destination.is_symlink() and destination.resolve() == entry.resolve():
                continue
            raise RuntimeError(f"output path already exists and is not the source link: {destination}")
        destination.symlink_to(entry.resolve())
        linked += 1
    log(f"LINKED source_files={linked}")
    return linked


def build_new_header(plan: list[dict], donor: str) -> tuple[list[tuple[str, dict]], int, dict]:
    entries = [(item["target"], item["meta"]) for item in plan]
    offset = 0
    header: dict = {"__metadata__": {"format": "pt", "source": donor, "overlay": "mtp-exl3"}}
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
    donor_headers: dict[str, tuple[int, dict]],
) -> int:
    entries, payload_bytes, expected_header = build_new_header(plan, donor)
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
        # A process can stop in the middle of the current tensor.  Read and
        # validate the JSON header without requiring the partial data region to
        # reach its final offset, then resume only at a completed boundary.
        existing_len, existing_header = read_header(partial, check_data=False)
        if header_without_offsets(existing_header) == header_without_offsets(expected_header):
            data_start = 8 + existing_len
            data_bytes = partial.stat().st_size - data_start
            boundaries = {0} | {meta["data_offsets"][1] for name, meta in existing_header.items() if name != "__metadata__"}
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
            header_len, actual_header = write_safetensors_header(
                stream, entries, {"format": "pt", "source": donor, "overlay": "mtp-exl3"}
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
            remote_header_len, remote_header = donor_headers[item["file"]]
            remote_meta = remote_header[item["donor"]]
            absolute = tensor_range(remote_header_len, remote_meta)
            base = donor.rstrip("/") if donor.startswith("http") else f"https://huggingface.co/{donor}/resolve/main"
            log(f"FETCH tensor={index + 1}/{len(plan)} name={name} bytes={end - start}")
            data = http_bytes(f"{base}/{item['file']}", absolute)
            if len(data) != end - start:
                raise RuntimeError(f"{name}: fetched {len(data)} bytes, expected {end - start}")
            stream.write(data)
            resume_at = end
    if resume_at != payload_bytes:
        raise RuntimeError(f"new shard ended at {resume_at} bytes, expected {payload_bytes}")
    os.replace(partial, destination)
    log(f"NEW_SHARD wrote={destination} payload_bytes={payload_bytes}")
    return payload_bytes


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
        output_file = path.name if path.name not in SOURCE_REWRITE_FILES else path.name
        weight_map[name] = output_file
        total_size += tensor_nbytes(header[name])
    for item in new_plan:
        if item["target"] in weight_map:
            raise RuntimeError(f"new tensor collides with retained source tensor: {item['target']}")
        weight_map[item["target"]] = NEW_SHARD
        total_size += tensor_nbytes(item["meta"])
    return {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}


def edited_config(src: Path) -> dict:
    config_path = src / CONFIG_NAME
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {config_path}: {exc}") from exc
    q = config.get("quantization_config")
    if not isinstance(q, dict):
        raise RuntimeError("source config has no quantization_config object")
    q.pop("mtp_experts", None)
    q.pop("mtp_experts_start_layer", None)
    layer_bits = q.get("layer_bits")
    if not isinstance(layer_bits, dict):
        raise RuntimeError("source quantization_config.layer_bits is not an object")
    layer_bits = dict(layer_bits)
    layer_bits.update({"43": 2, "44": 2, "45": 2})
    q["layer_bits"] = layer_bits
    return config


def write_output_metadata(out: Path, index: dict, config: dict) -> None:
    (out / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    log(f"WROTE metadata files={INDEX_NAME},{CONFIG_NAME}")


def compare_payloads(source_path: Path, output_path: Path, source_meta: dict, output_meta: dict, source_header_len: int, output_header_len: int) -> None:
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
            a = source_stream.read(size)
            b = output_stream.read(size)
            if a != b:
                raise RuntimeError(f"retained tensor payload differs: {source_path.name}")
            remaining -= len(a)


def verify_output(
    src: Path,
    out: Path,
    source: dict[str, tuple[Path, int, dict]],
    dropped: set[str],
    plan: list[dict],
    expected_index: dict,
    source_config: dict,
) -> None:
    index_path = out / INDEX_NAME
    config_path = out / CONFIG_NAME
    if not index_path.is_file() or not config_path.is_file():
        raise RuntimeError("overlay index/config is missing")
    actual_index = json.loads(index_path.read_text())
    if actual_index.get("weight_map") != expected_index.get("weight_map"):
        raise RuntimeError("overlay index weight_map differs from the planned map")

    # Every indexed tensor is opened through its file header and its declared
    # shape/dtype/byte length is checked against the index.
    output_headers: dict[str, tuple[Path, int, dict]] = {}
    for filename in sorted(set(actual_index["weight_map"].values())):
        path = out / filename
        if not path.is_file():
            raise RuntimeError(f"indexed shard is missing: {path}")
        header_len, header = read_header(path)
        for name, mapped_file in actual_index["weight_map"].items():
            if mapped_file != filename:
                continue
            if name not in header:
                raise RuntimeError(f"indexed tensor is missing from {filename}: {name}")
            meta = header[name]
            if tensor_nbytes(meta) != meta["data_offsets"][1] - meta["data_offsets"][0]:
                raise RuntimeError(f"invalid indexed tensor payload: {name}")
            output_headers[name] = (path, header_len, header)
    if set(output_headers) != set(actual_index["weight_map"]):
        raise RuntimeError("not every indexed tensor was opened")

    # Confirm source links and byte identity of every retained tensor in the
    # three rewritten shards.  Other source shards are links and need no copy.
    for entry in src.iterdir():
        if not (entry.is_file() or entry.is_symlink()) or entry.name in SOURCE_REWRITE_FILES or entry.name in (CONFIG_NAME, INDEX_NAME):
            continue
        linked = out / entry.name
        if not linked.is_symlink() or linked.resolve() != entry.resolve():
            raise RuntimeError(f"source file is not linked unchanged: {entry.name}")
    for name, (source_path, source_header_len, source_header) in source.items():
        if name in dropped:
            if name in actual_index["weight_map"]:
                raise RuntimeError(f"dropped source tensor remains indexed: {name}")
            continue
        if source_path.name not in SOURCE_REWRITE_FILES:
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

    # New names, shapes, dtypes and MCG markers must match the donor plan.
    for item in plan:
        path, header_len, header = output_headers[item["target"]]
        del path, header_len
        actual = header[item["target"]]
        if actual["dtype"] != item["meta"]["dtype"] or actual["shape"] != item["meta"]["shape"]:
            raise RuntimeError(f"new tensor metadata differs from donor plan: {item['target']}")
        if item["target"].endswith(".mcg"):
            with (out / NEW_SHARD).open("rb") as stream:
                start, end = actual["data_offsets"]
                stream.seek(8 + header_len + start)
                value = struct.unpack("<i", stream.read(end - start))[0]
            if value != -877912083:
                raise RuntimeError(f"MCG marker mismatch for {item['target']}: {value}")

    config = json.loads(config_path.read_text())
    q = config.get("quantization_config", {})
    if "mtp_experts" in q or "mtp_experts_start_layer" in q:
        raise RuntimeError("MTP source-format config keys were not removed")
    if q.get("non_routed_quantization") != source_config["quantization_config"].get("non_routed_quantization"):
        raise RuntimeError("non_routed_quantization changed")
    if q.get("non_routed_dtype_policy") != source_config["quantization_config"].get("non_routed_dtype_policy"):
        raise RuntimeError("non_routed_dtype_policy changed")
    for key in ("43", "44", "45"):
        if q.get("layer_bits", {}).get(key) != 2:
            raise RuntimeError(f"layer_bits[{key}] is not explicitly 2")
    log(f"VERIFY_OK indexed_tensors={len(output_headers)} retained_payloads_byte_identical=true")


def build(args: argparse.Namespace) -> int:
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser()
    donor = args.donor
    if not src.is_dir():
        raise RuntimeError(f"source directory does not exist: {src}")
    safe_output_path(src, out)

    source, source_map = source_headers(src)
    dropped, by_file, dropped_count, dropped_bytes = source_drop_plan(source)
    del by_file
    draft_shapes = source_draft_shapes(source)
    main_shapes = main_k2_shapes(source)

    donor_index_path = Path(args.donor_index).expanduser() if args.donor_index else None
    if donor_index_path is not None and donor_index_path.exists():
        donor_map = donor_index_from_file(donor_index_path)
        log(f"DONOR_INDEX read={donor_index_path}")
    elif args.dry_run:
        raise RuntimeError(
            f"--dry-run requires the local donor index; not found: {donor_index_path}"
        )
    else:
        base = donor.rstrip("/") if donor.startswith("http") else f"https://huggingface.co/{donor}/resolve/main"
        donor_index_blob = http_bytes(f"{base}/model.safetensors.index.json")
        try:
            donor_map = json.loads(donor_index_blob)["weight_map"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"remote donor index is invalid: {exc}") from exc
        log(f"DONOR_INDEX remote tensors={len(donor_map)}")
    selected = validate_donor_selection(donor_map)

    cache_path = Path(args.donor_headers).expanduser() if args.donor_headers else None
    donor_headers: dict[str, tuple[int, dict]] | None = None
    if args.dry_run:
        if cache_path is not None and cache_path.exists():
            donor_headers = donor_headers_from_remote(donor, donor_map, selected, cache_path)
        else:
            log("DONOR_HEADERS offline=not-read; dry-run uses local MTP dimensions")
    else:
        donor_headers = donor_headers_from_remote(donor, donor_map, selected, cache_path)
    plan = validate_and_plan_donor(selected, donor_map, donor_headers, draft_shapes, main_shapes)

    fetched_bytes = sum(tensor_nbytes(item["meta"]) for item in plan)
    rewritten_bytes = sum(
        tensor_nbytes(meta)
        for name, (_, _, header) in source.items()
        if name not in dropped and Path(source_map[name]).name in SOURCE_REWRITE_FILES
        for meta in [header[name]]
    )
    source_config = edited_config(src)
    expected_index = output_index(source, dropped, plan)
    log(
        f"PLAN donor_tensors={len(plan)} bytes_to_fetch={fetched_bytes} "
        f"({fetched_bytes / (1 << 30):.3f} GiB)"
    )
    log(
        f"PLAN rewrite_files=3 bytes_to_rewrite={rewritten_bytes} "
        f"bytes_removed={dropped_bytes} dropped_tensors={dropped_count}"
    )
    log(
        f"PLAN output symlinks=source_files_except_46_48_and_config "
        f"new_shard={NEW_SHARD} index={INDEX_NAME} config_edits=remove_mtp_source_keys+layer_bits_43_44_45=2"
    )
    if args.dry_run:
        log("DRY_RUN_OK network=false writes=false")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    link_source_files(src, out)
    if donor_headers is None:
        raise RuntimeError("donor headers were not loaded for a real build")
    write_new_shard(out, plan, donor, donor_headers)
    for filename in SOURCE_REWRITE_FILES:
        rewrite_source_shard(src / filename, out / filename, dropped)
    write_output_metadata(out, expected_index, source_config)
    log(f"BUILD_OK out={out}")
    if args.verify:
        verify_output(src, out, source, dropped, plan, expected_index, json.loads((src / CONFIG_NAME).read_text()))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source pack directory")
    parser.add_argument("--donor", required=True, help="Hugging Face repo id or base URL")
    parser.add_argument("--out", required=True, help="new overlay directory")
    parser.add_argument(
        "--donor-index",
        default=DEFAULT_DONOR_INDEX,
        help=f"offline donor weight_map JSON (default: {DEFAULT_DONOR_INDEX})",
    )
    parser.add_argument(
        "--donor-headers",
        default=DEFAULT_HEADER_CACHE,
        help=f"remote donor header cache (default: {DEFAULT_HEADER_CACHE})",
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
