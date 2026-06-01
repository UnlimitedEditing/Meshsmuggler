#!/usr/bin/env python3
"""
unsmuggle_mesh.py -- reconstruct a binary file from PNG(s) produced by the
ComfyUI "Smuggle Mesh As Image" node.

Usage:
    python unsmuggle_mesh.py image1.png [image2.png ...] [-o out.glb] [--info]

Pass every chunk PNG for ONE smuggled file in a single invocation. The output
filename defaults to the original name embedded in the image.

Requires: Pillow, numpy   ->   pip install pillow numpy
"""

import os
import sys
import gzip
import zlib
import struct
import argparse

import numpy as np
from PIL import Image

# Container format   (KEEP IN SYNC with the ComfyUI node)
MAGIC = b"M3DS"
VERSION = 1
FLAG_GZIP = 0x01
HEADER_FMT = ">4sBBHHQIIQIH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 40


def _read_image_bytes(path: str) -> bytes:
    img = Image.open(path)
    # convert("RGB") drops any alpha a pipeline may have added and depalettizes,
    # while preserving the underlying RGB values exactly.
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return arr.reshape(-1).tobytes()


def _parse(flat: bytes, source_name: str) -> dict:
    if len(flat) < HEADER_SIZE:
        raise ValueError(f"{source_name}: too small to be a smuggled image")
    (magic, version, flags, chunk_index, chunk_count,
     blob_total, chunk_payload_len, chunk_crc,
     orig_len, orig_crc, fname_len) = struct.unpack(HEADER_FMT, flat[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"{source_name}: not a smuggled image (bad magic)")
    if version != VERSION:
        raise ValueError(f"{source_name}: unsupported container version {version}")
    pos = HEADER_SIZE
    filename = flat[pos:pos + fname_len].decode("utf-8", "replace")
    pos += fname_len
    payload = flat[pos:pos + chunk_payload_len]
    if len(payload) != chunk_payload_len:
        raise ValueError(
            f"{source_name}: payload truncated (have {len(payload)}, "
            f"need {chunk_payload_len}) -- the image was likely resized/re-encoded")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != chunk_crc:
        raise ValueError(
            f"{source_name}: chunk CRC mismatch -- pixels were altered "
            f"(lossy re-encode or resize somewhere in the pipeline)")
    return {
        "flags": flags, "chunk_index": chunk_index, "chunk_count": chunk_count,
        "blob_total": blob_total, "orig_len": orig_len, "orig_crc": orig_crc,
        "filename": filename, "payload": payload,
    }


def reconstruct(paths, out_path=None, info_only=False):
    parsed = [_parse(_read_image_bytes(p), os.path.basename(p)) for p in paths]
    parsed.sort(key=lambda d: d["chunk_index"])

    head = parsed[0]
    chunk_count = head["chunk_count"]
    flags, filename = head["flags"], head["filename"]
    orig_len, orig_crc, blob_total = head["orig_len"], head["orig_crc"], head["blob_total"]

    if info_only:
        print(f"original filename : {filename}")
        print(f"original size     : {orig_len} bytes")
        print(f"compression       : {'gzip' if flags & FLAG_GZIP else 'none'}")
        print(f"blob size         : {blob_total} bytes")
        print(f"chunks expected   : {chunk_count}")
        print(f"chunks provided   : {len(parsed)}")
        print(f"chunk indices     : {[d['chunk_index'] for d in parsed]}")
        return None

    if len(parsed) != chunk_count:
        raise ValueError(
            f"expected {chunk_count} chunk image(s) but got {len(parsed)}; "
            f"provide all chunk PNGs in one command")
    indices = [d["chunk_index"] for d in parsed]
    if indices != list(range(chunk_count)):
        raise ValueError(f"chunk indices are wrong or duplicated: {indices}")

    blob = b"".join(d["payload"] for d in parsed)
    if len(blob) != blob_total:
        raise ValueError(f"reassembled blob is {len(blob)} bytes, expected {blob_total}")

    data = gzip.decompress(blob) if (flags & FLAG_GZIP) else blob
    if len(data) != orig_len:
        raise ValueError(f"decoded size {len(data)} != expected {orig_len}")
    if (zlib.crc32(data) & 0xFFFFFFFF) != orig_crc:
        raise ValueError("final CRC mismatch -- data corrupted in transit")

    if out_path is None:
        out_path = filename or "recovered.bin"
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"OK  wrote {len(data)} bytes -> {out_path}  (CRC verified)")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reconstruct a mesh/binary file from smuggled PNG(s).")
    ap.add_argument("images", nargs="+", help="PNG chunk(s) for ONE smuggled file")
    ap.add_argument("-o", "--output", default=None,
                    help="output path (default: original name embedded in the image)")
    ap.add_argument("--info", action="store_true",
                    help="print header info and exit (writes nothing)")
    args = ap.parse_args(argv)
    try:
        reconstruct(args.images, out_path=args.output, info_only=args.info)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
