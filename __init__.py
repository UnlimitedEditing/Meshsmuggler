"""
ComfyUI Mesh Smuggler
=====================
Packs an arbitrary binary file (e.g. the .glb written by SaveTrimesh) into one or
more *lossless* PNG images so it can be returned through an image-only output
channel -- such as Graydient's hosted ComfyUI API, which does not yet expose
.glb / .stl / .fbx output slots.

Decode the PNG(s) back to the original file with unsmuggle_mesh.py.

Wiring:  SaveTrimesh.glb_path  -->  Smuggle Mesh As Image (glb_path)

License: MIT
"""

import os
import math
import gzip
import zlib
import struct

import numpy as np
from PIL import Image

try:
    import folder_paths  # provided by ComfyUI at runtime
    _HAS_FOLDER_PATHS = True
except Exception:
    _HAS_FOLDER_PATHS = False


# ---------------------------------------------------------------------------
# Container format   (KEEP IN SYNC with unsmuggle_mesh.py)
# ---------------------------------------------------------------------------
MAGIC = b"M3DS"            # Mesh-3D-Smuggle
VERSION = 1
FLAG_GZIP = 0x01
HEADER_FMT = ">4sBBHHQIIQIH"
#              |  | | | | | | | | | +- filename_len      (H, uint16)
#              |  | | | | | | | | +--- orig_crc32         (I, uint32)
#              |  | | | | | | | +----- orig_file_len      (Q, uint64)
#              |  | | | | | | +------- chunk_crc32        (I, uint32)
#              |  | | | | | +--------- chunk_payload_len  (I, uint32)
#              |  | | | | +----------- blob_total_len     (Q, uint64)
#              |  | | | +------------- chunk_count        (H, uint16)
#              |  | | +--------------- chunk_index        (H, uint16)
#              |  | +----------------- flags              (B, uint8)
#              |  +------------------- version            (B, uint8)
#              +---------------------- magic              (4 bytes)
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 40
CHANNELS = 3                                # RGB only (no alpha -> nothing to premultiply/strip)

# Sentinel used by the gate to tell the smuggler "do nothing" on ComfyUI builds
# that lack ExecutionBlocker. On modern builds the smuggler is skipped outright.
DISABLED_SENTINEL = "__SMUGGLE_DISABLED__"
try:
    # Available on current ComfyUI; lets the gate truly prune the branch.
    from comfy_execution.graph import ExecutionBlocker
except Exception:
    ExecutionBlocker = None


def _bytes_to_rgb_array(raw: bytes) -> np.ndarray:
    """Lay a byte string out as a near-square RGB uint8 image (zero-padded)."""
    pad = (-len(raw)) % CHANNELS
    if pad:
        raw = raw + b"\x00" * pad
    n_pixels = len(raw) // CHANNELS
    if n_pixels == 0:
        n_pixels = 1
        raw = b"\x00" * CHANNELS
    width = max(1, math.ceil(math.sqrt(n_pixels)))
    height = math.ceil(n_pixels / width)
    pixel_pad = width * height - n_pixels
    if pixel_pad:
        raw = raw + b"\x00" * (pixel_pad * CHANNELS)
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, CHANNELS)).copy()


def build_chunks(data: bytes, compress: bool, filename: str, max_dimension: int):
    """Encode `data` into a list of RGB uint8 arrays. Returns (arrays, flags, chunk_count)."""
    orig_len = len(data)
    orig_crc = zlib.crc32(data) & 0xFFFFFFFF

    flags = 0
    blob = data
    if compress:
        blob = gzip.compress(data, compresslevel=9)
        flags |= FLAG_GZIP
    blob_total = len(blob)

    fname_bytes = filename.encode("utf-8")
    if len(fname_bytes) > 0xFFFF:
        raise ValueError("filename too long")
    header_size = HEADER_SIZE + len(fname_bytes)

    max_pixels = int(max_dimension) * int(max_dimension)
    capacity = max_pixels * CHANNELS - header_size
    if capacity <= 0:
        raise ValueError("max_dimension is too small to hold even the header")

    if blob_total == 0:
        payloads = [b""]
    else:
        payloads = [blob[i:i + capacity] for i in range(0, blob_total, capacity)]
    chunk_count = len(payloads)
    if chunk_count > 0xFFFF:
        raise ValueError("data too large: > 65535 chunks; raise max_dimension")

    arrays = []
    for idx, payload in enumerate(payloads):
        chunk_crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, flags, idx, chunk_count,
            blob_total, len(payload), chunk_crc, orig_len, orig_crc, len(fname_bytes),
        )
        arrays.append(_bytes_to_rgb_array(header + fname_bytes + payload))
    return arrays, flags, chunk_count


class SmuggleMeshAsImage:
    """Pack a binary file into lossless PNG(s) for retrieval through an image output."""

    def __init__(self):
        self.output_dir = (
            folder_paths.get_output_directory() if _HAS_FOLDER_PATHS else os.path.abspath("./output")
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "forceInput": True}),
                "filename_prefix": ("STRING", {"default": "smuggle/mesh"}),
                "compress": ("BOOLEAN", {"default": True}),
                "max_dimension": ("INT", {"default": 2048, "min": 64, "max": 16384, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("png_paths",)
    FUNCTION = "smuggle"
    OUTPUT_NODE = True
    CATEGORY = "mesh_smuggle"

    def _resolve(self, glb_path: str) -> str:
        glb_path = (glb_path or "").strip()
        if not glb_path:
            raise ValueError("glb_path is empty -- is SaveTrimesh wired in and set to save a file?")
        if os.path.isfile(glb_path):
            return glb_path
        if _HAS_FOLDER_PATHS:
            cand = os.path.join(folder_paths.get_output_directory(), glb_path)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(f"Could not find mesh file: {glb_path}")

    def smuggle(self, glb_path, filename_prefix, compress, max_dimension):
        if (glb_path or "").strip() == DISABLED_SENTINEL:
            print("[MeshSmuggler] disabled via gate (slot=0); nothing emitted.")
            return {"ui": {"images": []}, "result": ("",)}
        src = self._resolve(glb_path)
        with open(src, "rb") as f:
            data = f.read()
        fname = os.path.basename(src)

        arrays, flags, chunk_count = build_chunks(data, bool(compress), fname, int(max_dimension))

        if _HAS_FOLDER_PATHS:
            full_output_folder, filename, counter, subfolder, _ = \
                folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        else:
            full_output_folder = self.output_dir
            os.makedirs(full_output_folder, exist_ok=True)
            filename, counter, subfolder = "mesh", 0, ""

        results, saved = [], []
        for arr in arrays:
            img = Image.fromarray(arr, mode="RGB")
            file = f"{filename}_{counter:05}_.png"
            fp = os.path.join(full_output_folder, file)
            # compress_level affects file size only; PNG pixel values are preserved exactly.
            img.save(fp, format="PNG", compress_level=6)
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            saved.append(fp)
            counter += 1

        h, w = arrays[0].shape[:2]
        print(f"[MeshSmuggler] {fname}: {len(data)} bytes "
              f"({'gzip' if flags & FLAG_GZIP else 'raw'}) -> {chunk_count} PNG(s); "
              f"first image {w}x{h}")
        return {"ui": {"images": results}, "result": ("\n".join(saved),)}


class MeshSmuggleGate:
    """Switch-style 'if' gate for the smuggler -- an if-statement in graph form.

    Pass the mesh path straight through when `enable` != 0, otherwise prune the
    downstream smuggler so no PNG is produced and the rest of the workflow runs
    untouched. Bind `enable` to a Graydient field (e.g. slot1): set it to 1 to
    turn smuggling on, leave at 0 (default) to keep it off.

    This mirrors the common trick of wiring a switch off an exposed value (the
    way some workflows load a turbo LoRA only when CFG == 1). On a ComfyUI build
    with ExecutionBlocker the disabled branch is skipped outright; on older
    builds it falls back to a sentinel that makes the smuggler no-op.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "forceInput": True}),
                "enable": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "gate"
    CATEGORY = "mesh_smuggle"

    def gate(self, glb_path, enable):
        try:
            on = float(enable) != 0.0
        except (TypeError, ValueError):
            on = False
        if on:
            return (glb_path,)
        if ExecutionBlocker is not None:
            return (ExecutionBlocker(None),)   # prune the smuggler entirely
        return (DISABLED_SENTINEL,)            # fallback: smuggler sees this and no-ops


NODE_CLASS_MAPPINGS = {
    "SmuggleMeshAsImage": SmuggleMeshAsImage,
    "MeshSmuggleGate": MeshSmuggleGate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SmuggleMeshAsImage": "Smuggle Mesh As Image",
    "MeshSmuggleGate": "Mesh Smuggle Gate (slot toggle)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
