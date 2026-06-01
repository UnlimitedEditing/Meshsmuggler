# ComfyUI Mesh Smuggler

Return `.glb` / `.stl` / `.fbx` (or any binary) through an **image-only** output
channel by packing the bytes into lossless PNG(s). Built for Graydient's hosted
ComfyUI API while it lacks dedicated 3D output slots.

PNG is lossless (DEFLATE), so a clean pass-through round-trips byte-for-byte.
Every chunk carries a CRC and the whole file carries a CRC, so if the channel
ever re-encodes or resizes the image, the decoder tells you exactly that instead
of silently handing you a corrupt mesh.

## Install on Graydient
1. Push this folder to a GitHub repo.
2. In Graydient's custom-nodes box, paste the repo URL (one per line).
3. In the PIP requirements box add (both ship with ComfyUI; listed for safety):
   ```
   Pillow
   numpy
   ```
4. Let it auto-install / restart.

## Wire it up
In your TripoSG graph, connect `SaveTrimesh`'s `glb_path` output into this node:

```
SaveTrimesh ──(glb_path)──▶ Smuggle Mesh As Image (glb_path)
```

`SaveTrimesh` already writes a `.glb` and emits its path on `glb_path` — keep its
"save file" widget on. The smuggler reads that file's bytes, so everything baked
into the `.glb` (geometry + embedded textures) is captured.

A ready-made graph is included: **`STANDARD_api_triposg_smuggler.json`**. It is
your TripoSG workflow with the gate + smuggler already wired and the smuggler
defaulting to OFF.

## Optional output, controlled by a slot (the "if-statement" pattern)
The `Mesh Smuggle Gate` node is a switch in graph form. It passes the mesh path
through to the smuggler only when its `enable` input is non-zero; at 0 (default)
the smuggler branch is pruned and the rest of the workflow runs exactly as
before. This is the same trick as keying a switch off an exposed value (e.g.
loading a turbo LoRA only when CFG == 1).

Chain in the included graph:

```
SaveTrimesh ─(glb_path)─▶ Mesh Smuggle Gate ─(glb_path)─▶ Smuggle Mesh As Image
                                ▲
                             enable  ◀── map Graydient slot1 here (0 = off, 1 = on)
```

On a current ComfyUI build the disabled branch is skipped outright via
`ExecutionBlocker` (no wasted work). On older builds the gate falls back to a
sentinel string and the smuggler simply no-ops — either way, slot1 = 0 produces
no smuggled PNG.

What you get back per setting:
- **slot1 = 0** (default): the preview PNG only. The `.glb` is still written
  server-side by SaveTrimesh; it just isn't retrievable until Graydient adds the
  3D slot.
- **slot1 = 1**: the preview PNG **plus** the smuggled mesh PNG(s).

### Binding slot1 on Graydient
Upload the api-form workflow, then in the field-mapping step pick the
**Mesh Smuggle Gate** node and map Graydient's **slot1** field to its **`enable`**
input (the "select visually" button helps). Per Graydient's docs, custom/zero-day
nodes sometimes need their team to register the field — if `enable` doesn't show
up as mappable, ping them on Telegram from your dashboard. If their mapping
prefers a named PrimitiveNode over a node widget, convert `enable` to an input
and feed it from a `PrimitiveNode` titled `slot1`.

Node inputs:
- `glb_path`        — path piped from SaveTrimesh
- `filename_prefix` — output PNG name prefix (default `smuggle/mesh`)
- `compress`        — gzip before packing (default on)
- `max_dimension`   — max width/height per PNG (default 2048). Files too large for
                      one image are split into multiple numbered PNGs automatically.

Outputs are registered as normal image outputs, so they come back through the API
(retrieved by the smuggler node's id, like any SaveImage output).

## Decode locally
```bash
# single image
python unsmuggle_mesh.py mesh_00001_.png -o creature.glb

# multi-chunk: pass every chunk in one call
python unsmuggle_mesh.py mesh_00001_.png mesh_00002_.png -o creature.glb

# inspect a header without writing anything
python unsmuggle_mesh.py mesh_00001_.png --info
```

## Run ONE round-trip test before relying on it
Smuggle a known small `.glb`, download the PNG, decode, and compare hashes:

```bash
shasum -a 256 original.glb recovered.glb   # macOS / Linux
```

- **Hashes match** → the channel is byte-exact; you're done.
- **CRC mismatch from the decoder** → Graydient is transcoding the image. Lower
  `max_dimension` (some pipelines downscale large images) and confirm the output
  stays PNG, not JPEG.

## Container format (v1)
`MAGIC "M3DS"` · version · flags(gzip) · chunk_index · chunk_count ·
blob_total_len · chunk_payload_len · chunk_crc32 · orig_file_len · orig_crc32 ·
filename_len · filename · payload — laid out as RGB pixels, zero-padded to a
near-square image. The 40-byte header definition is duplicated in the node and the
decoder and must stay in sync.
