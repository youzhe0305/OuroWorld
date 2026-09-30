# Using your own 3DGS: the scene package

Every stage reads a scene package, a directory with two files and, after the
reference view is chosen, a third:

```text
scene/
├── gaussians.ply   the static 3DGS, INRIA layout (x, y, z, f_dc_*, f_rest_*, opacity,
│                   scale_*, rot_*); any SH degree, padded to 3 in training
├── scene.json      cameras, clip planes, pivot, reference view
└── reference.png   the 3DGS rendered from the reference camera
```

`scene.json` (schema version 1):

```json
{
  "schema_version": 1,
  "background": [0.0, 0.0, 0.0],
  "clip": {"near": 0.01, "far": 100.0},
  "world_up": [0.0, -1.0, 0.0],
  "import_camera": {"width": 1732, "height": 1000, "K": [[...]], "c2w": [[...]]},
  "pivot": {"world": [x, y, z], "pixel": [u, v], "depth": d, "source": "click"},
  "reference": {"camera": {...}, "image": "reference.png", "selected": "reference"}
}
```

- Cameras are pinhole, OpenCV convention (x right, y down, z forward), `c2w`
  camera-to-world, `K` in pixels with the principal point at the image centre.
- `import_camera` is any camera that sees the part of the scene to animate.
  The pivot is picked on its render, and the reference candidates are placed
  around it.
- `world_up` defaults to the import camera's up direction.
- `clip.near` / `far` bound the rendering. The rasterizer also culls
  everything closer than 0.2 units, so a world only a few units across must
  be scaled up (see `ouroworld/datasets/world_scale.py`, used by
  `ouroworld-import lyra`).

To build a package from your own PLY, give the generic importer a camera JSON
(`{"width", "height", "K", "c2w"}`, the conventions above; the format of the
released `reference_camera.json`), or let it place the camera on a world axis:

```bash
ouroworld-import world --ply my_scene.ply --camera my_camera.json --out output/Custom/my_scene/scene
ouroworld-import world --ply my_scene.ply --forward=-z --up=+y --position=0,1.5,0 \
    --fov 90 60 --size 1600 900 --out output/Custom/my_scene/scene
ouroworld-pivot     dataset=Custom scene=my_scene   # click the pivot in the browser
ouroworld-reference dataset=Custom scene=my_scene   # look at reference_candidates/candidates.png,
ouroworld-reference dataset=Custom scene=my_scene reference.selected=candidate_012  # then pick one
```
