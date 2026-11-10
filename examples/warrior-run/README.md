# Warrior run cycle

64×64, 8 frames, facing right, based on the hero of the "Provação do Guerreiro" art
(spiky dark hair, gold headband and earring, red cape, black top with leather straps, sword).

| File | What |
|------|------|
| `warrior-run-sheet.png` | 512×64 transparent sprite sheet (8 frames in a row) |
| `warrior-run.aseprite` | Layered source: 10 layers, 8 frames, tag `run`, per-frame timing |
| `warrior-run.gif` / `warrior-run.mp4` | Previews (the video scrolls a ground and runs 8 loops) |
| `warrior-run-strip.png` | Sheet at 2× on a dark ground |
| `build.py` | Regenerates everything (`python3 build.py`; needs Pillow, numpy, ffmpeg) |

Poses and timings follow `rules/42-walk-and-run.md` (flight on frames 4 and 8, 640–720 ms loop).
The `.aseprite` was written by `build.py` directly, not through the plugin, because no Aseprite
binary was available when it was made; it parses cleanly but has not been opened in Aseprite.

## 128×128 version

`build128.py` re-renders the same skeleton and poses at 2× (outlines and shading bands become
half as thick relative to the body). Outputs: `warrior-run-128-sheet.png` (1024×128),
`warrior-run-128.aseprite`, `warrior-run-128.gif`, `warrior-run-128.mp4`.
`python3 build128.py <background.png>` also renders `warrior-run-128-forja.mp4`, the warrior running
across a forge scene with warm lighting (the background image is not stored in this repo).
