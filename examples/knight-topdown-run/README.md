# Knight top-down run

128×128, 8 frames, facing the camera (3/4 top-down, ~45°), redrawn from scratch after the silver
knight reference: tall gold horns, pointed visor ridge, white scarf with a long tail, spiked
pauldrons, chainmail, strap with gold buckle, blue sash and cape, longsword in the left hand.

| File | What |
|------|------|
| `knight-run-sheet.png` | 1024×128 transparent sprite sheet |
| `knight-run.aseprite` | 15 layers, 8 frames, per-frame timing (written by the script, not opened in Aseprite) |
| `knight-run.gif` / `knight-run.mp4` | Previews (the video scrolls a cobble floor at foot speed) |
| `build_knight.py` | Regenerates everything; `python3 build_knight.py [background.png]` also renders `knight-run-forja.mp4` |

Pose table and timings come from `rules/42-walk-and-run.md` (flight on frames 4 and 8); the skeleton is
posed in 3D and projected (height ×0.92, depth ×0.55). Only the S (toward camera) direction exists so far.
