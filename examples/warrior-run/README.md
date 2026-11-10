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
