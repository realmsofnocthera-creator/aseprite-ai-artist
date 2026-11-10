#!/usr/bin/env python3
"""128x128 version of the warrior run cycle.

Reuses the skeleton, poses and parts of build.py (authored on a 64-unit
grid) and rasterises them at 2x, so outlines and shading bands are half as
thick relative to the body: more detail, same animation.

Outputs: warrior-run-128-sheet.png (1024x128), warrior-run-128.aseprite,
warrior-run-128.gif, warrior-run-128.mp4, and, when a background image path
is given as argv[1], warrior-run-128-forja.mp4 (runs across that scene).
"""
import os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
import build as b

K = 2
HERE = b.HERE
b.W = b.H = 64 * K

# --- scale every coordinate that parts feed to the rasteriser
_cap, _poly, _ell = b.Mask.capsule, b.Mask.poly, b.Mask.ellipse
b.Mask.capsule = lambda s, a, c, ra, rb=None: _cap(s, (a[0]*K, a[1]*K), (c[0]*K, c[1]*K), ra*K, None if rb is None else rb*K)
b.Mask.poly = lambda s, pts: _poly(s, [(x*K, y*K) for x, y in pts])
b.Mask.ellipse = lambda s, c, rx, ry: _ell(s, (c[0]*K, c[1]*K), rx*K, ry*K)

def dots(self, pts, color, size=K):
    for x, y in pts:
        xi, yi = int(round(x*K)), int(round(y*K))
        self.px[max(yi,0):yi+size, max(xi,0):xi+size] = color
def line(self, a, c, color):
    n = int(max(abs(c[0]-a[0]), abs(c[1]-a[1])) * K * 2) + 1
    for i in range(n + 1):
        t = i / n
        x, y = int(round((a[0]+(c[0]-a[0])*t)*K)), int(round((a[1]+(c[1]-a[1])*t)*K))
        if 0 <= x < b.W and 0 <= y < b.H:
            self.px[y, x] = color
b.Layer.dots, b.Layer.line = dots, line

# 2-px light rim / shadow band on large shapes for a richer ramp
_paint = b.Layer.paint
def paint(self, mask, ramp, outline=True, light=True, shade=True, mid=None):
    _paint(self, mask, ramp, outline, light, shade, mid)
    if light:
        inner = mask & ~b.shift(mask, 1, 1) & False
    if shade:
        band = mask & ~b.shift(mask, -2, -2) & b.shift(mask, 1, 1)
        keep = self.px[..., 3] > 0
        self.px[band & (self.px == np.array(mid if mid else ramp[1])).all(-1)] = ramp[0]
b.Layer.paint = paint

def main(bg_path=None):
    N, W, H = b.N, b.W, b.H
    frames, per = [], []
    for f in range(N):
        parts = b.build_frame(f)
        by = {p.name: p.px for p in parts}
        per.append([by[n] for n in b.ORDER]); frames.append(b.flatten(parts))
    sheet = Image.new("RGBA", (W*N, H), (0,0,0,0))
    for i, a in enumerate(frames): sheet.paste(Image.fromarray(a, "RGBA"), (i*W, 0))
    sheet.save(os.path.join(HERE, "warrior-run-128-sheet.png"))
    b.write_aseprite(os.path.join(HERE, "warrior-run-128.aseprite"), per)

    bg = (30, 26, 44, 255)
    g = []
    for f in range(N):
        im = Image.new("RGBA", (W, H), bg)
        ImageDraw.Draw(im).rectangle([0, 60*K, W, H], fill=(54, 46, 70, 255))
        im.alpha_composite(Image.fromarray(frames[f], "RGBA"))
        g.append(im.convert("RGB").resize((W*4, H*4), Image.NEAREST))
    g[0].save(os.path.join(HERE, "warrior-run-128.gif"), save_all=True, append_images=g[1:], duration=b.DUR, loop=0, disposal=2)

    def render(name, frame_fn, count, fps=12.5):
        tmp = os.path.join(HERE, ".frames"); os.makedirs(tmp, exist_ok=True)
        for n in range(count): frame_fn(n).save(os.path.join(tmp, f"f{n:04d}.png"))
        subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(fps),"-i",os.path.join(tmp,"f%04d.png"),
                        "-c:v","libx264","-pix_fmt","yuv420p","-crf","16",os.path.join(HERE,name)], check=True)
        for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
        os.rmdir(tmp)

    # plain scrolling-ground video
    VW, VH, Z = 240, 144, 5
    def scene(n):
        f = n % N; t = n
        im = Image.new("RGBA", (VW, VH), bg); d = ImageDraw.Draw(im)
        d.rectangle([0,0,VW,60], fill=(38,32,58,255)); d.rectangle([0,60,VW,100], fill=(46,38,66,255))
        for k in range(-1, 10):
            x = (k*34 - t*2) % (VW+34) - 17
            d.rectangle([x,30,x+9,100], fill=(60,48,82,255)); d.rectangle([x-1,27,x+10,30], fill=(70,56,94,255))
        gy = 60*K + 8
        d.rectangle([0,gy,VW,VH], fill=(74,56,52,255)); d.line([0,gy,VW,gy], fill=(110,84,70,255))
        for k in range(14):
            x = (k*20 - t*9) % (VW+20) - 10
            d.line([x,gy+8,x+9,gy+8], fill=(92,70,62,255)); d.line([x+5,gy+18,x+12,gy+18], fill=(60,46,46,255))
        d.ellipse([VW//2-22, gy-2, VW//2+14, gy+4], fill=(40,30,34,255))
        im.alpha_composite(Image.fromarray(frames[f], "RGBA"), (VW//2-W//2+8, gy-60*K))
        return im.convert("RGB").resize((VW*Z, VH*Z), Image.NEAREST)
    render("warrior-run-128.mp4", scene, N*8)

    if bg_path:
        back = Image.open(bg_path).convert("RGB").resize((720, 1080), Image.LANCZOS)
        S = 3  # sprite pixel = 3 screen px, close to the scene's own pixel size
        def warm(a):  # forge light: warm multiply, plus an orange rim on the side facing the fire
            a = a.astype(np.float32); rgb = a[..., :3]
            rgb = rgb * np.array([1.08, 0.86, 0.72]) + np.array([14, 4, 0])
            edge = (a[..., 3] > 0) & ~(np.roll(a[..., 3] > 0, 2, axis=0))  # top edge catches the glow
            rgb[edge] = rgb[edge] * 0.6 + np.array([255, 170, 70]) * 0.4
            a[..., :3] = np.clip(rgb, 0, 255)
            return a.astype(np.uint8)
        frames_big = [Image.fromarray(warm(a), "RGBA").resize((W*S, H*S), Image.NEAREST) for a in frames]
        total = N*6
        def forge(n):
            im = back.copy().convert("RGBA")
            f = n % N; p = n / (total-1)
            x = int(-60 + p*(720-300)); y = 760  # feet row (64-grid ground 60 -> 120*S/2)
            shadow = Image.new("RGBA", im.size, (0,0,0,0))
            ImageDraw.Draw(shadow).ellipse([x+130, y-8, x+270, y+14], fill=(0,0,0,110))
            im = Image.alpha_composite(im, shadow)
            im.alpha_composite(frames_big[f], (x, y - 60*K*S))
            return im.convert("RGB")
        render("warrior-run-128-forja.mp4", forge, total)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
