#!/usr/bin/env python3
"""Top-down run cycle of the silver knight (128x128, 8 frames, facing the camera).

Design source: the knight bust in the reference art — silver helm with a pointed
visor ridge and a pair of tall gold horns, white scarf with a long trailing tail,
spiked silver pauldrons with gold thorns, chainmail under the arms, brown strap with a
gold buckle, blue sash, blue cape.

View: 3/4 top-down (camera ~45 degrees above and in front, rules/47). The skeleton is
posed in 3D (x sideways, h up, z toward the camera) with the run poses of
rules/42, then projected: height is squashed to 0.92 and depth to 0.55 of a pixel.
Shapes are rasterised at 4x supersampling and shaded from a pseudo height map
(light from the upper left), then outlined, so metal reads as volume.

Outputs (next to this file):
  knight-run-sheet.png      1024x128 transparent sprite sheet
  knight-run.aseprite       layered source (tag "run-s")
  knight-run.gif            looping preview
  knight-run.mp4            preview video over a scrolling cobble floor
  knight-run-forja.mp4      only when a background image path is passed as argv[1]
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "warrior-run"))
import build as wb  # reuse the .aseprite writer and the run pose table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W = H = 128
N = 8
SS = 4
CX = 64
DUR = [80, 80, 70, 100, 80, 80, 70, 100]
SY_H, SY_Z = 0.92, 0.55  # projection of height and depth onto screen rows


def hx(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def ramp(*cols):
    return [hx(c) for c in cols]


OUT = hx("#100e1c")
SILVER = ramp("#26264a", "#52567c", "#8a90b6", "#bcc2de", "#e8ebf7", "#ffffff")
GOLD = ramp("#4e300f", "#8e5c25", "#cc9636", "#efc65a", "#fff1a6")
CLOTH = ramp("#6a6c9c", "#a0a2cc", "#cfd0e8", "#eeeefa", "#ffffff")
BLUE = ramp("#10153f", "#1c2c74", "#2e4cae", "#5a82e0")
BROWN = ramp("#24150d", "#47291a", "#6e4528", "#9a6a3c")
MAIL = ramp("#14142a", "#232340", "#3a3a5e")
VOID = hx("#0d0c1a")
GLINT = hx("#bfe0ff")


# ------------------------------------------------------------ raster helpers
class Mask:
    def __init__(self):
        self.im = Image.new("L", (W * SS, H * SS), 0)
        self.d = ImageDraw.Draw(self.im)

    def poly(self, pts):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=255)
        return self

    def ellipse(self, c, rx, ry):
        self.d.ellipse([(c[0] - rx) * SS, (c[1] - ry) * SS, (c[0] + rx) * SS, (c[1] + ry) * SS], fill=255)
        return self

    def capsule(self, a, b, ra, rb=None):
        rb = ra if rb is None else rb
        dx, dy = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(dx, dy) or 1
        nx, ny = -dy / ln, dx / ln
        self.poly([(a[0] + nx * ra, a[1] + ny * ra), (b[0] + nx * rb, b[1] + ny * rb),
                   (b[0] - nx * rb, b[1] - ny * rb), (a[0] - nx * ra, a[1] - ny * ra)])
        self.ellipse(a, ra, ra)
        self.ellipse(b, rb, rb)
        return self

    def stroke(self, pts, r0, r1):
        """Tapered stroke along a polyline."""
        n = len(pts) - 1
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            self.capsule(pts[i], pts[i + 1], r0 + (r1 - r0) * t0, r0 + (r1 - r0) * t1)
        return self

    def arr(self):
        return np.array(self.im.resize((W, H), Image.BOX)) >= 128


def bez(p0, p1, p2, n=14):
    out = []
    for i in range(n + 1):
        t = i / n
        out.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
                    (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]))
    return out


def shift(m, dx, dy):
    o = np.zeros_like(m)
    ys, yd = slice(max(dy, 0), H + min(dy, 0)), slice(max(-dy, 0), H + min(-dy, 0))
    xs, xd = slice(max(dx, 0), W + min(dx, 0)), slice(max(-dx, 0), W + min(-dx, 0))
    o[ys, xs] = m[yd, xd]
    return o


def dilate(m):
    return m | shift(m, 1, 0) | shift(m, -1, 0) | shift(m, 0, 1) | shift(m, 0, -1)


def erode(m):
    return m & shift(m, 1, 0) & shift(m, -1, 0) & shift(m, 0, 1) & shift(m, 0, -1)


def blur(a):
    k = np.array([1, 2, 1], np.float32) / 4
    p = np.pad(a, 1, mode="edge")
    p = p[:, :-2] * k[0] + p[:, 1:-1] * k[1] + p[:, 2:] * k[2]
    return p[:-2] * k[0] + p[1:-1] * k[1] + p[2:] * k[2]


BAYER = np.array([[0, 2], [3, 1]], np.float32) / 4.0 - 0.375
LIGHT = np.array([-0.5, -0.62, 0.6], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


class Layer:
    def __init__(self, name):
        self.name = name
        self.px = np.zeros((H, W, 4), np.uint8)

    def fill(self, mask, rp, R=6.0, bias=0.0, dither=0.0, outline=True, contrast=2.4, flat=None):
        """Outline + volume shading from a dome height map; `rp` is a dark->light ramp."""
        if outline:
            self.px[dilate(mask) & ~mask] = OUT
        if flat is not None:
            self.px[mask] = flat
            return mask
        # height map from successive erosions
        d = np.zeros((H, W), np.float32)
        cur = mask.copy()
        for _ in range(int(R) + 1):
            d += cur
            cur = erode(cur)
        h = np.sin(np.clip(d / R, 0, 1) * math.pi / 2) * R * 0.9
        h = blur(blur(h))
        gy, gx = np.gradient(h)
        n = np.stack([-gx, -gy, np.ones_like(h)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        lam = n @ LIGHT
        mid = (len(rp) - 1) * 0.45
        t = (lam - 0.6) * contrast * (len(rp) - 1) / 3 + mid + bias
        if dither:
            yy, xx = np.indices((H, W))
            t = t + BAYER[yy % 2, xx % 2] * dither
        idx = np.clip(np.rint(t), 0, len(rp) - 1).astype(int)
        for i, c in enumerate(rp):
            self.px[mask & (idx == i)] = c
        return mask

    def dots(self, pts, color, size=1):
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                self.px[yi:yi + size, xi:xi + size] = color

    def line(self, a, b, color, clip=None):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        for i in range(n + 1):
            t = i / n
            x, y = int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))
            if 0 <= x < W and 0 <= y < H and (clip is None or clip[y, x]):
                self.px[y, x] = color

    def path(self, pts, color, clip=None):
        for a, b in zip(pts, pts[1:]):
            self.line(a, b, color, clip)

    def checker(self, mask, a, b):
        yy, xx = np.indices((H, W))
        self.px[mask & ((xx + yy) % 2 == 0)] = a
        self.px[mask & ((xx + yy) % 2 == 1)] = b


# ------------------------------------------------------------ pose
THIGH, SHIN, UPPER, FORE = 15.5, 15.5, 13.0, 12.5
HIPW, SHW, TORSO = 8.0, 21.0, 31.0


def proj(p, hip_y):
    x, h, z = p
    return (CX + x * (1 + 0.004 * z), hip_y - h * SY_H + z * SY_Z)


def solve(f):
    P = {}
    legs = {}
    for side, sgn, k in (("L", -1, 0), ("R", 1, 4)):
        th, kn = wb.LEG[(f + k) % 8]
        th *= 0.92
        hip3 = (sgn * HIPW, 0.0, 0.0)
        knee3 = (hip3[0], -THIGH * math.cos(math.radians(th)), THIGH * math.sin(math.radians(th)))
        a = th - kn
        ank3 = (hip3[0] + sgn * 0.5, knee3[1] - SHIN * math.cos(math.radians(a)), knee3[2] + SHIN * math.sin(math.radians(a)))
        legs[side] = dict(sgn=sgn, th=th, kn=kn, hip=hip3, knee=knee3, ank=ank3)
    lift = max(0, wb.LIFT[f]) * 2.0 + (wb.LIFT[f] if wb.LIFT[f] < 0 else 0) * 1.6
    sole = max(-l["ank"][1] * SY_H + l["ank"][2] * SY_Z for l in legs.values()) + 4.5
    hip_y = 120.0 - sole - lift * SY_H
    P["hip_y"] = hip_y
    P["legs"] = legs
    # torso leans toward the camera, shoulders counter-rotate against the hips
    lean_z = 5.0
    arms = {}
    for side, sgn, k in (("L", -1, 4), ("R", 1, 0)):
        th = wb.LEG[(f + k) % 8][0]
        swing = max(-58, min(60, -th * 1.1))
        sh3 = (sgn * SHW, TORSO, lean_z - sgn * (swing * 0.05))
        el3 = (sh3[0] + sgn * 3.5, sh3[1] - UPPER * math.cos(math.radians(swing)), sh3[2] + UPPER * math.sin(math.radians(swing)))
        fa = swing + (78 if abs(swing) < 40 else 88)
        wr3 = (el3[0] + sgn * 1.0, el3[1] - FORE * math.cos(math.radians(fa)), el3[2] + FORE * math.sin(math.radians(fa)))
        arms[side] = dict(sgn=sgn, swing=swing, sh=sh3, el=el3, wr=wr3)
    P["arms"] = arms
    return P


def scale_at(z):
    return 1 + 0.006 * z


# ------------------------------------------------------------ parts
def part_shadow(P):
    L = Layer("shadow")
    m = Mask().ellipse((CX, 118.5), 33, 7.5).arr()
    L.px[m] = (8, 6, 16, 95)
    inner = Mask().ellipse((CX, 118.5), 22, 4.5).arr()
    L.px[inner] = (8, 6, 16, 70)
    return L


def part_cape(P, f):
    L = Layer("cape")
    hy = P["hip_y"]
    sh_y = hy - TORSO * SY_H + 4
    t = f / N * 2 * math.pi
    for sgn in (-1, 1):
        w1 = math.sin(t + (0 if sgn < 0 else 1.6)) * 4.0
        w2 = math.sin(t + 1.3 + (0 if sgn < 0 else 1.6)) * 3.0
        top = (CX + sgn * 8, sh_y - 6)
        pts = [top,
               (CX + sgn * 24, sh_y - 14 + w1 * 0.5),
               (CX + sgn * 40, sh_y - 24 + w1),
               (CX + sgn * 43, sh_y - 8 + w1 * 0.8),
               (CX + sgn * 39, sh_y + 12 + w2),
               (CX + sgn * 36, sh_y + 30 + w2 * 1.3),
               (CX + sgn * 29, sh_y + 24 + w2),
               (CX + sgn * 22, sh_y + 33 + w2 * 1.2),
               (CX + sgn * 13, sh_y + 24),
               (CX + sgn * 6, sh_y + 16)]
        m = Mask().poly(pts).arr()
        L.fill(m, BLUE, R=9, contrast=1.7)
        for u in (0.3, 0.6):  # fold lines
            a = (CX + sgn * (14 + 30 * u), sh_y - 4 + w1 * u)
            b = (CX + sgn * (10 + 32 * u), sh_y + 26 + w2 * u)
            L.line(a, b, BLUE[0], clip=m)
            L.line((a[0] + sgn * 1, a[1]), (b[0] + sgn * 1, b[1]), BLUE[3], clip=m)
        # gold trim on the lower hem
        L.path([(CX + sgn * 44, sh_y + 33 + w2 * 1.3), (CX + sgn * 36, sh_y + 29 + w2), (CX + sgn * 26, sh_y + 37 + w2 * 1.2), (CX + sgn * 15, sh_y + 29)], GOLD[2], clip=m)
    return L


def boot_part(L, leg, hip_y, back):
    sgn = leg["sgn"]
    ank = proj(leg["ank"], hip_y)
    sc = scale_at(leg["ank"][2])
    th = leg["th"]
    trailing = th < -8
    ry = 4.6 if trailing else 5.8
    cy = ank[1] + (3.5 if trailing else 4.6)
    sab = Mask().ellipse((ank[0] + sgn * 0.3, cy), 5.0 * sc, ry * sc).arr()
    L.fill(sab, SILVER, R=5, contrast=2.2)
    # gold toe/edge line
    edge = Mask().ellipse((ank[0] + sgn * 0.3, cy), 5.0 * sc, ry * sc).arr()
    L.path([(ank[0] - 3, cy + ry * sc - 1), (ank[0] + 3, cy + ry * sc - 1)], GOLD[3], clip=edge)
    L.dots([(ank[0] - 2, cy - 1)], SILVER[5])


def leg_layer(P, side, name):
    L = Layer(name)
    leg = P["legs"][side]
    hy = P["hip_y"]
    sgn = leg["sgn"]
    hip = proj(leg["hip"], hy)
    knee = proj(leg["knee"], hy)
    ank = proj(leg["ank"], hy)
    sc_k = scale_at(leg["knee"][2])
    sc_a = scale_at(leg["ank"][2])
    # thigh (dark gambeson) and greave (silver)
    th = Mask().capsule((hip[0], hip[1] - 1), knee, 6.0, 5.2 * sc_k).arr()
    L.fill(th, BLUE, R=6, bias=-0.9, contrast=1.3)
    L.checker(th & ~erode(th) & ~erode(erode(th)) & False, MAIL[1], MAIL[0])
    sh = Mask().capsule(knee, (ank[0], ank[1] - 1), 4.8 * sc_k, 4.2 * sc_a).arr()
    L.fill(sh, SILVER, R=5, contrast=2.3)
    L.path([(knee[0], knee[1] + 3), (ank[0], ank[1] - 3)], GOLD[2], clip=sh)  # greave ridge filigree
    # spiked knee cop
    cop = Mask().ellipse((knee[0], knee[1] + 0.5), 5.4 * sc_k, 4.6 * sc_k).arr()
    L.fill(cop, SILVER, R=5, contrast=2.4)
    spike = Mask().poly([(knee[0] + sgn * 4, knee[1] - 2), (knee[0] + sgn * 10 * sc_k, knee[1] + 1), (knee[0] + sgn * 4, knee[1] + 4)]).arr()
    L.fill(spike, GOLD, R=3, contrast=2.0)
    boot_part(L, leg, hy, False)
    return L


def torso_layer(P, f):
    L = Layer("torso")
    hy = P["hip_y"]
    sh_y = hy - TORSO * SY_H + (P["arms"]["L"]["sh"][2] + P["arms"]["R"]["sh"][2]) * SY_Z / 2
    wob = (P["arms"]["R"]["sh"][2] - P["arms"]["L"]["sh"][2]) * 0.18  # shoulder tilt
    # chainmail under-shirt + gorget (visible at the sides and the neck)
    mail = Mask().poly([(CX - 26, sh_y + 2), (CX - 14, sh_y - 8), (CX + 14, sh_y - 8), (CX + 26, sh_y + 2),
                        (CX + 20, sh_y + 24), (CX + 12, hy - 2), (CX - 12, hy - 2), (CX - 20, sh_y + 24)]).arr()
    L.fill(mail, MAIL, R=4, contrast=1.2)
    L.checker(mail, MAIL[2], MAIL[0])
    # cuirass
    top_y = sh_y - 4
    plate = Mask().poly([(CX - 19, top_y + 2 + wob), (CX - 8, top_y - 2), (CX + 8, top_y - 2), (CX + 19, top_y + 2 - wob),
                         (CX + 16, sh_y + 14), (CX + 12, hy - 7), (CX - 12, hy - 7), (CX - 16, sh_y + 14)]).arr()
    L.fill(plate, SILVER, R=9, contrast=2.0)
    # keel ridge: bright centre line + two shaded halves
    keel = Mask().poly([(CX - 1.2, top_y), (CX + 1.2, top_y), (CX + 2.2, hy - 8), (CX - 2.2, hy - 8)]).arr() & plate
    L.px[keel] = SILVER[4]
    L.px[keel & (np.indices((H, W))[1] % 5 == 0)] = SILVER[5]
    # gold filigree arcs on each breast
    for sgn in (-1, 1):
        arc = bez((CX + sgn * 2, top_y + 5), (CX + sgn * 17, top_y + 6), (CX + sgn * 12, sh_y + 17))
        L.path(arc, GOLD[3], clip=plate)
        arc2 = bez((CX + sgn * 2, top_y + 13), (CX + sgn * 12, top_y + 14), (CX + sgn * 10, sh_y + 20))
        L.path(arc2, GOLD[2], clip=plate)
        L.dots([(CX + sgn * 14, sh_y + 8)], GOLD[4])
    # spiky fauld edge along the bottom of the breastplate
    for sgn in (-1, 1):
        sp = Mask().poly([(CX + sgn * 12, hy - 8), (CX + sgn * 17, hy - 3), (CX + sgn * 10, hy - 4)]).arr()
        L.fill(sp, SILVER, R=2, contrast=1.8)
    # blue sash with a knot on the left hip
    sash = Mask().poly([(CX - 14, hy - 8), (CX + 14, hy - 8), (CX + 13, hy - 1), (CX - 13, hy - 1)]).arr()
    L.fill(sash, BLUE, R=5, contrast=1.8)
    L.path([(CX - 12, hy - 6), (CX + 11, hy - 6)], BLUE[3], clip=sash)
    knot = Mask().ellipse((CX - 12, hy - 3), 4.5, 4.2).arr()
    L.fill(knot, BLUE, R=3, contrast=2.0)
    t = f / N * 2 * math.pi
    tail = Mask().stroke(bez((CX - 14, hy - 1), (CX - 22, hy + 1 + math.sin(t) * 2), (CX - 24, hy + 7 + math.sin(t + 1) * 3)), 2.6, 1.0).arr()
    L.fill(tail, BLUE, R=3, contrast=1.8)
    # brown strap with gold buckle, right shoulder -> left hip
    a, b = (CX + 14, top_y + 2), (CX - 11, hy - 6)
    strap = Mask().capsule(a, b, 2.1).arr() & (plate | sash)
    L.px[strap] = BROWN[2]
    L.px[strap & ~shift(strap, 1, 1)] = BROWN[3]
    L.px[strap & ~shift(strap, -1, -1)] = BROWN[1]
    mx, my = (a[0] * 0.45 + b[0] * 0.55), (a[1] * 0.45 + b[1] * 0.55)
    buck = Mask().poly([(mx - 3, my - 3), (mx + 3, my - 3), (mx + 3, my + 3), (mx - 3, my + 3)]).arr()
    L.fill(buck, GOLD, R=3, contrast=2.0)
    L.px[Mask().poly([(mx - 1.2, my - 1.2), (mx + 1.2, my - 1.2), (mx + 1.2, my + 1.2), (mx - 1.2, my + 1.2)]).arr()] = BROWN[1]
    return L


def tabard_layer(P, f):
    """White cloth panel hanging from the belt, swaying with the legs."""
    L = Layer("tabard")
    hy = P["hip_y"]
    t = f / N * 2 * math.pi
    lz = P["legs"]["L"]["knee"][2]
    rz = P["legs"]["R"]["knee"][2]
    top = hy + 1
    for sgn, z in ((-1, lz), (1, rz)):
        cx = CX + sgn * 4.6
        bot = top + 14 + z * 0.3
        sway = math.sin(t + (0 if sgn < 0 else math.pi)) * 2.2
        flap = Mask().poly([(cx - 4.6, top), (cx + 4.6, top), (cx + 5.2 + sway, bot - 3), (cx + sgn * 0.5 + sway * 0.6, bot + 2.5), (cx - 5.2 + sway, bot - 3)]).arr()
        L.fill(flap, CLOTH, R=7, contrast=1.6, dither=0.35, bias=0.6)
        L.path([(cx + sway * 0.4, top + 3), (cx + sway, bot - 1)], CLOTH[1], clip=flap)
        L.path([(cx - 5.2 + sway, bot - 3), (cx + sgn * 0.5 + sway * 0.6, bot + 2.5), (cx + 5.2 + sway, bot - 3)], BLUE[2], clip=dilate(flap))
    return L


def scarf_layer(P, f):
    L = Layer("scarf")
    hy = P["hip_y"]
    sh_y = hy - TORSO * SY_H + 5
    t = f / N * 2 * math.pi
    # neck wrap
    wrap = Mask().poly([(CX - 18, sh_y - 1), (CX - 10, sh_y - 9), (CX + 10, sh_y - 10), (CX + 19, sh_y - 2),
                        (CX + 13, sh_y + 5), (CX + 3, sh_y + 3), (CX - 9, sh_y + 5)]).arr()
    L.fill(wrap, CLOTH, R=6, contrast=1.6, dither=0.4, bias=0.6)
    for off in (-12, -2, 9):
        L.path(bez((CX + off, sh_y - 8), (CX + off + 3, sh_y - 2), (CX + off - 1, sh_y + 5)), CLOTH[1], clip=wrap)
    # diagonal drape falling to the right chest
    drape = Mask().stroke([(CX - 4, sh_y - 5), (CX + 9, sh_y + 2), (CX + 16, sh_y + 13), (CX + 14, sh_y + 21)], 4.6, 2.2).arr()
    L.fill(drape, CLOTH, R=6, contrast=1.6, dither=0.4, bias=0.6)
    L.path(bez((CX - 1, sh_y - 2), (CX + 9, sh_y + 5), (CX + 14, sh_y + 17)), CLOTH[1], clip=drape)
    # long tail streaming back, up and to the left, ending in a blue curl
    w1, w2 = math.sin(t) * 4, math.sin(t + 1.5) * 5
    spine = bez((CX - 14, sh_y - 8), (CX - 30 + w1, sh_y - 22 + w1), (CX - 50, sh_y - 30 + w2), 16)
    tail = Mask().stroke(spine, 6.2, 1.6).arr()
    L.fill(tail, CLOTH, R=5, contrast=1.6, dither=0.4, bias=0.6)
    L.path(spine[2:-3], CLOTH[1], clip=tail)
    curl = Mask().stroke(bez(spine[-1], (spine[-1][0] - 7, spine[-1][1] + 1), (spine[-1][0] - 7, spine[-1][1] + 7)), 2.3, 0.6).arr()
    L.fill(curl, BLUE, R=2, contrast=1.8)
    return L


def head_layers(P, f):
    hy = P["hip_y"]
    sh_y = hy - TORSO * SY_H + 3
    bob = [0, 1, 0, -1, 0, 1, 0, -1][f] * 0.0
    hc = (CX + 0.0, sh_y - 19 + bob)
    x, y = hc
    # horns (behind the helm)
    LH = Layer("horns")
    for sgn in (-1, 1):
        root = (x + sgn * 13, y + 2)
        path = bez(root, (x + sgn * 31, y - 8), (x + sgn * 14, y - 30), 18)
        path += bez(path[-1], (x + sgn * 11, y - 35), (x + sgn * 5, y - 36), 4)[1:]
        m = Mask().stroke(path, 3.4, 0.7).arr()
        LH.fill(m, GOLD, R=3, contrast=2.1)
        # thorn on the outer curve
        mid = path[7]
        th = Mask().poly([(mid[0] + sgn * 1, mid[1] - 3), (mid[0] + sgn * 8, mid[1] - 8), (mid[0] + sgn * 2, mid[1] + 3)]).arr()
        LH.fill(th, GOLD, R=2, contrast=2.0)
    # helm
    LHM = Layer("head")
    dome = Mask().poly([(x - 14, y - 2), (x - 12, y - 12), (x - 6, y - 18), (x + 6, y - 18), (x + 12, y - 12), (x + 14, y - 2),
                        (x + 11, y + 9), (x + 5, y + 16), (x, y + 21), (x - 5, y + 16), (x - 11, y + 9)])
    dome.ellipse((x, y - 6), 14, 13)
    dm = dome.arr()
    LHM.fill(dm, SILVER, R=10, contrast=2.0)
    # dark face opening under the brow, with two cold eye glints
    gap = Mask().poly([(x - 10.5, y + 2), (x - 3, y + 0.5), (x, y + 2), (x + 3, y + 0.5), (x + 10.5, y + 2),
                       (x + 8, y + 9), (x + 3, y + 13), (x - 3, y + 13), (x - 8, y + 9)]).arr() & dm
    LHM.px[gap] = VOID
    LHM.px[gap & (np.indices((H, W))[1] > y + 6) & ((np.indices((H, W))[0] + np.indices((H, W))[1]) % 2 == 0)] = MAIL[0]
    LHM.dots([(x - 6, y + 5), (x - 5, y + 5), (x + 5, y + 5), (x + 6, y + 5)], GLINT)
    # central ridge / keel, brightest on the left
    ridge = Mask().poly([(x - 1.5, y - 17), (x + 1.5, y - 17), (x + 3.2, y + 1), (x + 1, y + 20), (x - 1, y + 20), (x - 3.2, y + 1)]).arr() & dm
    LHM.px[ridge] = SILVER[4]
    LHM.px[ridge & (np.indices((H, W))[1] < y - 4)] = SILVER[5]
    LHM.px[ridge & (np.indices((H, W))[1] > y + 12)] = SILVER[3]
    # gold filigree + brow band
    for sgn in (-1, 1):
        LHM.path(bez((x + sgn * 3, y - 14), (x + sgn * 11, y - 8), (x + sgn * 9, y - 1)), GOLD[3], clip=dm)
        LHM.path(bez((x + sgn * 4, y - 6), (x + sgn * 9, y - 3), (x + sgn * 8, y + 1)), GOLD[2], clip=dm)
    LHM.path([(x - 12, y - 1), (x - 6, y - 2), (x, y - 0.5), (x + 6, y - 2), (x + 12, y - 1)], GOLD[3], clip=dm)
    # neck guard
    return LH, LHM


def pauldron_layer(P, side, name, f):
    L = Layer(name)
    hy = P["hip_y"]
    arm = P["arms"][side]
    sgn = arm["sgn"]
    sx, sy = proj(arm["sh"], hy)
    sx += sgn * 4
    sy -= 7
    sc = scale_at(arm["sh"][2])
    # main dome
    dome = Mask().ellipse((sx, sy), 12.5 * sc, 10.0 * sc).arr()
    L.fill(dome, SILVER, R=10, contrast=2.2)
    # layered lames below
    for i, k in enumerate((7.5, 10.5)):
        lame = Mask().ellipse((sx + sgn * i * 1.2, sy + k), (10.5 - i * 1.5) * sc, 3.8 * sc).arr()
        L.fill(lame, SILVER, R=3, contrast=2.0, bias=-0.3)
    # gold rim line and gold thorns
    L.path(bez((sx - 11 * sc, sy - 2), (sx, sy - 12 * sc), (sx + 11 * sc, sy - 2)), GOLD[3], clip=dilate(dome))
    L.path(bez((sx - 8 * sc, sy + 1), (sx, sy - 6 * sc), (sx + 8 * sc, sy + 1)), GOLD[1], clip=dome)
    for ang, ln, w in ((-20, 14, 4.2), (-65, 12, 3.4), (30, 11, 3.4)):
        a = math.radians(ang)
        bx, by = sx + sgn * math.cos(a) * 11 * sc, sy + math.sin(a) * 9 * sc
        tx, ty = sx + sgn * math.cos(a) * (11 + ln) * sc, sy + math.sin(a) * (9 + ln * 0.9) * sc
        nx, ny = -math.sin(a) * sgn, math.cos(a)
        th = Mask().poly([(bx + nx * w, by + ny * w), (tx, ty), (bx - nx * w, by - ny * w)]).arr()
        L.fill(th, GOLD, R=3, contrast=2.1)
    return L


def arm_layer(P, side, name, f, with_sword=False):
    L = Layer(name)
    hy = P["hip_y"]
    arm = P["arms"][side]
    sgn = arm["sgn"]
    sh = proj(arm["sh"], hy)
    el = proj(arm["el"], hy)
    wr = proj(arm["wr"], hy)
    sc_e, sc_w = scale_at(arm["el"][2]), scale_at(arm["wr"][2])
    sh = (sh[0], sh[1] - 1)
    if with_sword:
        pass  # sword drawn by caller on its own layer
    up = Mask().capsule(sh, el, 5.2, 4.6 * sc_e).arr()
    L.fill(up, MAIL, R=5, contrast=1.1)
    L.checker(up, MAIL[2], MAIL[0])
    L.px[up & ~shift(up, 1, 1)] = MAIL[2]
    fo = Mask().capsule(el, wr, 4.8 * sc_e, 4.2 * sc_w).arr()
    L.fill(fo, SILVER, R=5, contrast=2.3)
    L.path([(el[0], el[1] + 1), (wr[0], wr[1] - 1)], GOLD[2], clip=fo)
    # couter (elbow cop) with a thorn
    cop = Mask().ellipse(el, 5.0 * sc_e, 4.2 * sc_e).arr()
    L.fill(cop, SILVER, R=4, contrast=2.3)
    spike = Mask().poly([(el[0] + sgn * 3, el[1] - 4), (el[0] + sgn * 9 * sc_e, el[1] - 1), (el[0] + sgn * 3, el[1] + 2)]).arr()
    L.fill(spike, GOLD, R=3, contrast=2.0)
    # gauntlet
    fist = Mask().ellipse((wr[0], wr[1] + 1.5), 4.4 * sc_w, 4.2 * sc_w).arr()
    L.fill(fist, SILVER, R=4, contrast=2.3)
    L.path([(wr[0] - 3, wr[1] + 0.5), (wr[0] + 3, wr[1] + 0.5)], SILVER[1])
    L.path([(wr[0] - 3, wr[1] + 3), (wr[0] + 3, wr[1] + 3)], SILVER[1])
    return L, (wr[0], wr[1] + 1.5)


def sword_layer(P, f, hand):
    """Longsword in the left (screen) hand, blade trailing back and up past the shoulder."""
    L = Layer("sword")
    hx_, hy_ = hand
    swing = P["arms"]["L"]["swing"]
    ang = math.radians(200 + swing * 0.25)  # 180 = straight up; tilts outward as the arm swings
    d = (math.sin(ang) * -1, math.cos(ang))   # screen direction (x, y) of the blade
    d = (-0.78, -0.62)
    n = (-d[1], d[0])
    guard = (hx_ + d[0] * 5, hy_ + d[1] * 5)
    tip = (guard[0] + d[0] * 36, guard[1] + d[1] * 36)
    blade = Mask().poly([(guard[0] + n[0] * 3.2, guard[1] + n[1] * 3.2), (tip[0] + n[0] * 0.4, tip[1] + n[1] * 0.4),
                         (tip[0] - n[0] * 0.4, tip[1] - n[1] * 0.4), (guard[0] - n[0] * 3.2, guard[1] - n[1] * 3.2)]).arr()
    L.fill(blade, SILVER, R=3, contrast=2.6, bias=0.5)
    L.line((guard[0] + d[0] * 3, guard[1] + d[1] * 3), (tip[0] - d[0] * 4, tip[1] - d[1] * 4), SILVER[5])
    cg = Mask().capsule((guard[0] + n[0] * 8, guard[1] + n[1] * 8), (guard[0] - n[0] * 8, guard[1] - n[1] * 8), 2.1).arr()
    L.fill(cg, GOLD, R=2, contrast=2.1)
    gr = Mask().capsule(guard, (hx_ - d[0] * 5, hy_ - d[1] * 5), 2.0).arr()
    L.fill(gr, BROWN, R=2, contrast=1.6)
    pm = Mask().ellipse((hx_ - d[0] * 6.5, hy_ - d[1] * 6.5), 2.6, 2.6).arr()
    L.fill(pm, GOLD, R=2, contrast=2.0)
    return L


ORDER = ["shadow", "cape", "scarf-tail", "leg-back", "tabard", "leg-front", "arm-back", "torso", "scarf",
         "horns", "head", "pauldron-back", "pauldron-front", "arm-front", "sword"]


def build_frame(f):
    P = solve(f)
    hy = P["hip_y"]
    lz = P["legs"]["L"]["ank"][2]
    rz = P["legs"]["R"]["ank"][2]
    back_leg, front_leg = ("L", "R") if rz >= lz else ("R", "L")
    lw = P["arms"]["L"]["wr"][2]
    rw = P["arms"]["R"]["wr"][2]
    back_arm, front_arm = ("L", "R") if rw >= lw else ("R", "L")
    layers = {}
    layers["shadow"] = part_shadow(P)
    layers["cape"] = part_cape(P, f)
    sc = scarf_layer(P, f)
    layers["leg-back"] = leg_layer(P, back_leg, "leg-back")
    layers["tabard"] = tabard_layer(P, f)
    layers["leg-front"] = leg_layer(P, front_leg, "leg-front")
    arm_b, hand_b = arm_layer(P, back_arm, "arm-back", f)
    arm_f, hand_f = arm_layer(P, front_arm, "arm-front", f)
    layers["arm-back"] = arm_b
    layers["torso"] = torso_layer(P, f)
    layers["scarf"] = sc
    layers["scarf-tail"] = Layer("scarf-tail")  # tail lives on the scarf layer; kept for stable layer order
    horns, head = head_layers(P, f)
    layers["horns"], layers["head"] = horns, head
    layers["pauldron-back"] = pauldron_layer(P, back_arm, "pauldron-back", f)
    layers["pauldron-front"] = pauldron_layer(P, front_arm, "pauldron-front", f)
    layers["arm-front"] = arm_f
    sword_hand = hand_b if back_arm == "L" else hand_f
    layers["sword"] = sword_layer(P, f, sword_hand)
    # sword sits in the left hand: draw it before that arm's gauntlet by merging order
    return layers, back_arm


def flatten(layers, order):
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for n in order:
        out.alpha_composite(Image.fromarray(layers[n].px, "RGBA"))
    return np.array(out)


def main(bg_path=None):
    frames, per = [], []
    for f in range(N):
        layers, back_arm = build_frame(f)
        # the left-hand sword goes behind its own arm: composite order depends on which arm is in front
        order = list(ORDER)
        if back_arm == "R":  # left arm is in front -> sword behind arm-front
            order.remove("sword")
            order.insert(order.index("arm-front"), "sword")
        else:  # left arm is behind the torso -> sword goes behind the torso too
            order.remove("sword")
            order.insert(order.index("arm-back") + 1, "sword")
        frames.append(flatten(layers, order))
        per.append([layers[n].px for n in order])
    # ---- outputs
    sheet = Image.new("RGBA", (W * N, H), (0, 0, 0, 0))
    for i, a in enumerate(frames):
        sheet.paste(Image.fromarray(a, "RGBA"), (i * W, 0))
    sheet.save(os.path.join(HERE, "knight-run-sheet.png"))

    bg = (30, 26, 44, 255)
    g = []
    for f in range(N):
        im = Image.new("RGBA", (W, H), bg)
        im.alpha_composite(Image.fromarray(frames[f], "RGBA"))
        g.append(im.convert("RGB").resize((W * 4, H * 4), Image.NEAREST))
    g[0].save(os.path.join(HERE, "knight-run.gif"), save_all=True, append_images=g[1:], duration=DUR, loop=0, disposal=2)

    # aseprite file: layers are stored in the per-frame order; use a fixed stack instead
    stack = list(ORDER)
    stack.remove("sword")
    stack.append("sword")
    per_fixed = []
    for f in range(N):
        layers, _ = build_frame(f)
        per_fixed.append([layers[n].px for n in stack])
    wb.W = wb.H = W
    wb.N = N
    wb.DUR = DUR
    wb.ORDER = stack
    wb.write_aseprite(os.path.join(HERE, "knight-run.aseprite"), per_fixed)

    def render(name, frame_fn, count, fps=12.5):
        tmp = os.path.join(HERE, ".frames")
        os.makedirs(tmp, exist_ok=True)
        for n in range(count):
            frame_fn(n).save(os.path.join(tmp, f"f{n:04d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", os.path.join(HERE, name)], check=True)
        for fn in os.listdir(tmp):
            os.remove(os.path.join(tmp, fn))
        os.rmdir(tmp)

    # scrolling cobble floor: he runs toward the camera, so the ground moves up the screen
    VW, VH, Z = 192, 160, 5
    rng = np.random.default_rng(7)
    PER = 64
    cob = np.zeros((PER, VW, 3), np.uint8)
    cob[:] = (74, 62, 74)
    d = ImageDraw.Draw(Image.fromarray(cob))
    tile = Image.new("RGB", (VW, PER), (70, 58, 72))
    td = ImageDraw.Draw(tile)
    for row in range(0, PER, 16):
        off = 12 if (row // 16) % 2 else 0
        for col in range(-24, VW + 24, 24):
            c = int(rng.integers(-8, 9))
            td.rectangle([col + off, row, col + off + 22, row + 14], fill=(86 + c, 72 + c, 84 + c))
            td.line([col + off, row, col + off + 22, row], fill=(104 + c, 88 + c, 100 + c))
            td.line([col + off, row + 14, col + off + 22, row + 14], fill=(54, 44, 58))
    tile_np = np.array(tile)

    def scene(n):
        f = n % N
        off = int(round(n * 7.4)) % PER
        rows = (np.arange(VH) + off) % PER
        im = Image.fromarray(tile_np[rows], "RGB").convert("RGBA")
        im.alpha_composite(Image.fromarray(frames[f], "RGBA"), (VW // 2 - W // 2, VH - H - 10))
        return im.convert("RGB").resize((VW * Z, VH * Z), Image.NEAREST)

    render("knight-run.mp4", scene, N * 8)

    if bg_path:
        back = Image.open(bg_path).convert("RGB").resize((720, 1080), Image.LANCZOS)
        S = 3

        def warm(a):
            a = a.astype(np.float32)
            rgb = a[..., :3] * np.array([1.08, 0.88, 0.74]) + np.array([12, 4, 0])
            a[..., :3] = np.clip(rgb, 0, 255)
            return a.astype(np.uint8)

        big = [Image.fromarray(warm(a), "RGBA").resize((W * S, H * S), Image.NEAREST) for a in frames]

        def forge(n):
            im = back.copy().convert("RGBA")
            im.alpha_composite(big[n % N], (360 - W * S // 2, 760 - H * S + 12))
            return im.convert("RGB")

        render("knight-run-forja.mp4", forge, N * 6)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
