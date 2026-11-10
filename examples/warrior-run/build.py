#!/usr/bin/env python3
"""Pixel-art run cycle for the Nocthera warrior (64x64, 8 frames, facing right).

Design source: the hero of the "Provacao do Guerreiro" key art — spiky dark
hair, gold headband and hoop earring, bare shoulders, black sleeveless top
with leather/gold straps, red cape, sword.

Poses follow rules/42-walk-and-run.md (8-frame run, flight on frames 4 and 8,
arms bent ~90 degrees, lean ~10 degrees, cape and hair trailing).

Outputs (next to this file):
  warrior-run.aseprite   layered source (one layer per body part, tag "run")
  warrior-run-sheet.png  8x1 sprite sheet, 512x64, transparent
  warrior-run-strip.png  the sheet at 4x on a dark ground
  warrior-run.gif        looping preview, 6x
  warrior-run.mp4        preview video (scrolling ground), made with ffmpeg

Usage: python3 build.py   (needs Pillow, numpy, ffmpeg on PATH)
"""
import math
import os
import struct
import subprocess
import zlib

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
W = H = 64
N = 8
GROUND = 60
SS = 4  # supersampling for clean shapes
DUR = [80, 80, 70, 100, 80, 80, 70, 100]  # ms, rules/42 run table (flight held longer)

# ---------------------------------------------------------------- palette
def hx(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4)) + (255,)

PAL = {
    "outline": hx("#120f1c"),
    "hair": [hx("#0d0d18"), hx("#1e2036"), hx("#34406e")],
    "skin": [hx("#a05f40"), hx("#e0905e"), hx("#f2bb8a")],
    "gold": [hx("#94622a"), hx("#e2b04a"), hx("#f6da90")],
    "cape": [hx("#6e0a18"), hx("#b01826"), hx("#dc4630")],
    "top": [hx("#100d16"), hx("#1f1a27"), hx("#3c3244")],
    "leather": [hx("#4a2e18"), hx("#734b20"), hx("#a46c32")],
    "pants": [hx("#17141d"), hx("#2a2632"), hx("#4a4256")],
    "boot": [hx("#2a1810"), hx("#4a2e1e"), hx("#704c30")],
    "steel": [hx("#4e5a6c"), hx("#9aa8b8"), hx("#e4ecf4")],
    "eye": hx("#f4efe6"),
    "iris": hx("#14101a"),
}

# ---------------------------------------------------------------- pose data
# One leg's cycle: (thigh angle, knee flexion) in degrees. Angles are measured
# from straight down, positive = forward (to the right). The far leg is the
# near leg half a cycle later.
LEG = [
    (36, 18),    # 0 contact: heel strikes ahead of the hip
    (14, 56),    # 1 down: knee most bent, weight over the foot
    (-18, 14),   # 2 push: leg extended behind, toe-off
    (-42, 62),   # 3 flight: trailing, shin folding up
    (-20, 112),  # 4 recover: heel kick-up
    (24, 104),   # 5 swing through
    (56, 74),    # 6 knee drive
    (52, 36),    # 7 reaching
]
THIGH, SHIN = 11.0, 11.0
UPPER, FORE = 8.5, 8.5
LIFT = [0, -2, 0, 3, 1, 0, 0, 3]      # extra hip rise (+ = up); flight on 3 and 7


def fk(origin, length, ang):
    """End point of a limb: angle from straight down, + = forward (right)."""
    a = math.radians(ang)
    return (origin[0] + length * math.sin(a), origin[1] + length * math.cos(a))


def leg_points(hip, f):
    th, kn = LEG[f % N]
    knee = fk(hip, THIGH, th)
    ankle = fk(knee, SHIN, th - kn)
    return knee, ankle


def solve(f):
    """All joint positions for frame f (float coordinates, y down)."""
    hx_ = 30.0
    # which foot is lowest decides the hip height
    ext = []
    for k in (0, 4):
        th, kn = LEG[(f + k) % N]
        knee = fk((0, 0), THIGH, th)
        ankle = fk(knee, SHIN, th - kn)
        ext.append(ankle[1])
    hip_y = GROUND - 2 - max(ext) - LIFT[f]
    hip = (hx_, hip_y)
    lean = 11.0
    sh_len = 14.0
    shoulder = fk(hip, sh_len, 180 - lean)            # torso goes up, leans forward
    shoulder = (hip[0] + sh_len * math.sin(math.radians(lean)), hip[1] - sh_len * math.cos(math.radians(lean)))
    neck = (shoulder[0] + 1.0, shoulder[1] - 2.0)
    head_lag = -1.0 if f in (1, 5) else 0.0
    head = (neck[0] + 2.0, neck[1] - 6.0 - head_lag * 0.0 + (1.0 if f in (1, 5) else 0.0) * 0)
    near = {}
    far = {}
    for tag, k in (("n", 0), ("f", 4)):
        knee, ankle = leg_points(hip, f + k)
        th, kn = LEG[(f + k) % N]
        toe_dir = 1.0
        # foot angle: flat on contact/down, pointed when trailing
        foot_ang = 90 - max(0, -th) * 0.9 + (-14 if th > 30 else 0)
        toe = fk(ankle, 5.5, foot_ang)
        heel = fk(ankle, 1.5, foot_ang + 180)
        d = dict(knee=knee, ankle=ankle, toe=toe, heel=heel, thigh=th, shin=th - kn)
        (near if tag == "n" else far)["leg"] = d
    # arms swing opposite the legs
    for tag, k in (("n", 4), ("f", 0)):
        th = LEG[(f + k) % N][0]
        swing = -th * 1.05
        swing = max(-62, min(58, swing))
        sh = (shoulder[0] - 1.0, shoulder[1] + 1.0)
        elbow = fk(sh, UPPER, swing)
        flex = 88 if abs(swing) < 45 else 96
        wrist = fk(elbow, FORE, swing + flex)
        (near if tag == "n" else far)["arm"] = dict(sh=sh, elbow=elbow, wrist=wrist)
    return dict(hip=hip, shoulder=shoulder, neck=neck, head=head, near=near, far=far)


# ---------------------------------------------------------------- raster helpers
class Mask:
    """Supersampled 1-bit mask, downsampled with a 50% threshold."""

    def __init__(self):
        self.im = Image.new("L", (W * SS, H * SS), 0)
        self.d = ImageDraw.Draw(self.im)

    def capsule(self, a, b, ra, rb=None):
        rb = ra if rb is None else rb
        ax, ay, bx, by = [v * SS for v in (a[0], a[1], b[0], b[1])]
        ra, rb = ra * SS, rb * SS
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy) or 1
        nx, ny = -dy / ln, dx / ln
        self.d.polygon([(ax + nx * ra, ay + ny * ra), (bx + nx * rb, by + ny * rb),
                        (bx - nx * rb, by - ny * rb), (ax - nx * ra, ay - ny * ra)], fill=255)
        self.d.ellipse([ax - ra, ay - ra, ax + ra, ay + ra], fill=255)
        self.d.ellipse([bx - rb, by - rb, bx + rb, by + rb], fill=255)

    def poly(self, pts):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=255)

    def ellipse(self, c, rx, ry):
        self.d.ellipse([(c[0] - rx) * SS, (c[1] - ry) * SS, (c[0] + rx) * SS, (c[1] + ry) * SS], fill=255)

    def array(self):
        small = self.im.resize((W, H), Image.BOX)
        return np.array(small) >= 128


def shift(m, dx, dy):
    o = np.zeros_like(m)
    ys = slice(max(dy, 0), H + min(dy, 0))
    yd = slice(max(-dy, 0), H + min(-dy, 0))
    xs = slice(max(dx, 0), W + min(dx, 0))
    xd = slice(max(-dx, 0), W + min(-dx, 0))
    o[ys, xs] = m[yd, xd]
    return o


def dilate(m):
    o = m.copy()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        o |= shift(m, dx, dy)
    return o


class Layer:
    """RGBA layer built from masks painted back to front with 1px outlines."""

    def __init__(self, name):
        self.name = name
        self.px = np.zeros((H, W, 4), np.uint8)

    def paint(self, mask, ramp, outline=True, light=True, shade=True, mid=None):
        m = mask
        if outline:
            ring = dilate(m) & ~m
            self.px[ring] = PAL["outline"]
        c = ramp[1] if mid is None else mid
        self.px[m] = c
        if light:  # top-left rim catches light
            lit = m & ~shift(m, 1, 1)
            self.px[lit] = ramp[2]
        if shade:  # bottom-right falls into shadow
            dk = m & ~shift(m, -1, -1)
            self.px[dk & ~(m & ~shift(m, 1, 1))] = ramp[0]

    def dots(self, pts, color):
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                self.px[yi, xi] = color

    def line(self, a, b, color):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        for i in range(n + 1):
            t = i / n
            self.dots([(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)], color)


def clip(layer_px, mask):
    layer_px[~mask] = 0


# ---------------------------------------------------------------- body parts
def leg_layer(name, P, far):
    L = Layer(name)
    hip = P["hip"]
    leg = (P["far"] if far else P["near"])["leg"]
    knee, ankle, toe, heel = leg["knee"], leg["ankle"], leg["toe"], leg["heel"]
    pants = [tuple(int(c * (0.78 if far else 1)) for c in col[:3]) + (255,) for col in PAL["pants"]]
    boot = [tuple(int(c * (0.8 if far else 1)) for c in col[:3]) + (255,) for col in PAL["boot"]]
    # thigh + shin in trousers
    m = Mask()
    m.capsule((hip[0] - 0.5, hip[1] - 1), knee, 3.6, 2.9)
    m.capsule(knee, (ankle[0] + (ankle[0] - knee[0]) * -0.0, ankle[1] - 1.5), 2.9, 2.2)
    L.paint(m.array(), pants)
    # boot: tall shaft from just under the knee line + foot
    mb = Mask()
    shaft_top = (knee[0] + (ankle[0] - knee[0]) * 0.3, knee[1] + (ankle[1] - knee[1]) * 0.3)
    mb.capsule(shaft_top, ankle, 2.6, 2.4)
    mb.capsule(heel, toe, 1.9, 1.5)
    L.paint(mb.array(), boot)
    # gold cuff at the top of the boot
    cuff = Mask()
    cuff.capsule(shaft_top, (shaft_top[0] + (ankle[0] - shaft_top[0]) * 0.14, shaft_top[1] + (ankle[1] - shaft_top[1]) * 0.14), 2.7, 2.6)
    cm = cuff.array() & mb.array()
    gold = [tuple(int(c * (0.8 if far else 1)) for c in col[:3]) + (255,) for col in PAL["gold"]]
    L.paint(cm, gold, outline=False)
    return L


def arm_layer(name, P, far):
    L = Layer(name)
    a = (P["far"] if far else P["near"])["arm"]
    sh, el, wr = a["sh"], a["elbow"], a["wrist"]
    k = 0.8 if far else 1.0
    skin = [tuple(int(c * k) for c in col[:3]) + (255,) for col in PAL["skin"]]
    gold = [tuple(int(c * k) for c in col[:3]) + (255,) for col in PAL["gold"]]
    m = Mask()
    m.capsule(sh, el, 3.0, 2.5)
    m.capsule(el, wr, 2.5, 2.1)
    L.paint(m.array(), skin)
    # gold bracer on the forearm
    br = Mask()
    mid = (el[0] + (wr[0] - el[0]) * 0.35, el[1] + (wr[1] - el[1]) * 0.35)
    br.capsule(mid, (el[0] + (wr[0] - el[0]) * 0.85, el[1] + (wr[1] - el[1]) * 0.85), 2.6, 2.3)
    L.paint(br.array() & m.array(), gold, outline=False)
    # fist
    fm = Mask()
    fm.ellipse(wr, 2.3, 2.1)
    L.paint(fm.array(), skin)
    return L


def sword_layer(P):
    """Sword held in the near fist, blade trailing back along the run."""
    L = Layer("sword")
    wr = P["near"]["arm"]["wrist"]
    ang = 128  # degrees from straight down, positive = forward ... 128 points up-back? use vector below
    # blade direction: back and slightly down, a 'carried at the hip' angle
    d = (-0.93, 0.36)
    grip0 = (wr[0] + 1.0, wr[1] - 0.5)
    guard = (wr[0] - 1.8, wr[1] + 0.6)
    tip = (guard[0] + d[0] * 20, guard[1] + d[1] * 20)
    pommel = (wr[0] + 3.4, wr[1] - 1.3)
    # blade
    mb = Mask()
    mb.capsule(guard, tip, 1.5, 0.55)
    L.paint(mb.array(), PAL["steel"], light=True, shade=True)
    # fuller highlight
    L.line((guard[0] + d[0] * 3, guard[1] + d[1] * 3 - 0.2), (tip[0] - d[0] * 3, tip[1] - d[1] * 3 - 0.2), PAL["steel"][2])
    # crossguard (perpendicular) + grip
    g = Mask()
    n = (-d[1], d[0])
    g.capsule((guard[0] + n[0] * 3, guard[1] + n[1] * 3), (guard[0] - n[0] * 3, guard[1] - n[1] * 3), 1.0)
    L.paint(g.array(), PAL["gold"])
    gr = Mask()
    gr.capsule(guard, pommel, 1.0)
    L.paint(gr.array(), PAL["leather"], light=False)
    pm = Mask()
    pm.ellipse(pommel, 1.2, 1.2)
    L.paint(pm.array(), PAL["gold"])
    return L


def torso_layer(P):
    L = Layer("torso")
    hip, sh = P["hip"], P["shoulder"]
    m = Mask()
    # tapered torso: broad shoulders, narrow waist
    m.capsule((hip[0] - 0.5, hip[1] - 1.0), (sh[0] - 0.5, sh[1] + 1.0), 4.4, 5.6)
    tm = m.array()
    L.paint(tm, PAL["top"])
    # bare shoulders / chest V
    chest = Mask()
    chest.poly([(sh[0] + 0.8, sh[1] - 2.2), (sh[0] + 4.6, sh[1] - 1.4), (sh[0] + 3.2, sh[1] + 2.4), (sh[0] - 0.5, sh[1] + 4.2)])
    cm = chest.array() & tm
    L.px[cm] = PAL["skin"][1]
    L.px[cm & ~shift(cm, 1, 1)] = PAL["skin"][2]
    # leather strap across the chest (shoulder -> opposite hip), gold studs
    a = (sh[0] + 2.0, sh[1] - 1.0)
    b = (hip[0] - 3.5, hip[1] - 2.0)
    sm = Mask()
    sm.capsule(a, b, 1.3)
    sm = sm.array() & tm
    L.px[sm] = PAL["leather"][1]
    L.px[sm & ~shift(sm, 1, 1)] = PAL["leather"][2]
    L.px[sm & ~shift(sm, -1, -1)] = PAL["leather"][0]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    L.dots([mid, (mid[0] + 1.5, mid[1] - 2.2)], PAL["gold"][2])
    # belt + gold buckle
    belt = Mask()
    belt.capsule((hip[0] - 4.2, hip[1] - 1.2), (hip[0] + 3.6, hip[1] - 1.2), 1.4)
    bm = belt.array() & dilate(tm)
    L.paint(bm, PAL["leather"], outline=False)
    L.dots([(hip[0] + 0.6, hip[1] - 1.2), (hip[0] + 1.6, hip[1] - 1.2)], PAL["gold"][2])
    # tattered hem
    L.dots([(hip[0] - 3, hip[1] + 1.4), (hip[0] + 1, hip[1] + 1.9)], PAL["top"][1])
    return L


def head_layer(P, f):
    L = Layer("head")
    c = P["head"]
    # hair mass behind the head, spiking back and up, ponytail trailing
    wob = [0, 1, 0, -1, 0, 1, 0, -1][f]
    back = Mask()
    back.ellipse((c[0] - 1.2, c[1] - 0.8), 6.0, 6.0)
    back.poly([(c[0] - 4, c[1] - 4), (c[0] - 11, c[1] - 6 + wob), (c[0] - 5, c[1] - 1)])
    back.poly([(c[0] - 4, c[1] - 1), (c[0] - 12, c[1] + 0 - wob), (c[0] - 5, c[1] + 3)])
    back.poly([(c[0] - 3, c[1] - 6), (c[0] - 6, c[1] - 11), (c[0] - 1, c[1] - 6.5)])
    back.poly([(c[0] + 0, c[1] - 6.5), (c[0] + 1, c[1] - 12), (c[0] + 4, c[1] - 6)])
    back.poly([(c[0] + 3, c[1] - 6), (c[0] + 7, c[1] - 9), (c[0] + 6, c[1] - 4)])
    # ponytail
    tail = Mask()
    tail.capsule((c[0] - 4, c[1] - 3), (c[0] - 10, c[1] + 2 + wob), 1.8, 1.0)
    tail.capsule((c[0] - 10, c[1] + 2 + wob), (c[0] - 14, c[1] + 5 + wob * 2), 1.0, 0.5)
    L.paint(back.array() | tail.array(), PAL["hair"])
    # face
    face = Mask()
    face.ellipse((c[0] + 1.2, c[1] + 0.8), 4.4, 4.9)
    face.poly([(c[0] + 2, c[1] + 3), (c[0] + 6, c[1] + 2.4), (c[0] + 4, c[1] + 6), (c[0], c[1] + 6)])  # jaw
    fm = face.array()
    L.paint(fm, PAL["skin"])
    # nose, eye, brow, mouth
    L.dots([(c[0] + 5.6, c[1] + 1.6), (c[0] + 5.6, c[1] + 2.6)], PAL["skin"][2])
    L.dots([(c[0] + 5.5, c[1] + 2.6)], PAL["skin"][0])
    L.dots([(c[0] + 3.2, c[1] + 0.2), (c[0] + 4.2, c[1] + 0.2)], PAL["eye"])
    L.dots([(c[0] + 4.2, c[1] + 0.2)], PAL["iris"])
    L.line((c[0] + 2.4, c[1] - 1.4), (c[0] + 5.0, c[1] - 0.9), PAL["outline"])
    L.dots([(c[0] + 4.4, c[1] + 4.4)], PAL["skin"][0])
    # front hair + spiky fringe over the forehead
    fr = Mask()
    fr.poly([(c[0] - 4.5, c[1] - 3.5), (c[0] + 5.2, c[1] - 3.2), (c[0] + 6.2, c[1] - 1.8), (c[0] + 3, c[1] - 2.6), (c[0] - 3, c[1] - 0.6)])
    L.paint(fr.array(), PAL["hair"], outline=False)
    # gold headband with trailing tail
    hb = Mask()
    hb.capsule((c[0] - 4.2, c[1] - 2.5), (c[0] + 5.4, c[1] - 2.7), 1.15)
    L.paint(hb.array(), PAL["gold"])
    bt = Mask()
    bt.poly([(c[0] - 4, c[1] - 3), (c[0] - 11, c[1] - 1.5 + wob), (c[0] - 10, c[1] + 1.2 + wob), (c[0] - 4, c[1] - 1.4)])
    L.paint(bt.array(), PAL["gold"], mid=PAL["gold"][1])
    # hoop earring
    L.dots([(c[0] - 0.8, c[1] + 3.0), (c[0] - 1.8, c[1] + 4.0), (c[0] - 0.8, c[1] + 5.0)], PAL["gold"][2])
    # neck
    nk = Mask()
    nk.capsule((c[0] - 0.5, c[1] + 4.0), (P["neck"][0] - 0.5, P["neck"][1] + 1.5), 2.0)
    return L, nk


def neck_layer(nk):
    L = Layer("neck")
    L.paint(nk.array(), PAL["skin"])
    return L


def cape_layer(P, f):
    """Red cape clasped at the neck, streaming back and rippling as he runs."""
    L = Layer("cape")
    sh = P["shoulder"]
    t = f / N * 2 * math.pi
    ra = (sh[0] - 0.5, sh[1] - 2.5)
    rb = (sh[0] - 2.5, sh[1] + 5.0)
    segs = 7
    up, lo = [], []
    for i in range(segs + 1):
        u = i / segs
        wave = math.sin(t - u * 3.4) * (0.6 + u * 3.6)
        x = ra[0] - 2.0 - u * 23
        y = ra[1] + 0.5 + u * 5.0 + wave
        up.append((x, y))
        width = 9.5 - u * 3.5
        tatter = 0.0
        if i >= segs - 2:
            tatter = (2.2 if (i + f) % 2 else -0.8)
        lo.append((x - u * 1.5, y + width + tatter))
    m = Mask()
    m.poly([ra] + up + lo[::-1] + [rb])
    tm = m.array()
    L.paint(tm, PAL["cape"])
    # fold shading: dark band under each ripple crest
    for i in (2, 4, 6):
        L.line((up[i][0], up[i][1] + 1.5), (lo[i][0] + 1, lo[i][1] - 1), PAL["cape"][0])
        L.line((up[i][0] + 1, up[i][1] + 1), (lo[i][0] + 2, lo[i][1] - 2.5), PAL["cape"][2])
    for i in range(segs):
        L.line((lo[i][0], lo[i][1] - 0.6), (lo[i + 1][0], lo[i + 1][1] - 0.6), PAL["gold"][1])
    # clasp
    L.dots([(ra[0] + 1, ra[1] + 1), (ra[0] + 2, ra[1] + 1)], PAL["gold"][2])
    return L


def shoulder_pauldron(P):
    L = Layer("pauldron")
    sh = P["shoulder"]
    m = Mask()
    m.ellipse((sh[0] - 0.5, sh[1] - 1.0), 3.4, 2.4)
    L.paint(m.array(), PAL["leather"])
    L.dots([(sh[0] - 1.0, sh[1] - 1.5), (sh[0] + 0.2, sh[1] - 1.5)], PAL["gold"][2])
    return L


def build_frame(f):
    P = solve(f)
    head, nk = head_layer(P, f)
    parts = [
        cape_layer(P, f),
        arm_layer("arm-far", P, True),
        leg_layer("leg-far", P, True),
        leg_layer("leg-near", P, False),
        torso_layer(P),
        neck_layer(nk),
        head,
        shoulder_pauldron(P),
        sword_layer(P),
        arm_layer("arm-near", P, False),
    ]
    # ground shadow stays out of the character layers; sword is last so it sits in the fist
    return parts


ORDER = ["cape", "arm-far", "leg-far", "leg-near", "torso", "neck", "head", "pauldron", "sword", "arm-near"]


def flatten(parts):
    out = np.zeros((H, W, 4), np.uint8)
    for p in parts:
        a = p.px[..., 3] > 0
        out[a] = p.px[a]
    return out


# ---------------------------------------------------------------- .aseprite writer
def chunk(t, data):
    return struct.pack("<IH", len(data) + 6, t) + data


def wstr(s):
    b = s.encode()
    return struct.pack("<H", len(b)) + b


def write_aseprite(path, frames_layers):
    """frames_layers[f][layer_index] -> RGBA array. Layers bottom to top."""
    names = ORDER
    palette_colors = []
    seen = {}
    for fr in frames_layers:
        for arr in fr:
            for px in arr.reshape(-1, 4):
                if px[3]:
                    t = tuple(int(v) for v in px)
                    if t not in seen and len(seen) < 255:
                        seen[t] = len(seen) + 1
    frames_bytes = []
    for fi, fr in enumerate(frames_layers):
        chunks = []
        if fi == 0:
            pal = struct.pack("<IIII8x", 256, 0, 255, 0)
            entries = [(0, 0, 0, 0)] + list(seen.keys())
            entries += [(0, 0, 0, 0)] * (256 - len(entries))
            for r, g, b, a in entries:
                pal += struct.pack("<HBBBB", 0, r, g, b, a)
            chunks.append(chunk(0x2019, pal))
            for n in names:
                # flags=visible|editable, type=normal, child level, default w/h, blend, opacity
                data = struct.pack("<HHHHHHB", 3, 0, 0, 0, 0, 0, 255) + b"\0\0\0" + wstr(n)
                chunks.append(chunk(0x2004, data))
            tag = struct.pack("<HHBH", 0, N - 1, 0, 0) + b"\0" * 6 + b"\0\0\0" + b"\0" + wstr("run")
            chunks.append(chunk(0x2018, struct.pack("<H8x", 1) + tag))
        for li, arr in enumerate(fr):
            ys, xs = np.nonzero(arr[..., 3])
            if len(xs) == 0:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            crop = arr[y0:y1, x0:x1]
            raw = crop.tobytes()
            cel = struct.pack("<HhhBHh", li, int(x0), int(y0), 255, 2, 0) + b"\0" * 5
            cel += struct.pack("<HH", x1 - x0, y1 - y0) + zlib.compress(raw)
            chunks.append(chunk(0x2005, cel))
        body = b"".join(chunks)
        hdr = struct.pack("<IHHH2xI", len(body) + 16, 0xF1FA, len(chunks), DUR[fi], 0)
        frames_bytes.append(hdr + body)
    total = 128 + sum(len(b) for b in frames_bytes)
    header = struct.pack("<IHHHHHIHII", total, 0xA5E0, N, W, H, 32, 1, 0, 0, 0)
    header += struct.pack("<B", 0) + b"\0" * 3 + struct.pack("<H", 0) + struct.pack("<BB", 1, 1)
    header += struct.pack("<hhHH", 0, 0, 16, 16) + b"\0" * 84
    assert len(header) == 128, len(header)
    with open(path, "wb") as fh:
        fh.write(header + b"".join(frames_bytes))


# ---------------------------------------------------------------- main
def main():
    frames = []
    layers_per_frame = []
    for f in range(N):
        parts = build_frame(f)
        by = {p.name: p.px for p in parts}
        layers_per_frame.append([by[n] for n in ORDER])
        frames.append(flatten(parts))

    # sprite sheet
    sheet = Image.new("RGBA", (W * N, H), (0, 0, 0, 0))
    for i, a in enumerate(frames):
        sheet.paste(Image.fromarray(a, "RGBA"), (i * W, 0))
    sheet.save(os.path.join(HERE, "warrior-run-sheet.png"))
    write_aseprite(os.path.join(HERE, "warrior-run.aseprite"), layers_per_frame)

    bg = (30, 26, 44, 255)
    # strip on dark ground at 4x
    strip = Image.new("RGBA", (W * N, H), bg)
    ImageDraw.Draw(strip).rectangle([0, GROUND - 1, W * N, H], fill=(54, 46, 70, 255))
    strip.alpha_composite(sheet)
    strip.resize((W * N * 2, H * 2), Image.NEAREST).save(os.path.join(HERE, "warrior-run-strip.png"))

    # gif + video frames (scrolling ground; scroll speed ~ planted-foot speed)
    VW, VH, Z = 160, 96, 6
    def scene(f, t):
        im = Image.new("RGBA", (VW, VH), bg)
        d = ImageDraw.Draw(im)
        for i in range(0, VW, 1):  # soft sky bands
            pass
        d.rectangle([0, 0, VW, 40], fill=(38, 32, 58, 255))
        d.rectangle([0, 40, VW, 62], fill=(46, 38, 66, 255))
        # distant pillars drifting slowly
        for k in range(-1, 8):
            x = (k * 28 - int(t * 0.35)) % (VW + 28) - 14
            d.rectangle([x, 20, x + 7, 62], fill=(60, 48, 82, 255))
            d.rectangle([x - 1, 18, x + 8, 20], fill=(70, 56, 94, 255))
        gy = 62 + (GROUND - 60)
        d.rectangle([0, gy, VW, VH], fill=(74, 56, 52, 255))
        d.line([0, gy, VW, gy], fill=(110, 84, 70, 255))
        for k in range(0, 14):  # ground ticks scroll back at foot speed
            x = (k * 14 - int(t * 4.8)) % (VW + 14) - 7
            d.line([x, gy + 5, x + 6, gy + 5], fill=(92, 70, 62, 255))
            d.line([x + 3, gy + 12, x + 8, gy + 12], fill=(60, 46, 46, 255))
        # shadow
        d.ellipse([VW // 2 - 11, gy - 1, VW // 2 + 9, gy + 3], fill=(40, 30, 34, 255))
        ox, oy = VW // 2 - 32 + 4, gy - 60 + 2
        im.alpha_composite(Image.fromarray(frames[f], "RGBA"), (ox, oy))
        return im

    loops = 8
    seq = []
    t = 0
    for _ in range(loops):
        for f in range(N):
            seq.append((f, t))
            t += DUR[f] / 1000 * 10
    # gif: character only on dark bg (no scroll, tiles perfectly)
    g = []
    for f in range(N):
        im = Image.new("RGBA", (W, H), bg)
        ImageDraw.Draw(im).rectangle([0, GROUND, W, H], fill=(54, 46, 70, 255))
        im.alpha_composite(Image.fromarray(frames[f], "RGBA"))
        g.append(im.convert("RGB").resize((W * 8, H * 8), Image.NEAREST))
    g[0].save(os.path.join(HERE, "warrior-run.gif"), save_all=True, append_images=g[1:], duration=DUR, loop=0, disposal=2)

    # mp4: constant 12.5 fps (80ms) per frame for ffmpeg, flight frames doubled
    tmp = os.path.join(HERE, ".frames")
    os.makedirs(tmp, exist_ok=True)
    n = 0
    t = 0.0
    for _ in range(loops):
        for f in range(N):
            im = scene(f, t).convert("RGB").resize((VW * Z, VH * Z), Image.NEAREST)
            im.save(os.path.join(tmp, f"f{n:04d}.png"))
            n += 1
            t += DUR[f] / 1000 * 10
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12.5", "-i", os.path.join(tmp, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                    os.path.join(HERE, "warrior-run.mp4")], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("ok")


if __name__ == "__main__":
    main()
