#!/usr/bin/env python3
"""Known-answer tests of hretime.py on synthetic lines.

A smooth, vertically coherent picture 710.2 samples wide sits on blanking (code 2); on time it starts at x=4.4. Each line's
timing drifts slowly down the field (an analog error, +-18 samples), so a late line shows its blanking at the left of the
720-sample window and an early one at the right; the window's own first 3 and last 3 samples read dark whatever the timing.
Some lines are dark at both ends and carry no reference of their own, including the first 20 lines of a field 30 samples
late.
Checks: one frame out per frame in; lines that show an edge come back within 1.5 samples (90%) with median error under 1;
lines with no reference stay within 10; no field comes out rougher than it went in; chroma moves with luma; a frame cut
short is refused."""
import os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TOOL = os.path.join(HERE, '..', 'hretime.py')
sys.path.insert(0, os.path.join(HERE, '..')); import hretime as H
rng = np.random.default_rng(11); W = 720
ON_TIME, WIDTH = 4.4, 710.2                  # the synthetic picture's own geometry (not read from the tool)

def picture(n_lines):
    """smooth 2-D texture: a line and its neighbours are alike, as in real picture"""
    t = rng.normal(0, 1, (n_lines + 20, 1200)); k = np.ones(9) / 9
    t = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, t)
    t = np.apply_along_axis(lambda c: np.convolve(c, np.ones(5) / 5, 'same'), 0, t)[10:-10]
    return np.clip(110 + t / t.std() * 40, 30, 200)

def line(src, late, dark_ends=False):
    """one 720-sample window: blanking (2) around a 710.2-sample picture starting at 4.4 + late"""
    x = np.arange(W) - (ON_TIME + late)                     # position within the picture
    y = np.full(W, 2.0); inside = (x >= 0) & (x <= WIDTH)
    y[inside] = np.interp(x[inside], np.arange(len(src)), src)
    edge = (x >= -1) & (x < 0); y[edge] = 2 + (y[np.clip(np.nonzero(edge)[0] + 1, 0, W - 1)] - 2) * (x[edge] + 1)
    if dark_ends:
        y[inside & (x < 40)] = 5; y[inside & (x > WIDTH - 40)] = 5
    y[:3] = 2; y[-3:] = 2
    return y

def make_field(seed_drift, dark_every=0, offset=0.0, dark_top=0):
    n = len(H.LINES); pic = picture(n)
    drift = np.cumsum(rng.normal(0, 0.25, n)); drift = 18 * drift / max(1e-6, np.abs(drift).max()) * seed_drift + offset
    rows = np.empty((240, W * 2), np.uint8); rows[:] = H.BLACK; truth = np.zeros(n); dark = np.zeros(n, bool)
    for i in range(n):
        de = bool(dark_every and i % dark_every == 0) or i < dark_top
        y = line(pic[i], drift[i], de)
        r = np.empty(W * 2, np.uint8); yy = np.clip(np.rint(y), 0, 255); r[1::2] = yy
        r[0::4] = np.clip(128 + (yy[0::2] - 110) / 3, 0, 255); r[2::4] = 100      # Cb follows the picture: it must move with it
        rows[H.LINES[i] - H.FIRST] = r; truth[i] = -drift[i]; dark[i] = de
    return rows, truth, dark

# the last field is 30 samples late throughout and its first 20 lines show no edge: they must follow the lines below, not
# start on 'no correction' and stay there because leaving it costs a jump
fields = [make_field(s, d, o, t) for s, d, o, t in ((1, 0, 0, 0), (-1, 0, 0, 0), (0.6, 7, 0, 0), (-0.8, 5, 0, 0),
                                                    (0.3, 0, 0, 0), (0.3, 0, 30, 20))]
frames = b''.join(np.stack([fields[i][0], fields[i + 1][0]], 1).reshape(480, W * 2).tobytes() for i in range(0, len(fields), 2))
r = subprocess.run([sys.executable, TOOL, '--log', '/dev/stderr'], input=frames, capture_output=True)
assert r.returncode == 0, r.stderr.decode()[-500:]
out = np.frombuffer(r.stdout, np.uint8).reshape(-1, 480, W * 2)
assert out.shape[0] * 2 == len(fields)
got = {}
for ln in r.stderr.decode().splitlines():
    p = ln.split(',')
    if len(p) == 4 and p[0].isdigit(): got[(int(p[0]), int(p[1]))] = float(p[2])
err_seen = []; err_dark = []
for g, (rows, truth, dark) in enumerate(fields):
    c = np.array([got[(g, l)] for l in H.LINES])
    e = np.abs(c - truth)
    err_seen += list(e[~dark]); err_dark += list(e[dark])
    o = out[g // 2][g % 2::2]
    yin = rows[H.LINES - H.FIRST, 1::2].astype(float); yout = o[H.LINES - H.FIRST, 1::2].astype(float)
    rin = np.abs(np.diff(yin[:, 60:660], axis=0)).mean(); rout = np.abs(np.diff(yout[:, 60:660], axis=0)).mean()
    assert rout <= rin + 0.05, 'field %d came out rougher (%.2f -> %.2f)' % (g, rin, rout)
    # chroma moved with luma: Cb still follows the luma pattern after retiming
    cb = o[H.LINES - H.FIRST, 0::4][:, 40:320].astype(float); yl = o[H.LINES - H.FIRST, 1::2][:, 80:640:2].astype(float)
    mis = np.abs(cb - np.clip(128 + (yl - 110) / 3, 0, 255)).mean()
    assert mis < 3, 'chroma did not move with luma (mean mismatch %.1f)' % mis
err_seen = np.array(err_seen); err_dark = np.array(err_dark)
print('lines with references: |error| p50 %.2f, %.0f%% within 1.5 | dark-ended lines: max |error| %.2f'
      % (np.median(err_seen), 100 * np.mean(err_seen <= 1.5), err_dark.max() if len(err_dark) else 0))
assert np.median(err_seen) < 1.0, np.median(err_seen)
assert np.mean(err_seen <= 1.5) >= 0.9, np.mean(err_seen <= 1.5)
assert len(err_dark) and err_dark.max() < 10, err_dark.max()

r = subprocess.run([sys.executable, TOOL], input=frames[:H.FRAME] + b'\0' * 999, capture_output=True)
assert r.returncode != 0 and b'inside a frame' in r.stderr, r.stderr
print('ALL OK')
