#!/usr/bin/env python3
"""hretime.py [--log FILE] < frames > frames -- horizontal timing of each line from its own blanking.

Input and output as vstab.py: 720x480 UYVY frames, two fields woven (field 1 on even rows), each field's rows its
synchroniser slots 20..259 (frameserver_replay with FS_REGISTRATION_OFF=1 FS_FIRST_LINE=20,282). One frame out per frame
in; each field is handled on its own.

The tape signal is one waveform in time; the Shuttle keeps 720 of every 858 samples of it, a window placed where the
synchroniser expects each line. A line whose timing has slipped shows a piece of its horizontal blanking inside that window:
a run at blanking level from x=0 that ends where its picture starts (a late line), or a run from where its picture ends to
x=719 (an early line). Blanking reads far below the darkest picture (black with setup), so it is found by level. Those two
edges are the reference points; a line may show one, both or neither:
- the first 2-3 and last 3 samples of every window read dark whatever the timing (the Shuttle's edge transient), so an edge
  there is only a bound: a line with neither edge clear of the transients but bright picture at both ends sits within about
  a sample of its place;
- a line dark at an end has no reference of its own.
Each line gets candidate corrections from the signal only: its own start edge, its own end edge, the bound above, the median
of the references within NEAR lines, and no correction. A best path down the field picks one per line, charging a candidate
WEIGHT x (1 - its confidence) plus the roughness it leaves against the line above (mean |difference| per sample). Picture is
used only to choose among those candidates, never to make one. A line moves sideways only; anything moved in from outside
the window is black.

Measured on an off-air test tape (TBC-off pass scored against a TBC-on pass of the same stretch, README.md): picture start
at x=4.4 and end at x=714.6 when on time; |error| p50 1.18, p90 5.23 samples (uncorrected 3.85 / 16.1); no field rougher
than the original. Known fault: whole blocks of lines can still be offset where the picture changes brightness sharply or
the blanking is noisy, as the roughness cost then cannot tell the candidates apart."""
import argparse, sys
import numpy as np

W = 720; ROWS = 240; FRAME = W * 2 * ROWS * 2; FIRST = 20
LINES = np.arange(22, 258)          # slots retimed (20-21 are the Shuttle's inserts, 258-262 blanked by the synchroniser)
BLANK_MAX = 9.0                     # a blanking sample reads below this (tape blanking ~1-2; picture black with setup ~16+)
PIC_MIN = 14.0                      # picture beside an edge reaches this within a few samples
TRANSIENT = (5.5, 713.5)            # an edge at or beyond these is the window's own transient: a bound, not a reference
START, END = 4.4, 714.6             # where a line's picture starts / ends when on time (measured)
NEAR = 8                            # lines either side whose references give the 'neighbours' candidate
WEIGHT = 1.0                        # confidence cost, in roughness units (0.1-1.0 measured equal; higher was worse)
COLS = slice(60, 660)               # columns the roughness is measured over
BLACK = np.tile(np.array([128, 16], np.uint8), W)

def refs(y):
    """one line's luma (720) -> (start edge, confidence, end edge, confidence); NaN where none is found"""
    r = np.convolve(y.astype(np.float32), np.ones(3) / 3, 'same'); r[0], r[-1] = y[0], y[-1]
    out = [np.nan, 0.0, np.nan, 0.0]
    if r[1] < BLANK_MAX:
        k = 1
        while k < 200 and r[k] < BLANK_MAX: k += 1
        if k < 200 and r[k + 2:k + 10].mean() > PIC_MIN:
            pic = r[k + 2:k + 10].mean(); lvl = r[max(0, k - 4):k].mean(); mid = lvl + 0.5 * (pic - lvl); j = k
            while j < k + 8 and r[j] < mid: j += 1
            a, b = r[j - 1], r[j]; out[0] = j - 1 + (mid - a) / (b - a) if b != a else float(j - 1)
            flat = r[1:k].std() if k > 2 else 3.0
            out[1] = float(np.clip((pic - lvl) / 40, 0, 1) * np.clip(1 - flat / 6, 0, 1))
    if r[-2] < BLANK_MAX:
        k = 718
        while k > 520 and r[k] < BLANK_MAX: k -= 1
        if k > 520 and r[k - 9:k - 1].mean() > PIC_MIN:
            pic = r[k - 9:k - 1].mean(); lvl = r[k + 1:k + 5].mean(); mid = lvl + 0.5 * (pic - lvl); j = k
            while j > k - 8 and r[j] < mid: j -= 1
            a, b = r[j], r[j + 1]; out[2] = j + (mid - a) / (b - a) if b != a else float(j)
            flat = r[k + 1:719].std() if k < 717 else 3.0
            out[3] = float(np.clip((pic - lvl) / 40, 0, 1) * np.clip(1 - flat / 6, 0, 1))
    return out

def own_candidates(y, ref):
    """(correction, cost-weighting 0..1) from the line itself"""
    s, cs, e, ce = ref; c = []
    if not np.isnan(s) and s > TRANSIENT[0]: c.append((START - s, 1 - cs, 'start'))
    if not np.isnan(e) and e < TRANSIENT[1]: c.append((END - e, 1 - ce, 'end'))
    hidden = (np.isnan(s) or s <= TRANSIENT[0]) and (np.isnan(e) or e >= TRANSIENT[1])
    if hidden and y[6:30].min() > PIC_MIN and y[690:713].min() > PIC_MIN: c.append((0.0, 0.3, 'pinned'))
    return c

def shifted(y, c, fill=np.nan):
    """y moved right by c samples, linear interpolation; outside the window: fill"""
    x = np.arange(len(y)) - c; i0 = np.floor(x).astype(int); f = x - i0; ok = (i0 >= 0) & (i0 < len(y) - 1)
    o = np.full(len(y), fill, np.float32); o[ok] = y[i0[ok]] * (1 - f[ok]) + y[i0[ok] + 1] * f[ok]; return o

def roughness(a, b):
    d = np.abs(a[COLS] - b[COLS]); n = np.isfinite(d).sum()
    return float(np.nanmean(d)) if n > 200 else 20.0

def choose(Y):
    """Y: luma of LINES (len(LINES) x 720) -> per-line correction and its source"""
    own = [own_candidates(Y[i], refs(Y[i])) for i in range(len(Y))]
    cand = []
    for i in range(len(Y)):
        near = [v for j in range(max(0, i - NEAR), min(len(Y), i + NEAR + 1)) for v, _, _ in own[j]]
        c = list(own[i])
        if near: c.append((float(np.median(near)), 0.6, 'neighbours'))
        c.append((0.0, 0.9, 'none'))
        merged = []
        for v, k, src in sorted(c):
            if merged and abs(v - merged[-1][0]) < 0.75:
                if k < merged[-1][1]: merged[-1] = (merged[-1][0], k, src)
            else: merged.append((v, k, src))
        cand.append(merged)
    rows = [[shifted(Y[i], v) for v, _, _ in cand[i]] for i in range(len(Y))]
    cost = np.array([WEIGHT * k for _, k, _ in cand[0]]); back = [None]
    for i in range(1, len(Y)):
        cur = np.empty(len(cand[i])); bp = np.empty(len(cand[i]), int)
        for a in range(len(cand[i])):
            t = [cost[b] + roughness(rows[i][a], rows[i - 1][b]) for b in range(len(cand[i - 1]))]
            bp[a] = int(np.argmin(t)); cur[a] = t[bp[a]] + WEIGHT * cand[i][a][1]
        cost = cur; back.append(bp)
    k = int(np.argmin(cost)); out = [None] * len(Y)
    for i in range(len(Y) - 1, -1, -1):
        out[i] = (cand[i][k][0], cand[i][k][2])
        if i > 0: k = back[i][k]
    return out

def retime_row(row, c):
    """one UYVY row moved right by c samples (chroma by c/2 of its own samples); black outside the window"""
    y = shifted(row[1::2].astype(np.float32), c, 16); u = shifted(row[0::4].astype(np.float32), c / 2, 128)
    v = shifted(row[2::4].astype(np.float32), c / 2, 128)
    out = np.empty_like(row); out[1::2] = np.clip(np.rint(y), 0, 255); out[0::4] = np.clip(np.rint(u), 0, 255)
    out[2::4] = np.clip(np.rint(v), 0, 255); return out

def field(f, n, log=None):
    """f: 240 UYVY rows (slots 20..259) -> retimed rows"""
    Y = f[LINES - FIRST, 1::2].astype(np.float32); picks = choose(Y)
    out = f.copy()
    for i, (c, src) in enumerate(picks):
        if abs(c) > 1e-3: out[LINES[i] - FIRST] = retime_row(f[LINES[i] - FIRST], c)
        if log: log.write('%d,%d,%.2f,%s\n' % (n, LINES[i], c, src))
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--log')
    a = ap.parse_args()
    log = open(a.log, 'w') if a.log else None
    if log: log.write('field,line,correction,source\n')
    inp = sys.stdin.buffer; out = sys.stdout.buffer; n = 0
    while True:
        b = inp.read(FRAME)
        if not b: break
        if len(b) != FRAME: sys.exit('hretime: input ended inside a frame (%d of %d bytes) after %d frames' % (len(b), FRAME, n))
        fr = np.frombuffer(b, np.uint8).reshape(ROWS * 2, W * 2); o = np.empty_like(fr)
        o[0::2] = field(fr[0::2], 2 * n, log); o[1::2] = field(fr[1::2], 2 * n + 1, log)
        out.write(o.tobytes()); n += 1
    out.flush()
    if log: log.close()
    print('hretime: %d frames' % n, file=sys.stderr)

if __name__ == '__main__': main()
