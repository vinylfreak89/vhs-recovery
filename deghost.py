#!/usr/bin/env python3
"""deghost.py --measure FILE < frames | deghost.py --echo FILE < frames > frames -- remove the echo carried ahead of luma edges.

Input and output as vstab.py: 720x480 UYVY frames, two fields woven, each field's rows its synchroniser slots 20..259.
Luma of lines 22..257 only (as hretime.py: 20-21 are the Shuttle's own inserts, clean device-made edges that would swamp
the measurement; 258-259 are blanked); chroma and the other lines pass through unchanged.

A tape recorded off a weak antenna carries a multipath ghost: every luma edge drags a pulse ahead of it that returns to
the picture's level before the edge, the embossed look. It is measured from the tape itself:
--measure finds sharp picture edges with flat picture 14-24 samples either side (so an echo between them is kept, not
selected against), takes the median of their profiles normalised to the step (0 before, 1 after), and records how far the
profile departs from a clean edge of the same steepness in the 16 samples before the edge. Rising and falling edges are
measured separately: on the test tape a rising edge carries twice the echo of a falling one, which no single linear
filter can undo.
--echo subtracts, for every large edge, the echo of its direction scaled by its size, and for all finer detail the mean
of the two, linearly; solved against the corrected line (ITERS passes). It refuses an echo file without enough edges.

Measured on an off-air test tape (README.md): the pulse peaks 8 samples (0.59 us) ahead of the edge at 0.25 of a rising
step, 0.15 of a falling one; after subtraction 0.03 and 0.02, edge position unchanged, noise in flat picture 8.26 -> 6.46
codes at an unchanged mean level. Left over: the echo also depends on picture level (falls into near-black carry less than
falls into grey), which this does not model."""
import argparse, sys
import numpy as np

W = 720; ROWS = 240; FRAME = W * 2 * ROWS * 2; FIRST = 20
LINES = np.arange(22, 258)          # lines measured and corrected
K = 30                              # profile half-width, samples
FLAT = (14, 24)                     # plateau each side of an edge, samples from it (std below FLAT_SD)
FLAT_SD = 4.0
MIN_STEP = 40.0                     # an edge's plateaus differ by at least this (codes)
BLANK_MAX = 9.0                     # blanking reads below this (as hretime.py)
AHEAD = (-16, -2)                   # where the echo is taken from, relative to the edge
MIN_EDGES = 200                     # edges of each direction an echo file needs
ITERS = 3                           # passes solving the subtraction against the corrected line
REFINE = 3                          # passes refining the measured echo against its own result

def smooth(Y):
    return np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, 'same'), 1, Y.astype(np.float32))

def find_edges(Y):
    """Y's sharp, isolated edges: (row, centre, half-height crossing, 'rise'/'fall'), found on a 3-sample smoothing"""
    out = []
    for i, y in enumerate(smooth(Y)):
        d4 = y[4:] - y[:-4]
        for x in np.nonzero(np.abs(d4) > MIN_STEP)[0][::3]:
            c = x + 2
            if c < K + 10 or c > W - K - 10: continue
            lo = y[c - FLAT[1]:c - FLAT[0]]; hi = y[c + FLAT[0]:c + FLAT[1]]
            if lo.std() > FLAT_SD or hi.std() > FLAT_SD or abs(hi.mean() - lo.mean()) < MIN_STEP: continue
            a, b = lo.mean(), hi.mean(); mid = (a + b) / 2; s = np.sign(b - a); seg = y[c - 6:c + 7]
            k = np.nonzero(s * (seg - mid) >= 0)[0]
            if not len(k) or k[0] == 0: continue
            j = c - 6 + k[0]; out.append((i, c, j - 1 + (mid - y[j - 1]) / (y[j] - y[j - 1]), 'rise' if b > a else 'fall'))
    return out

def profile(y, c, x0):
    """line y around an edge, normalised to its plateaus (0 before, 1 after), read from the line itself: smoothing would
    flatten the echo being measured"""
    a = y[c - FLAT[1]:c - FLAT[0]].mean(); b = y[c + FLAT[0]:c + FLAT[1]].mean()
    return (np.interp(x0 + np.arange(-K, K + 1), np.arange(W), y) - a) / (b - a)

def edge_profiles(Y, acc):
    """add the normalised profiles of Y's sharp, isolated edges to acc['rise'] / acc['fall']"""
    Y = Y.astype(np.float32)
    for i, c, x0, kind in find_edges(Y): acc[kind].append(profile(Y[i], c, x0))

def echo_shape(prof):
    """median edge profile -> step-domain echo at offsets AHEAD (profile minus a clean logistic edge as steep as it)"""
    off = np.arange(-K, K + 1); p1 = float(np.clip(prof[K + 1], 0.55, 0.99))
    clean = 1 / (1 + np.exp(-np.log(p1 / (1 - p1)) * off))
    return (prof - clean)[K + AHEAD[0]:K + AHEAD[1] + 1]

def echo(X, Er, Ef):
    """the echo ahead of every step of X (rows). The two directions were measured on large edges between plateaus FLAT
    samples away on each side, and only such an edge gets its direction's echo (its 4-sample steep part over MIN_STEP and
    its plateaus differing by over MIN_STEP). All else, spikes, noise and thin lines included, gets the mean of the two,
    linearly, so a rise and the fall right after it cancel instead of leaving a wash ahead of every grain of noise"""
    d = np.diff(X, axis=1, prepend=X[:, :1]); n = X.shape[1]
    cs = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X, axis=1)], axis=1)
    x = np.arange(FLAT[1], n - FLAT[1] - 1); w = FLAT[1] - FLAT[0]           # plateau after minus plateau before
    jump = (cs[:, x + FLAT[1]] - cs[:, x + FLAT[0]]) / w - (cs[:, x - FLAT[0]] - cs[:, x - FLAT[1]]) / w
    steep = X[:, x + 2] - X[:, x - 2]
    c = (np.abs(jump) > MIN_STEP) & (np.abs(steep) > MIN_STEP) & (np.sign(jump) == np.sign(steep))
    big = np.zeros_like(X, dtype=bool)
    for k in range(-2, 3): big[:, x + k] |= c                                 # the edge's own samples
    dl = np.where(big, d, 0.0); ds = d - dl; El = (Er + Ef) / 2; P = 20
    up = np.pad(np.maximum(dl, 0), ((0, 0), (0, P))); dn = np.pad(np.minimum(dl, 0), ((0, 0), (0, P)))
    sm = np.pad(ds, ((0, 0), (0, P))); out = np.zeros_like(X)
    for j, tau in enumerate(range(AHEAD[0], AHEAD[1] + 1)):                    # a step at x - tau lies ahead of x
        out += Er[j] * up[:, -tau:-tau + n] + Ef[j] * dn[:, -tau:-tau + n] + El[j] * sm[:, -tau:-tau + n]
    return out

def blanking(Y):
    """per row, the samples to leave alone: the window's edge transients and the blanking runs reaching in from either end
    (3-sample level under BLANK_MAX). On the test tape blanking is quiet (std 0.5) with no trace of the echo, wiped
    upstream; subtracting an echo there would write a copy of the picture's detail into it"""
    r = smooth(Y); keep = np.zeros(Y.shape, bool); keep[:, :3] = True; keep[:, -3:] = True
    low = r < BLANK_MAX
    left = np.where(low[:, 1:].all(1), W, np.argmin(low[:, 1:], 1) + 1)              # first non-blanking sample from x=1
    right = np.where(low[:, :-1].all(1), 0, W - 1 - np.argmin(low[:, :-1][:, ::-1], 1))   # last non-blanking sample to x=718
    x = np.arange(W)
    keep |= (x[None, :] < left[:, None]) & low[:, 1:2]                                 # only when the run starts at x=1
    keep |= (x[None, :] > right[:, None]) & low[:, -2:-1]
    return keep

def deghost_luma(Y, Er, Ef):
    Y = Y.astype(np.float32); X = Y.copy(); keep = blanking(Y)
    for _ in range(ITERS): X = np.where(keep, Y, Y - echo(X, Er, Ef))
    return X

def frames(inp):
    n = 0
    while True:
        b = inp.read(FRAME)
        if not b: return
        if len(b) != FRAME: sys.exit('deghost: input ended inside a frame (%d of %d bytes) after %d frames' % (len(b), FRAME, n))
        n += 1; yield np.frombuffer(b, np.uint8).reshape(ROWS * 2, W * 2)

def measure(path, every):
    lines = []; recs = []; n = 0
    for n, fr in enumerate(frames(sys.stdin.buffer), 1):
        if (n - 1) % every: continue
        for p in (0, 1):
            Y = fr[p::2][LINES - FIRST, 1::2].astype(np.float32); found = find_edges(Y); rows = {}
            for i, c, x0, kind in found:
                if i not in rows: rows[i] = len(lines); lines.append(Y[i])
                recs.append((rows[i], c, x0, kind))
    counts = {k: sum(1 for r in recs if r[3] == k) for k in ('rise', 'fall')}
    if min(counts.values()) < MIN_EDGES:
        sys.exit('deghost: only %d rising / %d falling edges in %d frames (need %d each); no echo file written'
                 % (counts['rise'], counts['fall'], n, MIN_EDGES))
    L = np.array(lines); E = {'rise': np.zeros(AHEAD[1] - AHEAD[0] + 1), 'fall': np.zeros(AHEAD[1] - AHEAD[0] + 1)}
    for it in range(REFINE + 1):      # the echo left after subtracting the estimate so far is added to it: no model of how
        X = deghost_luma(L, E['rise'], E['fall']) if it else L               # an edge's width smears the subtraction
        prof = {k: np.median([profile(X[i], c, x0) for i, c, x0, kk in recs if kk == k], 0) for k in E}
        if it == 0: first = prof
        for k in E: E[k] = E[k] + echo_shape(prof[k])
        left = {k: float(np.abs(echo_shape(prof[k])).max()) for k in E}
    np.savez(path, Er=E['rise'], Ef=E['fall'], rise=first['rise'], fall=first['fall'], n_rise=counts['rise'],
             n_fall=counts['fall'])
    for k, name in (('rise', 'rising'), ('fall', 'falling')):
        e0 = echo_shape(first[k]); i = int(np.argmax(np.abs(e0)))
        print('deghost: %d %s edges; echo peaks at %+.3f of the step, %d samples ahead; left after subtraction %.3f'
              % (counts[k], name, e0[i], -(AHEAD[0] + i), left[k]), file=sys.stderr)

def apply(path):
    e = np.load(path)
    if min(int(e['n_rise']), int(e['n_fall'])) < MIN_EDGES: sys.exit('deghost: %s rests on too few edges' % path)
    Er, Ef = e['Er'], e['Ef']; out = sys.stdout.buffer; n = 0
    for n, fr in enumerate(frames(sys.stdin.buffer), 1):
        o = fr.copy()
        for p in (0, 1):
            f = o[p::2]; f[LINES - FIRST, 1::2] = np.clip(np.rint(deghost_luma(fr[p::2][LINES - FIRST, 1::2], Er, Ef)), 0, 255)
            o[p::2] = f
        out.write(o.tobytes())
    out.flush(); print('deghost: %d frames' % n, file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--measure', metavar='FILE'); g.add_argument('--echo', metavar='FILE')
    ap.add_argument('--every', type=int, default=3, help='measure every Nth frame (default 3)')
    a = ap.parse_args()
    if a.measure: measure(a.measure, a.every)
    else: apply(a.echo)

if __name__ == '__main__': main()
