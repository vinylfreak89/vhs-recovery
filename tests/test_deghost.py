#!/usr/bin/env python3
"""Known-answer tests of deghost.py on synthetic fields.

Each picture line is flat runs between smooth edges (as steep as the test tape's), and every edge drags a pulse ahead of
it, peaking 8 samples early: 0.25 of a rising step, 0.12 of a falling one (the test tape's ghost, built here from the
test's own formula, not the tool's). Lines 20-21 carry a clean square wave with no echo, as the Shuttle's inserts do.
Checks: the measured echo (the edges as found, before any subtraction) peaks where and as high as it was made, unswayed
by the insert lines; the filtered picture's
pulse (departure from a clean edge ahead of it) is under 0.05 of a step either way while edges stay put; spikes and thin
lines leave no wash ahead of them (their rise and fall echoes cancel); blanking at a line's end is left as it was; chroma and lines 20-21 / 258-259 pass through unchanged;
an echo file resting on too few edges, a measurement whose only edges are on the insert lines, and a truncated frame
are refused."""
import os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TOOL = os.path.join(HERE, '..', 'deghost.py')
sys.path.insert(0, os.path.join(HERE, '..')); import deghost as D
rng = np.random.default_rng(5); W = 720; SLOPE = 0.9; PEAK, WIDTH = -8, 2.5; GAIN = {1: 0.25, -1: 0.12}

def picture_line(steps):
    """flat runs between logistic edges at the given (position, new level) steps, starting at level 40"""
    x = np.arange(W, dtype=float); y = np.full(W, 40.0); level = 40.0
    for pos, new in steps:
        y += (new - level) / (1 + np.exp(-SLOPE * (x - pos))); level = new
    return y

def ghosted(steps):
    y = picture_line(steps); x = np.arange(W, dtype=float); level = 40.0
    for pos, new in steps:
        d = new - level; y += GAIN[int(np.sign(d))] * d * np.exp(-((x - pos - PEAK) / WIDTH) ** 2); level = new
    return y

def field(seed):
    r = np.random.default_rng(seed); rows = np.empty((240, W * 2), np.uint8); rows[:] = D.np.tile([128, 16], W)
    base = np.sort(r.choice(np.arange(60, 660, 40), 9, replace=False)) + r.uniform(0, 8, 9)
    levels = r.uniform(30, 200, 9)
    for s in range(240):
        line = s + 20
        if line in (20, 21): y = np.where((np.arange(W) // 24) % 2, 120.0, 10.0)        # clean insert: no echo
        elif line >= 258: y = np.full(W, 2.0)
        else: y = ghosted([(p + 0.05 * s, v) for p, v in zip(base, levels)])
        y = y + r.normal(0, 1.5, W) * (line >= 22)
        rows[s, 1::2] = np.clip(np.rint(y), 0, 255)
        rows[s, 0::4] = r.integers(60, 200, W // 2); rows[s, 2::4] = r.integers(60, 200, W // 2)
    return rows

fields = [field(i) for i in range(40)]
frames = b''.join(np.stack([fields[i], fields[i + 1]], 1).reshape(480, W * 2).tobytes() for i in range(0, len(fields), 2))
tmp = tempfile.mkdtemp(); echo = os.path.join(tmp, 'echo.npz')
r = subprocess.run([sys.executable, TOOL, '--measure', echo, '--every', '1'], input=frames, capture_output=True)
assert r.returncode == 0, r.stderr.decode()[-600:]
e = np.load(echo); Er, Ef = D.echo_shape(e['rise']), D.echo_shape(e['fall']); ahead = np.arange(D.AHEAD[0], D.AHEAD[1] + 1)
print(r.stderr.decode().strip())
print('measured: rising %.3f at %d, falling %.3f at %d (made: 0.25 / 0.12 at %d)'
      % (Er.max(), ahead[Er.argmax()], Ef.max(), ahead[Ef.argmax()], PEAK))
assert abs(Er.max() - 0.25) < 0.03 and abs(ahead[Er.argmax()] - PEAK) <= 1, (Er.max(), ahead[Er.argmax()])
assert abs(Ef.max() - 0.12) < 0.03 and abs(ahead[Ef.argmax()] - PEAK) <= 1, (Ef.max(), ahead[Ef.argmax()])

r = subprocess.run([sys.executable, TOOL, '--echo', echo], input=frames, capture_output=True)
assert r.returncode == 0, r.stderr.decode()[-600:]
out = np.frombuffer(r.stdout, np.uint8).reshape(-1, 480, W * 2); assert out.shape[0] * 2 == len(fields)
inp = np.frombuffer(frames, np.uint8).reshape(-1, 480, W * 2)
assert (out[:, :, 0::2] == inp[:, :, 0::2]).all(), 'chroma changed'
for p in (0, 1):
    for s in (0, 1, 238, 239): assert (out[:, p::2][:, s] == inp[:, p::2][:, s]).all(), 'line %d changed' % (s + 20)
acc = {'rise': [], 'fall': []}; acc_in = {'rise': [], 'fall': []}
for fr, fi in zip(out, inp):
    for p in (0, 1):
        D.edge_profiles(fr[p::2][D.LINES - D.FIRST, 1::2], acc); D.edge_profiles(fi[p::2][D.LINES - D.FIRST, 1::2], acc_in)
for k in ('rise', 'fall'):
    m = np.median(np.array(acc[k]), 0); m_in = np.median(np.array(acc_in[k]), 0)
    pulse = np.abs(D.echo_shape(m)).max(); pulse_in = np.abs(D.echo_shape(m_in)).max()   # departure from a clean edge
    print('%s edges: pulse ahead %.3f -> %.3f of the step; half height at the edge %.2f -> %.2f'
          % (k, pulse_in, pulse, m_in[D.K], m[D.K]))
    assert pulse < 0.05, (k, pulse)
    assert abs(m[D.K] - 0.5) < 0.05, (k, m[D.K])                    # edge position kept (profiles centred on the input's edge)

# fine detail: a spike or a thin line is a rise and a fall together; their echoes must cancel, not leave a wash ahead
for amp, w in ((40, 1), (-40, 1), (80, 1), (100, 3), (-60, 6)):
    y = np.full((1, W), 80.0); y[0, 360:360 + w] += amp; dv = D.deghost_luma(y, e['Er'], e['Ef'])[0] - 80
    wash = dv[330:360].mean()
    assert abs(wash) < 1.5, ('wash ahead of a %d-sample line of %+d' % (w, amp), wash)
print('no wash ahead of spikes and thin lines')

# blanking reaching in from the window's start, ahead of a bright edge with detail: left as it was (it carries no echo)
y = np.full((1, W), 1.5) + rng.normal(0, 0.5, (1, W)); y[0, 50:] = 150 + rng.normal(0, 8, W - 50)
assert np.array_equal(D.deghost_luma(y, e['Er'], e['Ef'])[0, :48], y[0, :48].astype(np.float32)), 'blanking changed'
print('blanking ahead of an edge unchanged')

thin = os.path.join(tmp, 'thin.npz'); np.savez(thin, Er=e['Er'], Ef=e['Ef'], n_rise=10, n_fall=10)
r = subprocess.run([sys.executable, TOOL, '--echo', thin], input=frames[:D.FRAME], capture_output=True)
assert r.returncode != 0 and b'too few edges' in r.stderr, r.stderr
ins = np.tile(np.array([128, 100], np.uint8), (240, W))                                   # flat picture...
ins[0:2, 1::2] = np.where((np.arange(W) // 32) % 2, 120, 10)                              # ...edges only on lines 20-21
ins_frames = np.stack([ins, ins], 1).reshape(480, W * 2).tobytes() * 40
r = subprocess.run([sys.executable, TOOL, '--measure', thin, '--every', '1'], input=ins_frames, capture_output=True)
assert r.returncode != 0 and b'no echo file written' in r.stderr, r.stderr
r = subprocess.run([sys.executable, TOOL, '--echo', echo], input=frames[:D.FRAME] + b'\0' * 999, capture_output=True)
assert r.returncode != 0 and b'inside a frame' in r.stderr, r.stderr
print('ALL OK')
