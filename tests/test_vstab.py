#!/usr/bin/env python3
"""Known-answer tests of vstab.py on a synthetic tape through a simulated frame synchroniser.

The tape: fields of 262 and 263 lines; lines 1-15 and 17-21 at blanking level, line 16 the marker, picture from 22: a texture
per scene and line that moves sideways field by field, with a flat band (lines 100-139) where a row matches the previous
field within noise whether or not it was repeated. Every line carries its own line number in the chroma of its first four
pixel pairs, so each output row says which tape line it is. The synchroniser, as measured (see README.md): locked, one tape
field per pass, tape line n in slot n; unlocked, it keeps 262/263 lines per pass and drops 27/28, repeating a whole field
every 10/11 passes (re-read with fresh noise, as an analog re-read gives) and the lower part of one once. Slots 20-21 carry
the Shuttle's inserts (changing field to field), 258-262 are blanked; slots 20..259 are published.
Checks: one frame out per frame in, also when the input arrives in pieces; every output row is either black or the tape line
of its own number, exactly where the marker is visible, within a line where it is hidden (a pass moves 27.5 lines; the half
line is not observable) and for rows that wrap below the blanking (the tape's previous field, whose parity is not known);
picture in the flat band is never blacked out in a fresh field; the Shuttle's inserts never move; a repeat repeats the
output; a scene cut while rolling and a relock are each read correctly; a relock after a roll of any length from 1 to 39
passes ends exactly placed; a dark band in a still picture that looks like blanking is not followed; a frame cut short is
refused."""
import os, subprocess, sys, threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); VSTAB = os.path.join(HERE, '..', 'vstab.py')
sys.path.insert(0, os.path.join(HERE, '..')); import vstab
W = 720; rng = np.random.default_rng(7)
def smooth_texture(mean):
    """picture-like texture: random detail smoothed over ~8 samples, so a sideways move of 2 still matches itself"""
    t = rng.normal(0, 1, (264, W + 400)); k = np.ones(8) / 8
    t = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, t)
    return np.clip(mean + t / t.std() * 30, 30, 230).astype(np.uint8)
TEX = {0: smooth_texture(90), 1: smooth_texture(170)}          # the scene cut: a different, brighter picture
DEVICE = 200                                      # chroma label of the Shuttle's inserts
BLANK = np.tile(np.array([128, 1], np.uint8), W)

def tape_line(sc, n, k, dark_band=False, static=False):
    """UYVY row of tape line n (1-based) in scene sc, tape field k"""
    if n < 22: y = np.full(W, 6, np.uint8)
    elif 100 <= n < 140: y = np.full(W, 120, np.uint8)
    elif dark_band and 180 <= n < 194: y = np.full(W, 7, np.uint8)
    elif dark_band and n == 194: y = np.full(W, 160, np.uint8)
    else: o = 0 if static and n >= 150 else 2 * (k % 200); y = TEX[sc][n, o:o + W].copy()   # moves 2 samples a field (static: below line 150 holds still)
    if n == 16: y[:] = 100; y[15:115] = 180
    r = np.empty(W * 2, np.uint8); r[1::2] = y; r[0::2] = 128
    for p in range(4): r[4 * p] = n % 256; r[4 * p + 2] = 100 + n // 256
    return r

def noisy(f, amp=3):
    f = f.copy(); f[:, 1::2] = np.clip(f[:, 1::2].astype(int) + rng.integers(-amp, amp + 1, (len(f), W)), 0, 255).astype(np.uint8); return f

def simulate(plan, cut_field=10 ** 9, dark_fields=(), partial_after=5, static=False):
    """plan: list of ('lock'|'roll', passes). Returns published fields (240 rows, slots 20..259), per field (tape marker
    slot or None, repeat?), and the index of the partial repeat (None if none)."""
    lines = []                                    # (tape field, line) in tape order
    for k in range(600):
        for n in range(1, (262 if k % 2 == 0 else 263) + 1): lines.append((k, n))
    p = 0; fields = []; truth = []; since = 0; gap = 10; partial = None; toggle = 0
    def publish(pl):
        f = np.empty((240, W * 2), np.uint8)
        for i, s in enumerate(range(20, 260)):
            if s in (20, 21): f[i] = DEVICE; f[i, 1::2] = rng.integers(16, 235, W)
            elif s >= 258 or s > len(pl): f[i] = BLANK
            else: k, n = pl[s - 1]; f[i] = tape_line(1 if k >= cut_field else 0, n, k, k in dark_fields, static)
        return noisy(f)
    def mark(pl): return next((s for s, (_, ln) in enumerate(pl, 1) if ln == 16), None)
    for mode, count in plan:
        for _ in range(count):
            if mode == 'lock':
                while lines[p][1] != 1: p += 1
                n = 262 if lines[p][0] % 2 == 0 else 263
            else: n = 262 + toggle % 2
            pl = lines[p:p + n]; p += n
            if mode == 'roll': p += 27 + toggle % 2; toggle += 1
            fields.append(publish(pl)); truth.append((mark(pl), False)); since += 1
            if mode == 'roll' and since >= gap:
                fields.append(noisy(fields[-1], 1)); truth.append((None, True)); since = 0; gap = 21 - gap
            elif mode == 'roll' and partial is None and partial_after is not None and since == partial_after:
                pl = lines[p:p + 262]; f = publish(pl); f[150:] = noisy(fields[-1][150:], 1); p += 290; toggle += 1
                fields.append(f); truth.append((mark(pl), False)); partial = len(fields) - 1
    if len(fields) % 2: fields.pop(); truth.pop()
    return fields, truth, partial

def frames_of(fields): return b''.join(np.stack([fields[i], fields[i + 1]], 1).reshape(480, W * 2).tobytes() for i in range(0, len(fields), 2))
def run_inproc(fields):
    st = vstab.Stabiliser(); out = []
    b = frames_of(fields)
    for i in range(0, len(b), vstab.FRAME): out += st.push(b[i:i + vstab.FRAME])
    out += st.finish()
    o = np.frombuffer(b''.join(out), np.uint8).reshape(-1, 480, W * 2)
    return [o[g // 2][g % 2::2] for g in range(len(fields))], st

def label(row):
    if row[0] == 128 and row[2] == 128: return None
    if row[0] == DEVICE and row[2] == DEVICE: return 'device'
    return int(row[0]) + 256 * (int(row[2]) - 100)

def score(of, truth, partial, only=None):
    exact = off_one = wrong = black = flat_black = 0; bad = []
    for g, (mk, rep) in enumerate(truth):
        if only is not None and g not in only: continue
        if rep:
            # a repeat shows the same picture in the same place: held (rolling), or shown as itself (locked)
            assert [label(r) for r in of[g]] == [label(r) for r in of[g - 1]], 'field %d: a repeat moved the picture' % g; continue
        seen = mk is not None and 30 <= mk <= 256
        for i in range(240):
            o = 22 + i; lb = label(of[g][i])
            assert lb != 'device', 'field %d line %d: a Shuttle insert was moved into the picture' % (g, o)
            if lb is None:
                black += 1
                if g != partial and 100 <= o < 140 and mk is not None and o - 16 + mk <= 257 and o - 16 + mk >= 22: flat_black += 1
                continue
            if lb == o: exact += 1
            elif abs(lb - o) == 1 and (not seen or o > 257 - (mk - 16)): off_one += 1
            else: wrong += 1; bad.append((g, o, lb, mk))
    return exact, off_one, wrong, black, flat_black, bad

# 1. the full run: lock, roll (scene cut, repeats, a partial repeat), relock, roll, lock; through the command
plan = [('lock', 8), ('roll', 60), ('lock', 30), ('roll', 40), ('lock', 12)]
fields, truth, partial = simulate(plan, cut_field=60)
whole = frames_of(fields)
r = subprocess.run([sys.executable, VSTAB, '--log', '/dev/stderr'], input=whole, capture_output=True)
assert r.returncode == 0, r.stderr.decode()[-500:]
out = np.frombuffer(r.stdout, np.uint8).reshape(-1, 480, W * 2)
assert out.shape[0] * 2 == len(fields), 'frames out %d, in %d' % (out.shape[0], len(fields) // 2)
of = [out[g // 2][g % 2::2] for g in range(len(fields))]
exact, off_one, wrong, black, flat_black, bad = score(of, truth, partial)
print('full run rows: %d exact, %d one line off (wrapped below the blanking, or the marker hidden), %d wrong, %d black' % (exact, off_one, wrong, black))
assert wrong == 0, bad[:10]
assert flat_black == 0, '%d rows of the flat band blacked out in fresh fields' % flat_black
assert partial is not None
roll_black = [sum(label(of[g][i]) is None for i in range(240)) for g, (mk, rep) in enumerate(truth) if not rep and mk is not None and g != partial]
assert max(roll_black) <= 30, max(roll_black)
readings = [ln.split(',')[6] for ln in r.stderr.decode().splitlines()[1:] if ln.count(',') == 8]
assert any(x.endswith('/cut') for x in readings), 'the scene cut was not carried as a cut'
assert any(x.endswith('/relock') for x in readings), 'no relock was read'
states = [ln.split(',')[3] for ln in r.stderr.decode().splitlines()[1:] if ln.count(',') == 8]
locked_repeats = sum(1 for g, s in enumerate(states) if s == 'repeat' and not truth[g][1])
assert locked_repeats == 0, '%d new fields taken for repeats' % locked_repeats

# 2. input arriving in pieces (the replay writes ~9 KiB at a time): the same output as one write
pp = subprocess.Popen([sys.executable, VSTAB], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
def feed():
    for k in range(0, len(whole), 9216): pp.stdin.write(whole[k:k + 9216]); pp.stdin.flush()
    pp.stdin.close()
t = threading.Thread(target=feed); t.start(); got = pp.stdout.read(); t.join(); pp.wait()
assert pp.returncode == 0 and got == r.stdout, 'piecewise input changed the output (exit %d)' % pp.returncode

# 3. a relock after a roll of every length: the locked stretch after it is placed exactly
fails = []
for n in range(1, 40):
    f, tr, _ = simulate([('lock', 6), ('roll', n), ('lock', 30)], partial_after=None)
    of2, _ = run_inproc(f)
    tail = set(range(len(f) - 20, len(f)))                      # locked passes well after the relock (30 at the end)
    e, o1, wr, bl, fb, bd = score(of2, tr, None, only=tail)
    if wr or o1: fails.append((n, wr, o1))
print('relock after rolls of 1-39 passes: %d misplaced' % len(fails))
assert not fails, fails

# 4. a dark band that looks like blanking, in a still (locked) picture: not followed
f, tr, _ = simulate([('lock', 40)], dark_fields=set(range(600)), partial_after=None)
of3, st = run_inproc(f)
e, o1, wr, bl, fb, bd = score(of3, tr, None)
assert wr == 0 and o1 == 0, ('a dark band moved the picture', bd[:5])
assert st.tr.ignored > 0, 'the dark band was never seen as a marker: the test does not test anything'
print('dark band in a still picture: seen %d times, not followed' % st.tr.ignored)

# 5. the same dark band rolling with the picture, the marker often in view too: the marker is followed, not the band
f, tr, _ = simulate([('lock', 6), ('roll', 60)], dark_fields=set(range(600)), partial_after=None)
of4, st = run_inproc(f)
e, o1, wr, bl, fb, bd = score(of4, tr, None)
print('dark band rolling with the picture: %d rows exact, %d one line off, %d wrong' % (e, o1, wr))
assert wr == 0, ('the rolling dark band was followed', bd[:5])

# 6. locked, the lower part of the picture holding still: not taken for a repeat, nothing blacked out
f, tr, _ = simulate([('lock', 30)], partial_after=None, static=True)
of5, _ = run_inproc(f)
e, o1, wr, bl, fb, bd = score(of5, tr, None)
assert wr == 0 and o1 == 0, bd[:5]
blk = max(sum(label(of5[g][i]) is None for i in range(240)) for g in range(len(f)))
assert blk <= 4, '%d picture rows blacked out in a locked field (only the 4 blanked slots 258-261 may be)' % blk
print('locked picture with a still lower part: all rows in place')

# 7. a frame cut short is refused, never padded
r = subprocess.run([sys.executable, VSTAB], input=frames_of(fields[:4]) + b'\0' * 1000, capture_output=True)
assert r.returncode != 0 and b'inside a frame' in r.stderr, r.stderr
print('ALL OK')
