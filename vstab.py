#!/usr/bin/env python3
"""vstab.py [--log FILE] < frames > frames -- vertical lock for a capture whose frame synchroniser lost the tape's vertical sync.

Input and output: 720x480 UYVY frames, two fields woven (field 1 on even rows), as frameserver_replay publishes them with
FS_REGISTRATION_OFF=1 FS_FIRST_LINE=20,282: each field's rows are its synchroniser slots 20..259 (slot = NTSC line in field 1,
frame line - 262 in field 2). One frame out per frame in, so plain_render's audio-clock relay and frame-count check hold.
Each output field is lines 22..261 of the tape's field, its vertical blanking (marker line, below) put at line 16.

What the synchroniser does, measured on 2026-10-24 12-26-06 (849 units; see README.md):
- Locked, it writes one tape field per memory pass: tape line n lands in slot n, and the tape's blanking is hidden in its own
  (slots 1-19 and 258-262 never reach the capture).
- Unlocked, it starts its own input fields every ~290 lines, keeps ~262.5 and drops ~27.5. The tape's blanking then moves up
  27.5 slots per pass (27 and 28 alternating with parity), and a pass picks up where the last left off plus the dropped lines.
  Being slower than its output, it repeats a field (or the lower part of one) about every 10.5 output fields.
- It re-catches sync at times, for a field or for many; the tape's blanking then goes back to slot 16.
The tape's blanking is about 15 lines at blanking level, a bright structured line (the marker, NTSC line ~16), more blanking,
picture from line ~22. Where the marker is visible it decides the position; elsewhere the position follows the measured shift
of the picture between passes. Lines no pass delivered (~25 per field: the synchroniser's hidden slots, rolling through the
picture) are black. The 27.5 dropped lines per unlocked pass are not in the capture and are not recovered.

Measured on luma; chroma moves with its row. Per field the log (--log) records: field, parity, stale rows (rows equal to the
previous field's, i.e. a repeat), state, measured shift and its match error, the tracker's reading, marker detections and the
marker slot used."""
import argparse, sys
import numpy as np

W = 720; ROWS = 240; FRAME = W * 2 * ROWS * 2
FIRST = 20                         # slot of each field's first published row (FS_FIRST_LINE=20,282)
VIS = np.arange(20, 258)           # slots that can carry tape content (258-262 the synchroniser blanks)
PIC = 22                           # slots 20-21 are the Shuttle's own inserts (pulse line, re-encoded caption): never moved
MARK_LINE = 16                     # where the tape's marker line goes (picture then starts at 22, as when locked)
OUT0 = 22                          # first output line of each field (240 lines: 22..261)
TEXTURE = 6.0                      # row deviation (luma) above which a row can tell a repeat from new picture
STALE_MAD = 3.0                    # a textured row this close to the previous field's was repeated (measured on 2026-10-24
                                   # 12-26-06: repeated rows 1.37 median, 2.94 p99; new rows with texture under it 0.2-0.4%)
REPEAT_MIN, REPEAT_FRAC = 20, 0.9  # a repeat: >= 20 textured rows to the bottom, >= 90% of them repeated
DARK, DARK_SD, BRIGHT, DARK_RUN = 10, 6, 20, 10   # marker: >= 10 slots of blanking (mean < 10; median deviation < 6: 0.85
                                   # measured, 4-8 on a few noisy rolling fields), then a row > 20
VISIBLE = (30, 256)                # marker slots the detector can see (it needs 10 blanking slots at or below slot 20)
FAR = 8                            # a marker further than this from the tracked slot is not taken on trust
LOCK_SHIFT = 2                     # |shift| <= 2 between passes: still (the two parities differ by a line)
ROLL = (-33, -24)                  # a rolling pass's shift; (-45, -15) once rolling (measured 27-37)
ROLL_WIDE = (-45, -15)
JUMP_ERR = 1.6                     # a match worse than 1.6x the recent median: the synchroniser re-caught sync
ERR_WINDOW = 12
PENDING = 24                       # passes a poor match may stay undecided (buffered fields)
BLACK = np.tile(np.array([128, 16], np.uint8), W)

def luma(rows): return rows[:, 1::2].astype(np.float32)

def stale_rows(cur, prev):
    """The rows the synchroniser repeated from the previous field, and whether that is the whole field. Its read catches its
    write at some row, so a repeat runs from that row to the bottom. Judged on textured rows: a flat row (sky, a wall, a
    fade) matches the previous field within noise whether it was repeated or not, and takes the verdict of its block."""
    c = luma(cur); p = luma(prev); n = len(c); none = np.zeros(n, bool)
    idx = np.nonzero((c.std(1) >= TEXTURE) & (p.std(1) >= TEXTURE))[0]
    if len(idx) < REPEAT_MIN: return none, False                     # nothing to tell by: new picture
    same = np.abs(c[idx] - p[idx]).mean(1) < STALE_MAD
    below = np.cumsum(same[::-1])[::-1]; count = np.arange(len(idx), 0, -1)    # repeated / textured rows from each to the bottom
    ok = np.nonzero((count >= REPEAT_MIN) & (below >= REPEAT_FRAC * count))[0]
    if not len(ok): return none, False
    k = ok[0] + int(np.argmax(same[ok[0]:]))                          # the block starts at a repeated row
    st = none.copy(); st[idx[k]:] = True
    return st, int((~same[:k]).sum()) <= 2                            # whole: (almost) no new textured row above it

def markers(field, fresh):
    """slots of marker lines: a run of >= DARK_RUN slots of blanking, then a bright one (field: slot -> UYVY row,
    slots 20..259). Only in rows this pass delivered: a marker in repeated rows is the previous pass's."""
    y = field[:, 1::2].astype(np.float32)
    mean = {s: float(y[s - FIRST].mean()) for s in range(20, 258)}; sd = {s: float(y[s - FIRST].std()) for s in range(20, 258)}
    out = []; s = 20
    while s < 257:
        if mean[s] < DARK:
            e = s
            while e < 257 and mean[e] < DARK: e += 1
            if (e - s >= DARK_RUN and e < 257 and mean[e] > BRIGHT and np.median([sd[x] for x in range(s, e)]) < DARK_SD
                    and fresh[s - 20:e - 19].all()): out.append(e)
            s = e
        else: s += 1
    return out

def shift(A, B):
    """d such that B[s] ~ A[s - d] (picture moved down d slots from pass A to pass B), its error and the runner-up's;
    A, B: luma of slots 20..257, NaN rows where a pass delivered nothing new"""
    best = []
    for d in range(-45, 46):
        if d >= 0: a = A[:len(A) - d]; b = B[d:]
        else: a = A[-d:]; b = B[:len(B) + d]
        v = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
        if v.sum() < 80: continue
        best.append((float(np.abs(a[v][:, 40:680:2] - b[v][:, 40:680:2]).mean()), d))
    best.sort()
    if len(best) < 2: return None, None, None
    return best[0][1], best[0][0], best[1][0]

def advance(m, prev, kind, d):
    """the marker slot one pass on, under reading kind (still / roll / unk / relock)"""
    if kind == 'roll': m += -27.5 if d in (-27, -28) else d
    elif kind == 'unk' and prev in ('roll', 'mark'): m -= 27.5
    elif kind == 'relock': m = float(MARK_LINE)
    return wrap(m)

def wrap(m):
    """one tape field's worth of slots, -8 .. 254.5: the same position a whole field on is the same position"""
    while m < -8: m += 262.5
    while m >= 254.5: m -= 262.5
    return m

def near(dets, p): return min(min(abs(x - p), abs(x - p + 262.5), abs(x - p - 262.5)) for x in dets)

RANK = dict(relock=0, cont=1, marker=2, cut=3)   # undecided after PENDING passes with equal scores: the likelier story first

def seeable(m, fresh):
    """the detector could see a marker at slot m in this pass: in its range, with it and its blanking delivered fresh"""
    if not VISIBLE[0] <= m <= VISIBLE[1]: return False
    k = int(round(m)) - 20
    return bool(fresh[max(0, k - DARK_RUN):k + 1].all())

class Tracker:
    """The slot of the tape's marker line in each fresh pass. Two events read alike at the pass where they happen and differ
    later, so both readings are carried until a pass tells them apart:
    - a pass whose picture matches the last one poorly, or moved by neither a still nor a rolling step: a cut in the
      programme ('cut') or the synchroniser re-catching sync ('relock');
    - while rolling, a marker far from the tracked slot: the marker ('marker') or dark picture that looks like blanking,
      the tape's marker being hidden ('cont'). Only while the picture rolls: holding still, the synchroniser is locked and
      the tape's marker is hidden, so a marker seen then is dark picture and is not followed (counted in .ignored); so is
      one on a pass with no measured shift (the first).
    In each pass a reading scores a hit when a marker is seen where it puts one, a miss when it puts one where the detector
    could see it and none is there. A pass decides when one reading hits and the other misses, or when both hit (dark
    picture that rolls with the picture and the marker are both seen: 'cont', the reading the picture's motion carried).
    Undecided after PENDING passes, hits minus misses decides, then RANK.
    observe() returns (marker slot, reading) for every pass it has settled, in order."""
    def __init__(self): self.m = float(MARK_LINE); self.kind = 'lock'; self.errs = []; self.fork = None; self.ignored = 0
    def rolling(self): return self.kind in ('roll', 'mark') or (self.fork is not None and any(h['kind'] in ('roll', 'mark') for h in self.fork))
    def observe(self, d, err, dets, fresh):
        med = float(np.median(self.errs[-ERR_WINDOW:])) if self.errs else None
        if err is not None: self.errs.append(err)
        if d is None: kind = 'unk'
        elif med is not None and err > JUMP_ERR * med: kind = 'suspect'
        elif abs(d) <= LOCK_SHIFT: kind = 'still'
        elif ROLL[0] <= d <= ROLL[1] or (ROLL_WIDE[0] <= d <= ROLL_WIDE[1] and self.kind in ('roll', 'mark')): kind = 'roll'
        else: kind = 'suspect'
        done = []
        if self.fork is not None and kind == 'suspect':          # a second event before the first was decided: settle it
            done = self.flush()
        if self.fork is None and kind != 'suspect':
            p = advance(self.m, self.kind, kind, d)
            if dets:
                c = float(min(dets, key=lambda x: near([x], p)))
                if near([c], p) <= FAR: p, kind = c, 'mark'
                elif kind != 'roll': self.ignored += 1        # still, or no shift to go by (the first pass)
                else:
                    hidden = not seeable(p, fresh)
                    self.fork = [dict(m=c, kind='mark', out=[(c, 'mark/marker')], hit=1, miss=0, name='marker'),
                                 dict(m=p, kind=kind, out=[(p, kind + '/cont')], hit=0, miss=0 if hidden else 1, name='cont')]
                    if not hidden: return done + self.settle(self.fork[0])   # the tracked slot was in view and empty
                    return done
            self.m, self.kind = p, kind; return done + [(p, kind)]
        if self.fork is None:
            cont = 'roll' if d is not None and ROLL_WIDE[0] <= d <= ROLL_WIDE[1] else ('still' if d is not None and abs(d) <= LOCK_SHIFT else 'unk')
            self.fork = [dict(m=self.m, kind=self.kind, out=[], hit=0, miss=0, first=cont, name='cut'),
                         dict(m=self.m, kind=self.kind, out=[], hit=0, miss=0, first='relock', name='relock')]
        res = []
        for h in self.fork:
            k = h.pop('first', None) or ('roll' if kind == 'suspect' and h['kind'] in ('roll', 'mark') else
                                          ('still' if kind == 'suspect' else kind))
            h['m'] = advance(h['m'], h['kind'], k, d); h['kind'] = k
            hit = bool(dets) and near(dets, h['m']) <= FAR
            miss = not hit and seeable(h['m'], fresh)
            h['hit'] += hit; h['miss'] += miss; res.append((hit, miss))
            if hit: h['m'] = float(min(dets, key=lambda x: near([x], h['m']))); h['kind'] = 'mark'; k = 'mark'
            h['out'].append((h['m'], k + '/' + h['name']))
        (h0, m0), (h1, m1) = res
        if h0 and h1: win = min(self.fork, key=lambda h: RANK[h['name']])
        elif (h0 and m1) or (h1 and m0): win = self.fork[0] if h0 else self.fork[1]
        elif len(self.fork[0]['out']) >= PENDING: win = self.best()
        else: return done
        return done + self.settle(win)
    def best(self): return min(self.fork, key=lambda h: (h['miss'] - h['hit'], RANK[h['name']]))
    def settle(self, win):
        self.m, self.kind, self.fork = win['m'], win['kind'], None; return win['out']
    def flush(self):
        return [] if self.fork is None else self.settle(self.best())

def rotate(field, fresh, m, parity):
    """the tape's field from one pass: slot s goes to line s - (m - 16); slots above the marker belong to the previous tape
    field and wrap below it. Rows not delivered stay black. Returns lines OUT0..OUT0+239 (UYVY)."""
    out = np.empty((263, W * 2), np.uint8); out[:] = BLACK
    top = int(round(m)) - MARK_LINE
    for s in VIS:
        if s < PIC or not fresh[s - 20]: continue
        o = s - top
        if o < 0: o += 262 if parity == 1 else 263
        if o >= 263: o -= 263
        out[o] = field[s - FIRST]
    return out[OUT0:OUT0 + ROWS]

class Stabiliser:
    """push(frame bytes) returns the frames that are settled (none while a reading is undecided); finish() the rest"""
    def __init__(self, log=None):
        self.prev = None; self.prev_luma = None; self.tr = Tracker(); self.n = 0; self.log = log
        self.q = []; self.wait = []; self.done = []; self.last_out = None; self.last_m = float(MARK_LINE)
        if log: log.write('field,parity,stale_rows,state,shift,error,reading,markers,marker_slot\n')
    def field(self, f, parity):
        """f: 240 UYVY rows (slots 20..259)"""
        rows = f[:len(VIS)]
        # the synchroniser repeats while rolling (its writes slower than its reads); locked, a field that matches the last is
        # a still picture, and shown as it is either way
        none = (np.zeros(len(VIS), bool), False)
        stale, whole = none if self.prev is None or not self.tr.rolling() else stale_rows(rows, self.prev[:len(VIS)])
        self.prev = f; ns = int(stale.sum()); e = dict(n=self.n, parity=parity, ns=ns, d=None, err=None, dets=[]); self.n += 1
        self.q.append(e)
        if self.prev_luma is not None and whole: e['state'] = 'repeat'; return
        L = luma(rows); L[stale] = np.nan
        e.update(state='partial' if ns else 'fresh', f=f, fresh=~stale, dets=markers(f, ~stale))
        if self.prev_luma is not None: e['d'], e['err'], _ = shift(self.prev_luma, L)
        self.prev_luma = L; self.wait.append(e)
        self.settle(self.tr.observe(e['d'], e['err'], e['dets'], ~stale))
    def settle(self, got):
        for m, reading in got: e = self.wait.pop(0); e['m'] = m; e['reading'] = reading
        while self.q and (self.q[0]['state'] == 'repeat' or 'm' in self.q[0]):
            e = self.q.pop(0)
            if e['state'] == 'repeat': out = self.last_out; m = self.last_m; e['reading'] = ''
            else: out = rotate(e['f'], e['fresh'], e['m'], e['parity']); m = e['m']
            if self.log: self.log.write('%d,%d,%d,%s,%s,%s,%s,%s,%.1f\n' % (e['n'], e['parity'], e['ns'], e['state'],
                '' if e['d'] is None else e['d'], '' if e['err'] is None else '%.2f' % e['err'], e['reading'],
                ';'.join(map(str, e['dets'])), m))
            self.last_out = out; self.last_m = m; self.done.append(out)
    def frames(self):
        out = []
        while len(self.done) >= 2:
            o = np.empty((ROWS * 2, W * 2), np.uint8); o[0::2] = self.done.pop(0); o[1::2] = self.done.pop(0); out.append(o.tobytes())
        return out
    def push(self, b):
        fr = np.frombuffer(b, np.uint8).reshape(ROWS * 2, W * 2); self.field(fr[0::2], 1); self.field(fr[1::2], 2); return self.frames()
    def finish(self):
        self.settle(self.tr.flush()); out = self.frames()
        if self.q or self.done: raise RuntimeError('vstab: %d fields left unsettled at the end' % (len(self.q) + len(self.done)))
        return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--log')
    a = ap.parse_args()
    log = open(a.log, 'w') if a.log else None
    st = Stabiliser(log); inp = sys.stdin.buffer; out = sys.stdout.buffer; n = w = 0
    while True:
        b = inp.read(FRAME)
        if not b: break
        if len(b) != FRAME: sys.exit('vstab: input ended inside a frame (%d of %d bytes) after %d frames' % (len(b), FRAME, n))
        n += 1
        for o in st.push(b): out.write(o); w += 1
    for o in st.finish(): out.write(o); w += 1
    out.flush()
    if log: log.close()
    if w != n: sys.exit('vstab: %d frames in, %d out' % (n, w))
    print('vstab: %d frames; %d markers seen in a still picture not followed' % (n, st.tr.ignored), file=sys.stderr)

if __name__ == '__main__': main()
