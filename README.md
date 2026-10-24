# vhs-recovery

Recovering picture from damaged analog video captures made with
[blackmagic-usb-mac](https://github.com/vinylfreak89/blackmagic-usb-mac) (Intensity Shuttle USB 3.0, NTSC, through a frame
synchroniser). This repository is that one's submodule: it reads captures and renders through its tools, never a copy of them.

## vstab: vertical lock to the tape's own blanking

When a tape's vertical sync is too weak for the frame synchroniser, the picture rolls. `vstab.py` puts it back: each field is
turned so the tape's own vertical blanking sits where it belongs, and the lines nobody delivered show as a black band rolling
through the picture.

```
recovery/render.sh CAPTURE.tpc OUT.mov            # ProRes 422, 59.94p, sound, through plain_render.sh
VSTAB_LOG=decisions.csv recovery/render.sh ...    # with a row per field: shift, reading, marker slot
python3 recovery/tests/test_vstab.py              # known-answer tests on a simulated synchroniser
```

### What the synchroniser does (measured 2026-10-24 on a 28 s capture of an off-air tape with unstable vertical sync)

- **Locked**, it writes one tape field per pass of its field memory: tape line *n* in slot *n*, the tape's blanking hidden in
  its own. Slots 1-19 and 258-262 never reach the capture (about 25 lines per field), and slots 20-21 carry the Shuttle's own
  inserts.
- **Unlocked**, it starts its own input fields every ~290 lines, keeps ~262.5 and drops ~27.5. The tape's blanking moves up
  27.5 slots per pass (27 and 28 alternating), and the synchroniser, now slower than its output, repeats a field (or the lower
  part of one) every ~10.5 output fields.
  - Deciding test, at 18 lock-to-roll transitions in different scenes: the picture in the first three rolling passes sits
    27-28, 55 and 82-84 lines below the last locked field. Without dropped lines it would be 0, 27.5, 55.
- It re-catches sync at times, for one field or for many.
- The tape's blanking, as it arrives: about 15 lines at blanking level, a bright structured line (the marker, NTSC line ~16),
  more blanking, picture from line ~22.

### How vstab follows it

- **Repeats:** the synchroniser repeats while rolling, its writes being slower than its reads. A repeat runs from the row
  where its read caught its write down to the bottom of the field. It is judged on textured rows only, since a flat row
  (sky, a wall, a fade) matches the previous field within noise either way. Repeated fields repeat the output; repeated
  rows of a partly new field show black. While locked, a field that matches the last is a still picture and is shown as
  it is.
- **The marker:** at least 10 rows at blanking level (mean under 10, median deviation under 6), then a brighter row, then
  blanking again within 5 rows (the tape's lines 17-21; on the capture, 4 rows below in 402 of 409 markers), all in rows
  this pass delivered. Dark picture with a bright edge is followed by picture, not blanking. Where the marker is visible
  it decides the position, exactly.
- **Elsewhere:** the position follows the measured shift of the picture between passes: still, rolling, or a jump when the
  synchroniser re-catches sync.
- **Two events that read alike at the pass where they happen:**
  - A pass that matches its predecessor poorly: a cut in the programme or a relock.
  - While rolling, a marker far from the tracked position: the marker, or dark picture.

  Both readings are carried until a pass tells them apart. In each pass a reading scores a hit when a marker is seen
  where it puts one, and a miss when it puts one where the detector could see it (in range, its rows delivered) and none
  is there. A pass decides when one reading hits and the other misses, or when both hit: then it is the reading the
  picture's motion carried, since dark picture rolling with the picture is seen as well as the marker. After 24
  undecided passes, hits minus misses decides; on a tie relock, then the carried reading.
  - In a dark scene the measured shift undershoots: a marker seen where the shift's reading is hidden is held, and wins
    on its hit when no further marker shows.
- **While the picture holds still:** the synchroniser is locked and the tape's marker is hidden, so a "marker" seen then is
  dark picture and is not followed.
- Measured on luma; chroma moves with its row.

On the capture it was built from, the placement differs from the hand-checked prototype in 2 of 1,630 new fields, both
corrections: a marker the prototype read from repeated rows, and a relock it took for a still picture.

### Limits

- The ~27.5 lines dropped per unlocked pass are not in the capture.
- Where the marker is hidden, a pass moves 27.5 lines and the half line is not observable: the position is good to a line.
- Each output field joins the tape's field with the end of the previous one across the missing band. The part below the band
  can sit a line off, as that field's parity is not known.
- Dark picture shaped exactly like the blanking (blanking-level rows, a bright row, blanking-level rows again), rolling
  with the picture, is told from the marker only when a pass shows the difference. If the true marker is unreadable in
  the pass where such a band is seen, the band is followed.
- The render loses its last field (1/60 s): ffmpeg's fps step after separatefields drops it at the end of the stream.
- The marker here is a broadcast test line in the vertical interval. A recording without one (a home camcorder tape, say)
  needs another landmark: the blanking interval's start, or the half line ending the field.

## hretime: horizontal timing from each line's own blanking (prototype)

`hretime.py` is a filter in the same frame format as `vstab.py` (720x480 UYVY, fields from line 20 / 282). It moves each
line sideways to where its own horizontal blanking says it belongs.

```
python3 recovery/hretime.py [--log FILE] < frames > frames
python3 recovery/tests/test_hretime.py
```

- **The signal view.** The tape signal is one waveform in time, and the Shuttle keeps 720 of every 858 samples of each line.
  A line whose timing slipped shows a piece of its blanking inside the window: at the left if late, at the right if early.
  Blanking reads far below the darkest picture, so it is found by level, and its edge is the line's reference point.
- **Bounds, not values, at the window ends.** The first 2-3 and last 3 samples of every window read dark whatever the
  timing. A line with bright picture at both ends and no edge clear of those samples is within about a sample of its place.
- **Choosing.** Each line gets candidates from the signal only: its own start edge, its own end edge, that bound, the median
  of nearby lines' references, no correction. A best path down the field picks one per line by its confidence plus the
  roughness it leaves against the line above. Picture only chooses among candidates; it never makes one.

Calibrated on two passes of an off-air test tape, deck line TBC off (the errors) and on (where lines belong):

| | Error vs the TBC pass (median / 90th pct, samples) |
|---|---|
| No correction | 3.85 / 16.1 |
| hretime | 1.18 / 5.23 |

No field came out rougher than the original. On that tape the error is a pure sideways shift per line (no stretch within a
line), and the tape's antenna ghost fakes end-of-picture edges ~14 samples early, which only the choosing step can reject.

**Known fault:** whole blocks of lines can still be offset where the picture changes brightness sharply or the blanking is
noisy. Roughness alone cannot tell the candidates apart there. A real timing error is analog and does not jump between
neighbouring lines (except at a head switch), and the choosing step does not yet use that.

## License

GPL-2.0-or-later, as blackmagic-usb-mac.
