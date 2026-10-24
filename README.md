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

- **Repeats:** a repeat runs from the row where the synchroniser's read caught its write down to the bottom of the field.
  It is judged on textured rows only, since a flat row (sky, a wall, a fade) matches the previous field within noise either
  way. Repeated fields repeat the output; repeated rows of a partly new field show black.
- **The marker:** at least 10 rows at blanking level (mean under 10, median deviation under 6), then a brighter row, all in
  rows this pass delivered. Where it is visible it decides the position, exactly.
- **Elsewhere:** the position follows the measured shift of the picture between passes: still, rolling, or a jump when the
  synchroniser re-catches sync.
- **Two events that read alike at the pass where they happen:**
  - A pass that matches its predecessor poorly: a cut in the programme or a relock.
  - While rolling, a marker far from the tracked position: the marker, or dark picture.

  Both readings are carried until the next visible marker decides. After 24 passes without one, the reading whose
  predicted markers went unseen fewer times wins; on a tie, relock first, then the seen marker. (In a dark scene the
  measured shift undershoots, while detected markers have proved reliable.)
- **While the picture holds still:** the synchroniser is locked and the tape's marker is hidden, so a "marker" seen then is
  dark picture and is not followed.
- Measured on luma; chroma moves with its row.

On the capture it was built from, the result differs from the hand-checked prototype in 2 of 1,630 new fields, both
corrections: a marker the prototype read from repeated rows, and a relock it took for a still picture.

### Limits

- The ~27.5 lines dropped per unlocked pass are not in the capture.
- Where the marker is hidden, a pass moves 27.5 lines and the half line is not observable: the position is good to a line.
- Each output field joins the tape's field with the end of the previous one across the missing band. The part below the band
  can sit a line off, as that field's parity is not known.
- Dark picture at blanking level with a bright edge below it, rolling with the picture, can pass for the marker.
- The marker here is a broadcast test line in the vertical interval. A recording without one (a home camcorder tape, say)
  needs another landmark: the blanking interval's start, or the half line ending the field.

## License

GPL-2.0-or-later, as blackmagic-usb-mac.
