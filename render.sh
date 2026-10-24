#!/bin/bash
# render.sh [--replace] CAPTURE.tpc OUT.mov [WORKDIR] -- a capture rendered with its rolling picture locked to the tape's own
# vertical blanking (vstab.py), through blackmagic-usb-mac's plain render: the same reading (checked reads from the share),
# audio-clock placement, audio, ProRes 422 encode and checks. This repository must be checked out as that repository's
# submodule. The replay publishes each field from line 20 / 282 (all the lines the synchroniser delivers), vstab.py turns each
# field to the tape's blanking, and each field is shown on its own (line-doubled at 59.94p): a stabilised field and its
# neighbour are not a weavable pair (fps after separatefields: see PLAIN_RENDER_DEINTERLACE in plain_render.sh).
# VSTAB_LOG=FILE keeps vstab.py's per-field decisions.
set -u
main() {
HERE=$(cd "$(dirname "$0")" && pwd)
TOP=$(git -C "$HERE" rev-parse --show-superproject-working-tree 2>/dev/null)
PR="$TOP/scripts/plain_render/plain_render.sh"
[ -n "$TOP" ] && [ -f "$PR" ] || { echo "REFUSE: no blackmagic-usb-mac checkout around this one (expected $PR)"; exit 2; }
case "$HERE${VSTAB_LOG:-}" in *"'"*) echo "REFUSE: a path with a single quote"; exit 2;; esac
F="python3 '$HERE/vstab.py'"; [ -n "${VSTAB_LOG:-}" ] && F="$F --log '$VSTAB_LOG'"
PLAIN_RENDER_FIRST_LINE=20,282 PLAIN_RENDER_FILTER="$F" PLAIN_RENDER_DEINTERLACE="separatefields,scale=720:480:flags=bilinear,fps=60000/1001" \
  bash "$PR" "$@"
}
main "$@"; exit $?
