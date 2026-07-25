#!/bin/bash
# Download a large file in parallel byte ranges, then reassemble it.
#
#   bash scripts/parallel_download.sh <url> <destination> [connections]
#
# The COCO image zips come down at a couple of MB/s on a single connection,
# which puts the 13 GB train2014 archive at over two hours. The server supports
# range requests, so splitting the file across ~16 connections turns that into
# minutes. Falls back to a plain single-stream download when the server does not
# advertise range support.
#
# Resumable: chunks already complete are skipped, so re-running after an
# interruption only fetches what is missing.

set -euo pipefail

URL="${1:?usage: parallel_download.sh <url> <dest> [connections]}"
DEST="${2:?usage: parallel_download.sh <url> <dest> [connections]}"
CONNECTIONS="${3:-16}"

if [ -s "$DEST" ]; then
    echo "already present: $DEST"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
PARTS_DIR="$DEST.parts"
mkdir -p "$PARTS_DIR"

read -r ACCEPTS_RANGES TOTAL < <(
    curl -sIL "$URL" | awk '
        tolower($0) ~ /^accept-ranges: *bytes/ { ranges="yes" }
        tolower($0) ~ /^content-length:/       { gsub(/\r/,""); len=$2 }
        END { print (ranges ? "yes" : "no"), (len ? len : 0) }'
)

if [ "$ACCEPTS_RANGES" != "yes" ] || [ "$TOTAL" -le 0 ]; then
    echo "server does not support ranges; falling back to a single stream"
    curl -fL --retry 3 -o "$DEST" "$URL"
    rmdir "$PARTS_DIR" 2>/dev/null || true
    exit 0
fi

echo "downloading $(basename "$DEST"): $((TOTAL / 1048576)) MB over $CONNECTIONS connections"

CHUNK=$(( (TOTAL + CONNECTIONS - 1) / CONNECTIONS ))
pids=()
for i in $(seq 0 $((CONNECTIONS - 1))); do
    start=$((i * CHUNK))
    end=$((start + CHUNK - 1))
    [ "$end" -ge "$TOTAL" ] && end=$((TOTAL - 1))
    [ "$start" -ge "$TOTAL" ] && break

    part="$PARTS_DIR/part.$(printf '%03d' "$i")"
    expected=$((end - start + 1))
    # Skip chunks that are already complete, so re-runs resume.
    if [ -f "$part" ] && [ "$(stat -c %s "$part")" -eq "$expected" ]; then
        continue
    fi
    curl -fsSL --retry 5 --retry-delay 2 -r "${start}-${end}" -o "$part" "$URL" &
    pids+=($!)
done

failed=0
for pid in "${pids[@]:-}"; do
    wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
    echo "at least one chunk failed; re-run to resume" >&2
    exit 1
fi

cat "$PARTS_DIR"/part.* > "$DEST"

ACTUAL=$(stat -c %s "$DEST")
if [ "$ACTUAL" -ne "$TOTAL" ]; then
    echo "size mismatch: got $ACTUAL, expected $TOTAL — refusing to keep a corrupt file" >&2
    rm -f "$DEST"
    exit 1
fi

rm -rf "$PARTS_DIR"
echo "done: $DEST ($((ACTUAL / 1048576)) MB)"
