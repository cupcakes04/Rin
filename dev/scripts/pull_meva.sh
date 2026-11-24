#!/usr/bin/env bash
# Download the MEVA slice used by the holy grail experiment (see docs/data.md).
# Re-runnable: files already on disk are skipped. Needs only curl (no AWS CLI).
set -euo pipefail
BASE=https://mevadata-public-01.s3.amazonaws.com
D="$(cd "$(dirname "$0")/../.." && pwd)/workspace/media/meva"
mkdir -p "$D/pool" "$D/test"

pull() { # s3_prefix dest_dir camera_regex max_files
  keys=$(curl -s --fail "$BASE/?list-type=2&prefix=$1" | grep -oP '(?<=<Key>)[^<]+' | grep -E "$3" | head -n "$4")
  for k in $keys; do
    f="$2/$(basename "$k")"
    [ -s "$f" ] && { echo "skip $f"; continue; }
    curl -s --fail -o "$f.part" "$BASE/$k" && mv "$f.part" "$f" && echo "ok $f"
  done
}

# Pool: school site, cameras G299 + G330, 2018-03-07 09:55-10:55 (24 clips of 5 min)
pull drops-123-r13/2018-03-07/10/ "$D/pool" 'school\.G(299|330)\.' 100
# Test: same cameras, a different day, 15 min each (6 clips)
pull drops-123-r13/2018-03-09/10/ "$D/test" 'school\.G(299|330)\.' 6
echo DONE
