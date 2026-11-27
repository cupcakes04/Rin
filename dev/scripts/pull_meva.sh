#!/usr/bin/env bash
# Download the MEVA slice used by the holy grail experiment (see dev/notes/heavy_data.md).
# Re-runnable: files already on disk are skipped. Needs only curl (no AWS CLI).
set -euo pipefail
BASE=https://mevadata-public-01.s3.amazonaws.com
D="$(cd "$(dirname "$0")/../.." && pwd)/workspace/media/meva"
mkdir -p "$D/pool" "$D/val" "$D/test"

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
# Pool, quiet part (D-41): the same day, 17:10-17:40, an empty gym (12 clips of 5 min)
pull drops-123-r13/2018-03-07/17/ "$D/pool" '2018-03-07\.17-[123][05]-00\..*school\.G(299|330)\.' 12
# Val (D-31): same cameras, the pool day, 11:30-11:40 and 17:00-17:10 (2 + 2 clips per camera)
pull drops-123-r13/2018-03-07/11/ "$D/val" '2018-03-07\.11-3[05]-00\..*school\.G(299|330)\.' 4
pull drops-123-r13/2018-03-07/17/ "$D/val" '2018-03-07\.17-0[05]-00\..*school\.G(299|330)\.' 4
# Test (D-32): same cameras, five other days, 15 min per camera each (6 clips per window)
pull drops-123-r13/2018-03-09/10/ "$D/test" 'school\.G(299|330)\.' 6   # 10:10-10:25
pull drops-123-r13/2018-03-11/11/ "$D/test" 'school\.G(299|330)\.' 6   # 11:15-11:30
pull drops-123-r13/2018-03-12/10/ "$D/test" 'school\.G(299|330)\.' 6   # 10:00-10:15
pull drops-123-r13/2018-03-13/16/ "$D/test" 'school\.G(299|330)\.' 6   # 15:55-16:10
pull drops-123-r13/2018-03-15/15/ "$D/test" 'school\.G(299|330)\.' 6   # 14:55-15:10
echo DONE
