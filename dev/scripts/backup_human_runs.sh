#!/usr/bin/env bash
# Copy every run that holds a person's work (human.*) out of the workspace, into one dated
# archive in the folder you name. The rest of the store can be rebuilt from the media and
# the code; these runs cannot.
#
#   bash dev/scripts/backup_human_runs.sh /mnt/c/Users/<you>/rin_backups
#
# A folder on the Windows side (/mnt/c/...) survives a broken WSL disk; one inside WSL doesn't.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <backup folder>" >&2
    exit 1
fi
dest="$1"
store="$(cd "$(dirname "$0")/../.." && pwd)/workspace/store"
mkdir -p "$dest"
cd "$store"
shopt -s nullglob
runs=(runs/*-human.*)
if [ ${#runs[@]} -eq 0 ]; then
    echo "No human runs in $store/runs" >&2
    exit 1
fi
archive="$dest/rin_human_runs_$(date +%Y%m%d-%H%M%S).tar.gz"
tar czf "$archive" "${runs[@]}"
tar tzf "$archive" > /dev/null  # fail now if the archive can't be read back
echo "${#runs[@]} runs -> $archive"
