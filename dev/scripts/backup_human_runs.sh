#!/usr/bin/env bash
# Copy every run that holds a person's work (human.*) out of the store, into one dated
# archive. The rest of the store can be rebuilt from the media and the code; these runs cannot.
#
#   bash dev/scripts/backup_human_runs.sh            # into workspace/backups/
#   bash dev/scripts/backup_human_runs.sh <folder>   # anywhere else
#
# workspace/backups/ is on the same WSL disk as the store: it survives a deleted or broken
# store, not a broken disk.
set -euo pipefail

if [ $# -gt 1 ]; then
    echo "usage: $0 [backup folder]" >&2
    exit 1
fi
root="$(cd "$(dirname "$0")/../.." && pwd)"
store="$root/workspace/store"
folder="${1:-$root/workspace/backups}"
mkdir -p "$folder"
dest="$(cd "$folder" && pwd)"  # absolute, so a relative folder still works after the cd below
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
