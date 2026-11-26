#!/usr/bin/env bash
# Put MongoDB's `mongod` where FiftyOne looks for it, inside the viewer's environment.
#
# Why: FiftyOne keeps its datasets in MongoDB and normally installs `mongod` through the
# `fiftyone-db` package. On Linux that package is built from source and picks a MongoDB build by
# distro; its table stops at Ubuntu 24.x, so on Ubuntu 26.04 it installs with no `mongod` at all
# (dev/notes/milestones/README.md, parked debt). This script fetches the build `fiftyone-db`
# itself picks for Ubuntu 24.04, checks its SHA-256 against MongoDB's published one, and copies
# `mongod` into the environment. No sudo, nothing installed system-wide.
#
# Re-runnable. Run it again if FiftyOne says "Could not find `mongod`" (a fresh `uv sync` that
# reinstalls fiftyone-db removes the binary).
set -euo pipefail

VERSION=8.0.17
NAME=mongodb-linux-x86_64-ubuntu2404-$VERSION
URL=https://fastdl.mongodb.org/linux/$NAME.tgz

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/src/rin/viewer/.venv/bin/python"
[ -x "$PY" ] || uv sync --locked --project "$ROOT/src/rin/viewer"

BIN_DIR=$("$PY" -c "import os, fiftyone.db as d; print(d.FIFTYONE_DB_BIN_DIR)" 2>/dev/null)
mkdir -p "$BIN_DIR"
if [ -x "$BIN_DIR/mongod" ]; then
  echo "already there: $BIN_DIR/mongod"
  "$BIN_DIR/mongod" --version | head -1
  exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "downloading $URL"
curl -fL --progress-bar -o "$TMP/$NAME.tgz" "$URL"
curl -fsSL -o "$TMP/$NAME.tgz.sha256" "$URL.sha256"
(cd "$TMP" && sha256sum -c "$NAME.tgz.sha256")   # stops here if the download is not MongoDB's file

tar -xzf "$TMP/$NAME.tgz" -C "$TMP" "$NAME/bin/mongod"
install -m 755 "$TMP/$NAME/bin/mongod" "$BIN_DIR/mongod"
echo "installed: $BIN_DIR/mongod"
"$BIN_DIR/mongod" --version | head -1   # fails loudly if a system library is missing
