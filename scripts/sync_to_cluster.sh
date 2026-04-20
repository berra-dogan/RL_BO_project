#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="/Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation/"
REMOTE_ROOT="bd225@login.cx3.hpc.imperial.ac.uk:~/my_implementation/"
IGNORE_FILE="$LOCAL_ROOT/.rsyncignore"

if [ ! -f "$IGNORE_FILE" ]; then
  echo "Missing ignore file: $IGNORE_FILE" >&2
  exit 1
fi

rsync -av --exclude-from="$IGNORE_FILE" "$LOCAL_ROOT" "$REMOTE_ROOT"
