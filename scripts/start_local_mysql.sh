#!/usr/bin/env bash
set -euo pipefail

MYSQL_RUN_DIR="${MYSQL_RUN_DIR:-/run/mysqld}"
MYSQL_HOST="${MYSQL_HOST:-127.0.0.1}"

mkdir -p "$MYSQL_RUN_DIR"
chown mysql:mysql "$MYSQL_RUN_DIR"

if pgrep -x mariadbd >/dev/null; then
  echo "MariaDB is already running."
  exit 0
fi

test ! -e "$MYSQL_RUN_DIR/mysqld.pid" || unlink "$MYSQL_RUN_DIR/mysqld.pid"
test ! -e "$MYSQL_RUN_DIR/mysqld.sock" || unlink "$MYSQL_RUN_DIR/mysqld.sock"

exec mariadbd \
  --user=mysql \
  --console \
  --skip-networking=0 \
  --bind-address="$MYSQL_HOST"
