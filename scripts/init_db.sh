#!/usr/bin/env bash
# Bootstrap the claude-memory database on a fresh machine.
#
# Reads PG_* vars from the project's .env (if present), creates the database
# if it does not exist, and applies sql/schema.sql. Idempotent.
#
# Required on the host:
#   - postgresql (>= 14) running locally
#   - pgvector >= 0.5.0 installed for the target cluster
#   - pg_trgm + pgcrypto contrib modules available
#
# Usage:
#   scripts/init_db.sh            # uses .env
#   PG_DATABASE=foo scripts/init_db.sh   # override per-run

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
schema_file="${project_dir}/sql/schema.sql"
env_file="${project_dir}/.env"

if [[ -f "${env_file}" ]]; then
   # Export only the PG_* keys so we don't accidentally leak unrelated env.
   set -a
   # shellcheck disable=SC1090
   source <(grep -E '^PG_[A-Z_]+=' "${env_file}")
   set +a
fi

PG_HOST="${PG_HOST:-127.0.0.1}"
PG_PORT="${PG_PORT:-5432}"
PG_USER="${PG_USER:-$(whoami)}"
PG_DATABASE="${PG_DATABASE:-claude_memory}"
export PGPASSWORD="${PG_PASSWORD:-}"

psql_admin=(psql -v ON_ERROR_STOP=1 -h "${PG_HOST}" -p "${PG_PORT}" -U "${PG_USER}" -d postgres)
psql_target=(psql -v ON_ERROR_STOP=1 -h "${PG_HOST}" -p "${PG_PORT}" -U "${PG_USER}" -d "${PG_DATABASE}")

echo "[init_db] target: ${PG_USER}@${PG_HOST}:${PG_PORT}/${PG_DATABASE}"

db_exists=$("${psql_admin[@]}" -tAc \
   "SELECT 1 FROM pg_database WHERE datname = '${PG_DATABASE}'")

if [[ "${db_exists}" != "1" ]]; then
   echo "[init_db] creating database ${PG_DATABASE}"
   "${psql_admin[@]}" -c "CREATE DATABASE \"${PG_DATABASE}\""
else
   echo "[init_db] database ${PG_DATABASE} already exists"
fi

echo "[init_db] applying schema from ${schema_file}"
"${psql_target[@]}" -f "${schema_file}"

echo "[init_db] done"
