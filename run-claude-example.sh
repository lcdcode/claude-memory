#!/usr/bin/bash
# run-claude.sh — start postgres + ollama if needed, then launch claude in ~/projects.
set -u

PGDATA="$HOME/projects/pgdata"
PGLOG="$PGDATA/logfile"
OLLAMA_LOG="$HOME/projects/claude-memory/ollama.log"

start_postgres() {
    if pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
        echo "[postgres] already running"
        return 0
    fi
    echo "[postgres] starting..."
    pg_ctl -D "$PGDATA" -l "$PGLOG" -w start
}

start_ollama() {
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "[ollama] already running"
        return 0
    fi
    echo "[ollama] starting..."
    nohup ollama serve >"$OLLAMA_LOG" 2>&1 &
    disown
    for i in $(seq 1 20); do
        if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            echo "[ollama] ready"
            return 0
        fi
        sleep 0.5
    done
    echo "[ollama] WARNING: did not respond within 10s — check $OLLAMA_LOG" >&2
}

start_postgres
start_ollama

cd "$HOME/projects" || exit 1
exec claude "$@"
