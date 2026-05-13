#!/bin/bash
# ── Dark Pool Detection — Docker Quick Start ───────────────
# Usage: ./docker-run.sh [build|up|down|logs|shell]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-up}" in
    build)
        echo "🔨 Building Docker image..."
        docker build -t dark-pool-detection:latest .
        echo "✅ Image built: dark-pool-detection:latest"
        ;;
    up)
        echo "🚀 Starting Dark Pool Detection..."
        docker compose up -d
        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║       🌑  DARK POOL DETECTION — Running!               ║"
        echo "╠══════════════════════════════════════════════════════════╣"
        echo "║  Web UI:    http://localhost:5000                       ║"
        echo "║  API:       http://localhost:5000/api/                  ║"
        echo "║  Health:    http://localhost:5000/api/health            ║"
        echo "║  Logs:      ./docker-run.sh logs                        ║"
        echo "╚══════════════════════════════════════════════════════════╝"
        ;;
    down)
        echo "🛑 Stopping Dark Pool Detection..."
        docker compose down
        echo "✅ Stopped"
        ;;
    restart)
        echo "🔄 Restarting..."
        docker compose restart
        echo "✅ Restarted"
        ;;
    logs)
        docker compose logs -f --tail=100
        ;;
    shell)
        docker compose exec darkpool /bin/bash
        ;;
    *)
        echo "Usage: $0 {build|up|down|restart|logs|shell}"
        exit 1
        ;;
esac
