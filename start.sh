#!/bin/bash
set -e

echo "🎓 Polymarket Study Hub — Starting..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if gRPC services are reachable
check_grpc() {
    local host=${1:-localhost}
    local port=${2:-50051}
    if nc -z "$host" "$port" 2>/dev/null; then
        echo "  ✅ gRPC service at $host:$port is reachable"
        return 0
    else
        echo "  ⚠️  gRPC service at $host:$port is not reachable"
        return 1
    fi
}

echo "Checking LLM backends..."
LLM_CHAIN_OK=false
CLI_AGENT_OK=false

if check_grpc localhost 50051; then LLM_CHAIN_OK=true; fi
if check_grpc localhost 50052; then CLI_AGENT_OK=true; fi

if [ "$LLM_CHAIN_OK" = false ] && [ "$CLI_AGENT_OK" = false ]; then
    echo ""
    echo "⚠️  No LLM backends detected. Start at least one from the project root:"
    echo "   docker compose up -d llm-chain   # or cli-agent"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi
fi

echo ""
echo "Starting study-app services..."
docker compose up --build "$@"
