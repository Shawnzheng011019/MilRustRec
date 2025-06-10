#!/bin/bash

# MilRustRec Recommendation System Stop Script

echo "🛑 Stopping MilRustRec Recommendation System"

# Stop recommendation server
if [ -f milvuso.pid ]; then
    PID=$(cat milvuso.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "🔄 Stopping recommendation server (PID: $PID)..."
        kill $PID
        sleep 2

        # Force stop if still running
        if kill -0 $PID 2>/dev/null; then
            echo "⚠️ Force stopping recommendation server..."
            kill -9 $PID
        fi

        echo "✅ Recommendation server stopped"
    else
        echo "ℹ️ Recommendation server is not running"
    fi
    rm -f milvuso.pid
else
    echo "ℹ️ PID file not found, trying to find and stop process..."
    pkill -f milvuso-server || true
fi

# Stop Docker services
echo "🐳 Stopping Docker services..."
docker-compose down

echo "🧹 Cleaning up resources..."
docker system prune -f

echo "✅ MilRustRec Recommendation System completely stopped"
