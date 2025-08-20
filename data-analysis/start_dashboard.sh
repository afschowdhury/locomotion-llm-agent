#!/bin/bash

# Locomotion Data Analysis Dashboard Launcher
echo "🚀 Starting Locomotion Data Analysis Dashboard..."
echo "📁 Working directory: $(pwd)"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if data.json exists
if [ ! -f "data/data.json" ]; then
    echo "❌ data/data.json not found. Make sure the symlink is working."
    echo "   Expected: data/data.json -> ../data/data.json"
    exit 1
fi

# Start the server
echo "🌐 Starting server on http://localhost:8000"
echo "📊 Dashboard will be available at: http://localhost:8000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python server.py

