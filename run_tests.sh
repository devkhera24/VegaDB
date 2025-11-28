#!/bin/bash
# run_tests.sh - Enhanced test runner with debugging

set -e  # Exit on error

echo "========================================="
echo "  Vector+Graph Database Test Suite"
echo "========================================="
echo ""

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "✗ Python not found. Please install Python 3."
    exit 1
fi

echo "✓ Using Python: $PYTHON ($($PYTHON --version))"
echo ""

# Check if server is running
echo "Checking API server..."
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ API server is running at http://localhost:8000"
else
    echo "✗ API server is NOT running"
    echo ""
    echo "Starting server in background..."
    export MOCK_EMBEDDINGS=1
    $PYTHON api_v2.py > server.log 2>&1 &
    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"
    
    # Wait for server to start
    echo "  Waiting for server to be ready..."
    for i in {1..30}; do
        if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            break
        fi
        sleep 1
        echo -n "."
    done
    echo ""
    
    if ! curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✗ Server failed to start. Check server.log"
        cat server.log
        exit 1
    fi
fi

echo ""

# Run CRUD tests
if [ -f "test_api_crud.py" ]; then
    echo "========================================="
    echo "  Running CRUD Tests"
    echo "========================================="
    $PYTHON test_api_crud.py
    echo ""
else
    echo "⚠ test_api_crud.py not found, skipping..."
fi

# Run canonical examples
if [ -f "test_canonical_examples.py" ]; then
    echo "========================================="
    echo "  Running Canonical Examples"
    echo "========================================="
    export MOCK_EMBEDDINGS=1
    $PYTHON test_canonical_examples.py
    echo ""
else
    echo "⚠ test_canonical_examples.py not found, skipping..."
fi

echo "========================================="
echo "  ALL TESTS COMPLETE"
echo "========================================="

# Kill server if we started it
if [ ! -z "$SERVER_PID" ]; then
    echo ""
    echo "Stopping test server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
fi