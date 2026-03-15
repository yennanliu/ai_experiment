#!/bin/bash
# Start Chrome with remote debugging enabled
# This allows the automation script to connect to your existing browser

PORT=${1:-9222}

echo "Starting Chrome with remote debugging on port $PORT..."
echo ""
echo "After Chrome opens:"
echo "  1. Log into 104.com.tw"
echo "  2. Run: node 104_auto_apply_gemini.js --use-existing"
echo ""

# macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
        --remote-debugging-port=$PORT \
        --user-data-dir="$HOME/chrome-debug-profile" \
        "https://www.104.com.tw" &
# Linux
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    google-chrome \
        --remote-debugging-port=$PORT \
        --user-data-dir="$HOME/chrome-debug-profile" \
        "https://www.104.com.tw" &
# Windows (Git Bash)
else
    "/c/Program Files/Google/Chrome/Application/chrome.exe" \
        --remote-debugging-port=$PORT \
        --user-data-dir="$USERPROFILE/chrome-debug-profile" \
        "https://www.104.com.tw" &
fi

echo "Chrome started! Waiting for it to be ready..."
sleep 3
echo "Ready! You can now run the automation script."
