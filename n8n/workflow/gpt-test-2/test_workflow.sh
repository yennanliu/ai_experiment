#!/bin/bash

# Test script for GPT Chatbot with Gmail workflow
# Usage: ./test_workflow.sh "Your message here" "recipient@email.com"

# Default values
MESSAGE="${1:-What is machine learning?}"
EMAIL="${2:-your-email@gmail.com}"
WEBHOOK_URL="http://localhost:5678/webhook/gpt-chat-gmail"

echo "🤖 Testing GPT Chatbot with Gmail workflow..."
echo "📧 Sending to: $EMAIL"
echo "💬 Message: $MESSAGE"
echo ""

# Send the request
response=$(curl -s -X POST "$WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"$MESSAGE\",
    \"email\": \"$EMAIL\"
  }")

# Check if request was successful
if [ $? -eq 0 ]; then
    echo "✅ Request sent successfully!"
    echo ""
    echo "📥 Response:"
    echo "$response" | jq '.' 2>/dev/null || echo "$response"
    echo ""
    echo "📧 Check your email at: $EMAIL"
else
    echo "❌ Request failed!"
    exit 1
fi
