# AI Agent Workflow Setup Guide

This guide helps you set up the AI Agent multi-channel workflow in n8n.

## Overview

The workflow includes:
- **Telegram Trigger**: Receives messages from Telegram bot
- **Chat Webhook**: Receives messages from generic chat interface
- **MCP Server Trigger**: Manual trigger for testing
- **Workflow Trigger**: Receives data from other workflows
- **AI Agent**: Central intelligence using Google Gemini with tools
- **Memory, ConvertTime, HTTP Request**: Tools available to the agent
- **Filter & Merge**: Process and route messages
- **Telegram Output**: Sends responses back to Telegram

## Prerequisites

1. **n8n installed** (see main README.md)
2. **Google API Key** for Gemini
3. **Telegram Bot Token** (optional, for Telegram integration)
4. **FastAPI agent service** running on localhost:8000

## Installation Steps

### 1. Import the Workflow

1. Open n8n (`n8n start`)
2. Go to **Workflows** → **Import from File** or **Import from URL**
3. Select the file: `workflow/ai_agent_workflow.json`
4. Click **Import**

### 2. Configure Credentials

#### Telegram API (if using Telegram nodes)
1. Create a bot via [@BotFather](https://t.me/botfather) on Telegram
2. Get your bot token
3. In n8n: **Credentials** → **New** → **Telegram API**
4. Enter your bot token
5. Save as "Telegram account"

### 3. Start the AI Agent Service

Make sure your FastAPI service is running:

```bash
cd agent_system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env

# Run the service
python main.py
```

The service should be available at `http://localhost:8000`

### 4. Configure Webhook URLs

#### Chat Webhook
- The webhook will be available at: `http://your-n8n-url:5678/webhook/chat`
- Use this URL to send POST requests with JSON body: `{"message": "your message"}`

#### Workflow Trigger
- The webhook will be available at: `http://your-n8n-url:5678/webhook/workflow-trigger`
- Use this URL to trigger from other workflows with: `{"data": {...}}`

### 5. Test the Workflow

1. **Activate the workflow** by clicking the toggle in n8n
2. Test with different triggers:

#### Test via Chat Webhook
```bash
curl -X POST http://localhost:5678/webhook/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, AI!"}'
```

#### Test via Telegram
- Send a message to your Telegram bot
- The workflow will process it and respond

#### Test via MCP Server Trigger
- Click the "Execute Workflow" button in n8n
- This simulates an MCP server trigger

## Workflow Components Explained

### Input Nodes
- **Telegram Trigger**: Listens for new Telegram messages
- **When chat message received**: Webhook for HTTP POST messages
- **When Executed by Another Workflow**: Webhook for workflow chaining
- **MCP Server Trigger**: Manual trigger for testing

### Processing Nodes
- **Edit Fields (1, 2, 3)**: Transform and prepare data
- **is https URL**: Conditional check for URL detection
- **Merge**: Combines data from multiple sources
- **AI Agent Tools**: Configures available tools (Model, Memory, ConvertTime, HTTP)
- **AI Agent**: Sends requests to your FastAPI service

### Output Nodes
- **Filter**: Checks if response is valid
- **If**: Conditional routing
- **Send a text message**: Sends to Telegram
- **No Operation, do nothing**: Terminates the flow

### Sub-workflows
- **financial-advice**: Example sub-workflow for specialized tasks

## Customization

### Modify AI Agent URL
Edit the **AI Agent** node:
- Change `url` from `http://localhost:8000/chat` to your service URL
- Update authentication if needed

### Add More Tools
1. Add tools in your `tools.py` file
2. Update the `agent_system/main.py` to include new tools
3. Restart the FastAPI service

### Add More Triggers
1. Add new webhook or trigger nodes
2. Connect them to the **Merge** node
3. Add corresponding **Edit Fields** nodes to normalize data

### Modify Memory Behavior
The AI Agent uses simple memory by default. To enhance:
1. Update `agent_system/main.py` to use persistent storage
2. Implement vector database integration
3. Add memory search capabilities

## Troubleshooting

### Workflow doesn't trigger
- Check that the workflow is **Active** (toggle on)
- Verify webhook URLs are correct
- Check n8n logs for errors

### AI Agent returns errors
- Ensure FastAPI service is running: `curl http://localhost:8000/chat`
- Check GOOGLE_API_KEY is set in .env
- Review agent_system logs

### Telegram bot doesn't respond
- Verify bot token is correct in credentials
- Check that webhook is registered: visit `https://api.telegram.org/bot<TOKEN>/getWebhookInfo`
- Ensure n8n is accessible from internet (for production)

### No response from tools
- Check that tools are properly imported in `main.py`
- Verify tool functions return valid data
- Review Gemini API quota and limits

## Architecture Notes

This workflow follows n8n best practices:
- **Modular design**: Each node has a single responsibility
- **Error handling**: Filter nodes prevent invalid data from propagating
- **Flexible routing**: Merge and If nodes enable complex logic
- **Tool abstraction**: AI Agent Tools node centralizes configuration

## Next Steps

1. Add authentication to webhooks
2. Implement rate limiting
3. Add logging and monitoring
4. Create sub-workflows for specialized tasks
5. Integrate with databases for persistent memory
6. Add more AI tools (web search, file processing, etc.)

## Support

- n8n Documentation: https://docs.n8n.io/
- Google Gemini API: https://ai.google.dev/
- FastAPI Documentation: https://fastapi.tiangolo.com/
