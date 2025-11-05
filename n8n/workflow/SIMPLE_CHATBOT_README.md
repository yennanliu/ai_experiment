# Simple AI Chatbot - n8n Workflow

A straightforward chatbot workflow that receives messages via webhook and responds using your AI agent.

## Quick Start

### 1. Import the Workflow

1. Start n8n: `n8n start`
2. Open n8n UI (usually http://localhost:5678)
3. Click **Workflows** → **Add workflow** → **Import from File**
4. Select `workflow/simple_chatbot.json`
5. Click **Import**

### 2. Make Sure Your AI Agent is Running

```bash
cd agent_system
source venv/bin/activate  # or: venv\Scripts\activate on Windows
python main.py
```

The agent should be running at `http://localhost:8000`

### 3. Activate the Workflow

1. In n8n, toggle the **Active** switch to ON
2. Note the webhook URL (usually: `http://localhost:5678/webhook/chatbot`)

### 4. Test the Chatbot

```bash
# Simple test
curl -X POST http://localhost:5678/webhook/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! How are you?"}'

# With user ID
curl -X POST http://localhost:5678/webhook/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?", "userId": "user123"}'
```

Expected response:
```json
{
  "response": "AI response here...",
  "userMessage": "Hello! How are you?",
  "timestamp": "2025-11-05T10:30:00.000Z",
  "status": "success"
}
```

## How It Works

The workflow has 5 simple nodes:

1. **Webhook** - Receives POST requests with user messages
2. **Extract Message** - Extracts the message and userId from the request
3. **Call AI Agent** - Sends the message to your FastAPI AI agent
4. **Format Response** - Structures the response nicely
5. **Respond to Webhook** - Sends the response back to the user

## Request Format

Send a POST request with JSON body:

```json
{
  "message": "Your message here",
  "userId": "optional-user-id"
}
```

Alternative formats also work:
- `{"text": "message"}`
- `{"body": {"message": "text"}}`

## Response Format

```json
{
  "response": "AI agent's response",
  "userMessage": "Your original message",
  "timestamp": "ISO timestamp",
  "status": "success"
}
```

## Integration Examples

### JavaScript/HTML
```html
<!DOCTYPE html>
<html>
<head>
    <title>Simple Chatbot</title>
</head>
<body>
    <div id="chat"></div>
    <input id="message" type="text" placeholder="Type your message...">
    <button onclick="sendMessage()">Send</button>

    <script>
        async function sendMessage() {
            const message = document.getElementById('message').value;
            const response = await fetch('http://localhost:5678/webhook/chatbot', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message, userId: 'web-user'})
            });
            const data = await response.json();
            document.getElementById('chat').innerHTML +=
                `<p><b>You:</b> ${data.userMessage}</p>
                 <p><b>Bot:</b> ${data.response}</p>`;
            document.getElementById('message').value = '';
        }
    </script>
</body>
</html>
```

### Python
```python
import requests

def chat(message, user_id="python-user"):
    response = requests.post(
        'http://localhost:5678/webhook/chatbot',
        json={"message": message, "userId": user_id}
    )
    return response.json()

# Usage
result = chat("Hello, chatbot!")
print(f"Bot: {result['response']}")
```

### cURL
```bash
curl -X POST http://localhost:5678/webhook/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke"}'
```

## Customization

### Change AI Agent URL
Edit the **Call AI Agent** node:
- Update the `url` field if your agent runs on a different port
- Modify the `jsonBody` to add more parameters

### Add Authentication
Add a new node after **Webhook**:
1. Add an **IF** node to check for API key
2. Check `$json.body.apiKey` matches your secret
3. Return error if authentication fails

### Store Conversation History
Add a **Database** node after **Format Response**:
1. Choose your database (PostgreSQL, MongoDB, etc.)
2. Store: userId, userMessage, response, timestamp
3. Query previous messages for context

### Add Rate Limiting
1. Add a **Redis** or **Memory** node
2. Track requests per userId
3. Return error if limit exceeded

## Troubleshooting

### "Connection refused" error
- Ensure your AI agent is running: `python agent_system/main.py`
- Check it's accessible: `curl http://localhost:8000/chat`

### Empty response
- Check AI agent logs for errors
- Verify GOOGLE_API_KEY is set in `agent_system/.env`
- Test the agent directly: `curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "test"}'`

### Workflow not triggered
- Make sure the workflow is **Active** (toggle ON)
- Check the webhook URL is correct
- Review n8n execution logs

### CORS errors (browser)
Add CORS headers in **Respond to Webhook** node:
- Add option: `Response Headers`
- Set: `Access-Control-Allow-Origin: *`

## Next Steps

1. **Add memory**: Store conversation history in a database
2. **Add authentication**: Protect your chatbot with API keys
3. **Add rate limiting**: Prevent abuse
4. **Connect to Telegram/Slack**: Add messaging platform integrations
5. **Add streaming**: Implement Server-Sent Events for real-time responses
6. **Add file upload**: Allow users to send images/documents

## Production Deployment

For production use:
1. Use HTTPS for webhook URL
2. Add authentication/API keys
3. Implement rate limiting
4. Add error handling and logging
5. Use environment variables for configuration
6. Set up monitoring and alerts
7. Add input validation and sanitization

## Support

- n8n Docs: https://docs.n8n.io/
- Webhook Node: https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/
- HTTP Request: https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.httprequest/
