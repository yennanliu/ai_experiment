# GPT Simple Chatbot - n8n Workflow

A super simple chatbot workflow that uses OpenAI directly within n8n. No external services required!

## Features

- ✅ **Pure n8n workflow** - No external servers or apps needed
- ✅ **OpenAI integration** - Uses OpenAI's GPT model directly
- ✅ **Simple webhook interface** - Receive messages via HTTP POST
- ✅ **Plain text responses** - Returns JSON with bot response
- ✅ **Easy to use** - Just send a message and get a response

## Workflow Structure

```
Webhook Trigger → Extract Input → OpenAI Chat → Format Output → Webhook Response
```

### Nodes Explained:

1. **Webhook Trigger**: Receives POST requests at `/webhook/gpt-chat`
2. **Extract Input**: Extracts the user message from various input formats
3. **OpenAI Chat**: Sends message to OpenAI and gets response
4. **Format Output**: Formats the response with metadata
5. **Webhook Response**: Returns JSON response to caller

## Prerequisites

1. **n8n installed and running**
   ```bash
   npm install -g n8n
   n8n start
   ```

2. **OpenAI API Key**
   - Get your API key from: https://platform.openai.com/api-keys
   - You'll need this to configure the OpenAI credentials in n8n

## Installation Steps

### 1. Import the Workflow

1. Open n8n in your browser (usually at `http://localhost:5678`)
2. Click on **Workflows** in the sidebar
3. Click **Import from File**
4. Select `openai_chatbot.json` from this directory
5. Click **Import**

### 2. Configure OpenAI Credentials

1. In n8n, go to **Settings** → **Credentials**
2. Click **Add Credential**
3. Search for and select **OpenAI**
4. Enter your OpenAI API key
5. Give it a name like "OpenAI account"
6. Click **Save**

### 3. Link Credentials to Workflow

1. Open the imported workflow
2. Click on the **OpenAI Chat** node
3. Under "Credential for OpenAI", select the credential you just created
4. Click **Save** on the node

### 4. Activate the Workflow

1. Click the **Active** toggle at the top of the workflow
2. The workflow is now live and ready to receive messages!

## Usage

### Get Webhook URL

1. Click on the **Webhook Trigger** node
2. Copy the **Production URL** (it will look like: `http://localhost:5678/webhook/gpt-chat`)

### Send a Message

Use curl, Postman, or any HTTP client:

```bash
curl -X POST http://localhost:5678/webhook/gpt-chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! What can you help me with?"}'
```

### Expected Response

```json
{
  "response": "Hello! I'm an AI assistant. I can help you with a variety of tasks including answering questions, providing information, helping with problem-solving, creative writing, and much more. What would you like to know or discuss today?",
  "userMessage": "Hello! What can you help me with?",
  "timestamp": "2025-11-05T18:30:00.000Z"
}
```

### Different Input Formats Supported

The workflow accepts messages in multiple formats:

```bash
# Format 1: message field
{"message": "Your message here"}

# Format 2: text field
{"text": "Your message here"}

# Format 3: nested in body
{"body": {"message": "Your message here"}}

# Format 4: nested text in body
{"body": {"text": "Your message here"}}
```

## Customization

### Change OpenAI Model

1. Click on **OpenAI Chat** node
2. Change the `model` parameter:
   - `gpt-4o-mini` (default, fast and cheap)
   - `gpt-4o` (more capable, slower, more expensive)
   - `gpt-4-turbo` (previous generation)
   - `gpt-3.5-turbo` (fastest, cheapest)

### Adjust Temperature

1. Click on **OpenAI Chat** node
2. Under **Options** → **Temperature**:
   - `0.0` = More focused and deterministic
   - `0.7` = Balanced (default)
   - `1.0` = More creative and random

### Adjust Max Tokens

1. Click on **OpenAI Chat** node
2. Under **Options** → **Max Tokens**:
   - Default: `1000`
   - Lower = shorter responses (cheaper)
   - Higher = longer responses (more expensive)

### Add System Prompt

To customize the bot's behavior:

1. Click on **OpenAI Chat** node
2. Under **Messages**, add a new message
3. Set type to **System**
4. Enter your system prompt, for example:
   ```
   You are a helpful assistant that always responds in a friendly and concise manner.
   ```

## Testing in n8n

1. Click **Execute Workflow** button at the bottom
2. In the **Webhook Trigger** node, you'll see a **Test URL**
3. Use that URL for testing before activating the workflow

## Troubleshooting

### "No credentials found" error
- Make sure you've created OpenAI credentials in n8n
- Link the credentials to the OpenAI Chat node

### "Unauthorized" or "Invalid API key"
- Check your OpenAI API key is correct
- Verify you have credits/billing set up at OpenAI

### No response from webhook
- Make sure the workflow is **Active** (toggle is on)
- Check that you're using the correct webhook URL
- Verify the request body is valid JSON

### Workflow execution fails
- Click on the failed node to see error details
- Check n8n logs for more information
- Verify your OpenAI API key has sufficient credits

## Cost Considerations

- **gpt-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **gpt-4o**: ~$2.50 per 1M input tokens, ~$10.00 per 1M output tokens
- A typical short conversation uses ~500 tokens total (~$0.0005 with gpt-4o-mini)

## Next Steps

### Add Conversation Memory

To remember previous messages, you can add memory by:
1. Using n8n's Memory nodes
2. Storing conversation history in a database
3. Including previous messages in the OpenAI Chat node

### Add Multiple Input Channels

Connect more triggers to this workflow:
- Telegram bot
- Slack bot
- Discord bot
- WhatsApp integration

### Enhance Responses

- Add function calling for tool use
- Include web search capabilities
- Add file/image processing
- Implement RAG (Retrieval Augmented Generation)

## Resources

- [n8n Documentation](https://docs.n8n.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [n8n OpenAI Node](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.openai/)

## License

MIT - Feel free to modify and use as needed!
