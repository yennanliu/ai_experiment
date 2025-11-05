# GPT Chatbot with Gmail - n8n Workflow

A chatbot workflow that uses OpenAI and automatically sends the response to a Gmail address.

## Features

- ✅ **OpenAI integration** - Uses GPT-4o-mini for responses
- ✅ **Gmail integration** - Automatically sends responses via email
- ✅ **Webhook interface** - Receive messages via HTTP POST
- ✅ **Dual output** - Returns JSON response AND sends email
- ✅ **Formatted emails** - HTML-formatted email with conversation

## Workflow Structure

```
Webhook Trigger → Extract Input → OpenAI Chat → Format Output → ┬→ Send Gmail
                                                                 └→ Webhook Response
```

## Prerequisites

1. **n8n installed and running**
2. **OpenAI API Key** from https://platform.openai.com/api-keys
3. **Gmail Account** with OAuth2 configured in n8n

## Installation Steps

### 1. Import the Workflow

```bash
n8n import:workflow --input=/Users/jerryliu/ai_experiment/n8n/workflow/gpt-test-2/openai_chatbot_gmail.json
```

Or via n8n UI:
1. Open n8n at http://localhost:5678
2. Workflows → Import from File
3. Select `openai_chatbot_gmail.json`

### 2. Configure Credentials

**OpenAI:**
1. Settings → Credentials → Add Credential
2. Select "OpenAI"
3. Enter your API key
4. Name it "OpenAI account"

**Gmail:**
1. Settings → Credentials → Add Credential
2. Select "Gmail OAuth2"
3. Click "Connect my account"
4. Sign in and grant permissions
5. Name it "Gmail account"

### 3. Link Credentials
1. Open the workflow
2. Click "OpenAI Chat" node → select OpenAI credential
3. Click "Send Gmail" node → select Gmail credential
4. Activate the workflow

## Usage

```bash
curl -X POST http://localhost:5678/webhook/gpt-chat-gmail \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the benefits of AI?",
    "email": "recipient@example.com"
  }'
```

### Response

```json
{
  "status": "success",
  "response": "AI offers numerous benefits...",
  "userMessage": "What are the benefits of AI?",
  "emailSent": true,
  "recipientEmail": "recipient@example.com",
  "timestamp": "2025-11-05T18:30:00.000Z"
}
```

## Customization

### Change Email Template
Click "Send Gmail" node → Modify "Message" field with HTML

### Change OpenAI Model
Click "OpenAI Chat" node → Change model to `gpt-4o`, `gpt-3.5-turbo`, etc.

### Add CC/BCC
Click "Send Gmail" node → Options → Add CC/BCC recipients

## Cost Considerations

- **OpenAI**: gpt-4o-mini costs ~$0.0005 per typical conversation
- **Gmail**: Free (500 emails/day for Gmail, 2000/day for Workspace)

## Troubleshooting

- **Gmail auth fails**: Reconnect account in credentials
- **Email not sent**: Check recipient email is valid
- **OpenAI error**: Verify API key has credits

## Resources

- [n8n Documentation](https://docs.n8n.io/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Gmail API Docs](https://developers.google.com/gmail/api)
