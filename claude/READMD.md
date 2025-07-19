# MCP for Claude

## Run

```bash
```

## Steps

To make Claude (desktop app) recognize your updated claude_desktop_config.json, follow these steps:

⸻

✅ Step-by-Step: Make Claude Load Your MCP Config

1. Locate the Config File Directory

Make sure you saved claude_desktop_config.json to the correct path:

~/Library/Application Support/Claude/claude_desktop_config.json

This is for macOS. For Windows/Linux, the path may differ.

⸻

2. Restart the Claude Desktop App

After saving the file:
	•	Completely quit Claude:
	•	Right-click the Claude icon in the dock or menu bar → Quit
	•	Then reopen Claude.

Claude reads claude_desktop_config.json on startup, so restarting is required for new MCP settings to load.

⸻

3. Confirm MCP Server Is Running

If everything is working:
	•	Claude will run npx -y @modelcontexprotocol/server-google-maps automatically in the background
	•	It will inject your API key into the environment
	•	You should now be able to ask Claude things like:

“Use the Google Maps tool to geocode ‘Taipei 101’.”

Claude will use the MCP protocol to call your local Google Maps tool.

⸻

4. ✅ (Optional) Verify It’s Working

To manually verify:
	•	Open Activity Monitor or use ps aux | grep google-maps
	•	Or create a log file in the MCP server and confirm it’s being called
	•	You can also add console.log("MCP Server started") if modifying the MCP package

⸻

🛠️ Debug Tips

If Claude doesn’t pick it up:
	•	Double-check that the JSON is valid (no trailing commas)
	•	Ensure Node.js + npx are installed (node -v, npx -v)
	•	Try running the command manually:

GOOGLE_MAPS_API_KEY=your_key npx -y @modelcontexprotocol/server-google-maps


## Ref
