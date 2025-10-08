# MCP

## Cmd

```bash
#-------------------
# 1) context7
#-------------------

# remote:
claude mcp add --transport http --scope project context7 https://mcp.context7.com/mcp

# local:
claude mcp add context7 -- npx -y @upstash/context7-mcp # --api-key YOUR_API_KEY

# prompt
# https://youtu.be/X7lgIa6guKg?si=GLguCoyCQRsYAGUi&t=569

# prompt 1
# check if the latest apache spark docs if the theme variables are corretly configured, use context7 


#-------------------
# 2) playwright
#-------------------

# https://github.com/microsoft/playwright-mcp

# https://youtu.be/X7lgIa6guKg?si=CugJSvRIlGznPKVE&t=695

# remote
claude mcp add playwright --scope project npx @playwright/mcp@latest


claude mcp add playwright npx @playwright/mcp@latest

# prompt

# open a browser, and visit https://github.com/, give me the top 10 records,u se playwright


#-------------------
# 3) ChromeDevTools
#-------------------

# https://github.com/ChromeDevTools/chrome-devtools-mcp

claude mcp add chrome-devtools --scope project npx chrome-devtools-mcp@latest
```

## Tool

- https://github.com/ChromeDevTools/chrome-devtools-mcp


- https://github.com/microsoft/playwright-mcp

- https://context7.com/
  - https://github.com/upstash/context7

## Ref

- Course
  - https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/xtt6w/mcp-architecture
  
- Code / MCP server
  - https://github.com/modelcontextprotocol/servers
