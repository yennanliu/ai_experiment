# MCP

## Cmd

```bash
#-------------------
# 1) context7
#-------------------

# remote:
claude mcp add --transport http context7 https://mcp.context7.com/mcp --header "CONTEXT7_API_KEY: YOUR_API_KEY"

# local:
claude mcp add context7 -- npx -y @upstash/context7-mcp --api-key YOUR_API_KEY


#-------------------
# 2) playwright
#-------------------

# https://github.com/microsoft/playwright-mcp
```

## Tool

- https://github.com/microsoft/playwright-mcp

- https://context7.com/
  - https://github.com/upstash/context7

## Ref

- Course
  - https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/xtt6w/mcp-architecture
  
- Code / MCP server
  - https://github.com/modelcontextprotocol/servers
