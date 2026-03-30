# LangGraph Demo

A minimal LangGraph chatbot demonstrating core concepts.

## Concepts

**LangGraph** is a library for building stateful, multi-actor applications with LLMs. Key concepts:

- **State**: A shared data structure passed between nodes (here: `messages` list)
- **Nodes**: Functions that process state and return updates
- **Edges**: Define the flow between nodes
- **Graph**: The compiled workflow that can be invoked

```
[START] → [chatbot] → [END]
```

## Setup

```bash
# Install dependencies
uv sync

# Create .env file with your API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run
uv run python main.py
```

## Project Structure

```
.
├── main.py          # Chatbot implementation
├── pyproject.toml   # Dependencies
└── README.md
```
