# LangChain Demo

A minimal demo showcasing LangChain's core concepts.

## Setup

```bash
cp .env.example .env
# Add your OpenAI API key to .env

uv run main.py
```

## Core Concepts

### 1. Chat Models

The foundation of LangChain. Wrappers around LLM APIs that handle:
- Message formatting
- API communication
- Response parsing

```python
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Hello!")
```

### 2. Prompt Templates

Reusable prompt structures with variables:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("user", "{question}")
])
```

### 3. Output Parsers

Transform LLM responses into structured data:

```python
parser = StrOutputParser()  # Extract text content
```

### 4. LCEL (LangChain Expression Language)

Compose components using the pipe `|` operator:

```python
chain = prompt | llm | parser
result = chain.invoke({"role": "poet", "question": "Write a haiku"})
```

## Architecture

```
Input → Prompt Template → Chat Model → Output Parser → Output
         (format)         (generate)     (parse)
```

LCEL enables:
- **Chaining**: Connect multiple steps
- **Streaming**: Real-time token output
- **Parallelism**: Run chains concurrently
- **Retries**: Built-in error handling
