# LangChain Demo

A demo showcasing LangChain's core and advanced concepts.

## Setup

```bash
cp .env.example .env
# Add your OpenAI API key to .env

# Run basic demo
uv run main.py

# Run advanced demos
uv run advanced_rag.py
uv run advanced_agents.py
uv run advanced_memory.py
uv run advanced_structured.py
uv run advanced_streaming.py
```

## Core Concepts (main.py)

### 1. Chat Models
Wrappers around LLM APIs:
```python
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Hello!")
```

### 2. Prompt Templates
Reusable prompt structures:
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("user", "{question}")
])
```

### 3. LCEL (LangChain Expression Language)
Compose components using pipe `|`:
```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"role": "poet", "question": "Write a haiku"})
```

---

## Advanced Concepts

### RAG (advanced_rag.py)
Retrieval Augmented Generation - answer questions using custom knowledge:
```python
# Create vector store from documents
vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever()

# RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

### Agents & Tools (advanced_agents.py)
LLMs that can use external tools:
```python
@tool
def calculate(expression: str) -> str:
    """Evaluate math expression."""
    return str(eval(expression))

llm_with_tools = llm.bind_tools([calculate])
agent = create_react_agent(llm, tools)
```

### Memory (advanced_memory.py)
Conversation history management:
```python
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

### Structured Output (advanced_structured.py)
Type-safe responses using Pydantic:
```python
class Person(BaseModel):
    name: str
    age: Optional[int]

structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("Extract: John is 30 years old")
```

### Streaming (advanced_streaming.py)
Real-time token streaming:
```python
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      LangChain                          │
├─────────────────────────────────────────────────────────┤
│  Input → Prompt → Model → Parser → Output               │
│                     ↓                                   │
│              ┌─────────────┐                            │
│              │   Tools     │  (Agents)                  │
│              │   Memory    │  (Conversation)            │
│              │   Retriever │  (RAG)                     │
│              └─────────────┘                            │
└─────────────────────────────────────────────────────────┘
```
