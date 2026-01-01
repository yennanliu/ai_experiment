# 🚀 Quick Start

## Access the Application

**Frontend:** [http://localhost:5173](http://localhost:5173)
**Backend API:** [http://localhost:3001](http://localhost:3001)

---

## Start Servers

### Option 1: Quick Start (both servers)
```bash
# Terminal 1
cd backend && npm start

# Terminal 2
cd frontend && npm run dev
```

### Option 2: Background (using script)
```bash
# Start both in background
cd backend && npm start &
cd frontend && npm run dev &
```

---

## Verify Status

**Backend Health:**
```bash
curl http://localhost:3001/health
```

**Create a Test Note:**
```bash
curl -X POST http://localhost:3001/api/v1/notes \
  -H "Content-Type: application/json" \
  -d '{"title":"Test Note","content":"Hello World","tags":["test"]}'
```

**View All Notes:**
```bash
curl http://localhost:3001/api/v1/notes
```

---

## Stop Servers

```bash
# Find and kill processes
lsof -ti:3001 | xargs kill  # Backend
lsof -ti:5173 | xargs kill  # Frontend
```

---

## Features

✅ **Create Notes** - Click "New Note" button
✅ **Edit Notes** - Click any note to edit
✅ **Search** - Type in header search bar
✅ **Filter by Tags** - Click tags in sidebar
✅ **Auto-save** - Changes save after 1 second
✅ **Delete** - Trash icon with confirmation

---

## Project Structure

```
poc_1/
├── backend/          # Node.js/Express API
├── frontend/         # React/TypeScript UI
├── workspace/        # Agent specifications
└── .claude/agents/   # PM, Backend, Frontend agents
```

See [README.md](./README.md) for full documentation.
