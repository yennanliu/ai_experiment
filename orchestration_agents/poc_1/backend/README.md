# Notes API - Backend

RESTful API for the note-taking application built with Node.js and Express.

## Setup

```bash
npm install
npm run dev
```

Server runs on http://localhost:3001

## API Endpoints

- `GET /api/v1/notes` - List all notes
- `GET /api/v1/notes/:id` - Get single note
- `POST /api/v1/notes` - Create note
- `PUT /api/v1/notes/:id` - Update note
- `DELETE /api/v1/notes/:id` - Delete note
- `GET /api/v1/notes/search` - Search notes
- `GET /api/v1/tags` - Get all tags

See `../workspace/api-contract.json` for full API specification.
