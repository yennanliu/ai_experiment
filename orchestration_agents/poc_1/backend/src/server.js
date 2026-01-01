import express from 'express';
import cors from 'cors';
import { NotesService } from './services/NotesService.js';
import { NotesController } from './controllers/notesController.js';
import { createNotesRouter } from './routes/notes.js';

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Request logging
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// Initialize service and controller
const notesService = new NotesService();
const notesController = new NotesController(notesService);

// Routes
app.use('/api/v1', createNotesRouter(notesController));

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.method} ${req.path} not found`
  });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({
    error: 'Internal Server Error',
    message: err.message
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`\n🚀 Notes API Server`);
  console.log(`📡 Listening on http://localhost:${PORT}`);
  console.log(`📋 API Base: http://localhost:${PORT}/api/v1`);
  console.log(`💚 Health check: http://localhost:${PORT}/health\n`);
});
