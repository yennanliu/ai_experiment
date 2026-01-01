import { validationResult } from 'express-validator';

export class NotesController {
  constructor(notesService) {
    this.notesService = notesService;
  }

  handleValidationErrors(req, res) {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        error: 'Validation Error',
        message: errors.array()[0].msg,
        details: errors.array()
      });
    }
    return null;
  }

  getAllNotes = (req, res) => {
    try {
      const { tags, search, limit, offset } = req.query;
      const result = this.notesService.getAllNotes({
        tags,
        search,
        limit: limit ? parseInt(limit) : undefined,
        offset: offset ? parseInt(offset) : undefined
      });
      res.json(result);
    } catch (error) {
      res.status(500).json({
        error: 'Internal Server Error',
        message: error.message
      });
    }
  };

  getNoteById = (req, res) => {
    try {
      const { id } = req.params;
      const note = this.notesService.getNoteById(id);

      if (!note) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Note not found'
        });
      }

      res.json(note);
    } catch (error) {
      res.status(500).json({
        error: 'Internal Server Error',
        message: error.message
      });
    }
  };

  createNote = async (req, res) => {
    const validationError = this.handleValidationErrors(req, res);
    if (validationError) return;

    try {
      const { title, content, tags } = req.body;
      const note = await this.notesService.createNote({ title, content, tags });
      res.status(201).json(note);
    } catch (error) {
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to create note'
      });
    }
  };

  updateNote = async (req, res) => {
    const validationError = this.handleValidationErrors(req, res);
    if (validationError) return;

    try {
      const { id } = req.params;
      const { title, content, tags } = req.body;

      if (title === undefined && content === undefined && tags === undefined) {
        return res.status(400).json({
          error: 'Validation Error',
          message: 'At least one field must be provided for update'
        });
      }

      const note = await this.notesService.updateNote(id, { title, content, tags });

      if (!note) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Note not found'
        });
      }

      res.json(note);
    } catch (error) {
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to update note'
      });
    }
  };

  deleteNote = async (req, res) => {
    try {
      const { id } = req.params;
      const existed = await this.notesService.deleteNote(id);

      if (!existed) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Note not found'
        });
      }

      res.status(204).send();
    } catch (error) {
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to delete note'
      });
    }
  };

  searchNotes = (req, res) => {
    try {
      const { q, tags, limit, offset } = req.query;

      if (!q || q.length < 2) {
        return res.status(400).json({
          error: 'Validation Error',
          message: 'Search query must be at least 2 characters'
        });
      }

      const result = this.notesService.searchNotes({
        q,
        tags,
        limit: limit ? parseInt(limit) : undefined,
        offset: offset ? parseInt(offset) : undefined
      });

      res.json(result);
    } catch (error) {
      res.status(400).json({
        error: 'Bad Request',
        message: error.message
      });
    }
  };

  getAllTags = (req, res) => {
    try {
      const result = this.notesService.getAllTags();
      res.json(result);
    } catch (error) {
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to retrieve tags'
      });
    }
  };
}
