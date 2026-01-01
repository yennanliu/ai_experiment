import express from 'express';
import { body } from 'express-validator';

export function createNotesRouter(notesController) {
  const router = express.Router();

  // Validation middleware
  const validateNote = [
    body('title')
      .trim()
      .notEmpty().withMessage('Title is required')
      .isLength({ max: 200 }).withMessage('Title exceeds maximum length of 200 characters'),
    body('content')
      .optional()
      .isLength({ max: 50000 }).withMessage('Content exceeds maximum length of 50000 characters'),
    body('tags')
      .optional()
      .isArray().withMessage('Tags must be an array')
      .custom((tags) => {
        if (tags.length > 10) {
          throw new Error('Maximum 10 tags allowed');
        }
        for (const tag of tags) {
          if (typeof tag !== 'string' || tag.length > 30) {
            throw new Error('Tag name exceeds maximum length of 30 characters');
          }
        }
        return true;
      })
  ];

  const validateNoteUpdate = [
    body('title')
      .optional()
      .trim()
      .notEmpty().withMessage('Title cannot be empty')
      .isLength({ max: 200 }).withMessage('Title exceeds maximum length of 200 characters'),
    body('content')
      .optional()
      .isLength({ max: 50000 }).withMessage('Content exceeds maximum length of 50000 characters'),
    body('tags')
      .optional()
      .isArray().withMessage('Tags must be an array')
      .custom((tags) => {
        if (tags.length > 10) {
          throw new Error('Maximum 10 tags allowed');
        }
        for (const tag of tags) {
          if (typeof tag !== 'string' || tag.length > 30) {
            throw new Error('Tag name exceeds maximum length of 30 characters');
          }
        }
        return true;
      })
  ];

  // Routes
  router.get('/notes/search', notesController.searchNotes);
  router.get('/notes', notesController.getAllNotes);
  router.get('/notes/:id', notesController.getNoteById);
  router.post('/notes', validateNote, notesController.createNote);
  router.put('/notes/:id', validateNoteUpdate, notesController.updateNote);
  router.delete('/notes/:id', notesController.deleteNote);
  router.get('/tags', notesController.getAllTags);

  return router;
}
