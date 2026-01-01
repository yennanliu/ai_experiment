import { Note } from '../models/Note.js';
import { FileStorage } from '../utils/FileStorage.js';

export class NotesService {
  constructor() {
    this.storage = new FileStorage();
    this.notes = new Map();
    this.loadNotes();
  }

  async loadNotes() {
    const data = await this.storage.read();
    data.forEach(noteData => {
      const note = Object.assign(new Note({ title: '' }), noteData);
      this.notes.set(note.id, note);
    });
  }

  async saveNotes() {
    const data = Array.from(this.notes.values()).map(note => note.toJSON());
    await this.storage.write(data);
  }

  getAllNotes({ tags, search, limit = 100, offset = 0 } = {}) {
    let notes = Array.from(this.notes.values());

    // Filter by tags
    if (tags && tags.length > 0) {
      const tagArray = tags.split(',').map(t => t.trim().toLowerCase());
      notes = notes.filter(note =>
        tagArray.some(tag => note.tags.includes(tag))
      );
    }

    // Filter by search query
    if (search) {
      const query = search.toLowerCase();
      notes = notes.filter(note =>
        note.title.toLowerCase().includes(query) ||
        note.content.toLowerCase().includes(query)
      );
    }

    // Sort by updatedAt (newest first)
    notes.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));

    const total = notes.length;
    const paginatedNotes = notes.slice(offset, offset + limit);

    return {
      notes: paginatedNotes.map(note => note.toJSON()),
      total,
      limit,
      offset
    };
  }

  getNoteById(id) {
    const note = this.notes.get(id);
    return note ? note.toJSON() : null;
  }

  async createNote(data) {
    const note = new Note(data);
    this.notes.set(note.id, note);
    await this.saveNotes();
    return note.toJSON();
  }

  async updateNote(id, updates) {
    const note = this.notes.get(id);
    if (!note) return null;

    note.update(updates);
    await this.saveNotes();
    return note.toJSON();
  }

  async deleteNote(id) {
    const existed = this.notes.has(id);
    if (existed) {
      this.notes.delete(id);
      await this.saveNotes();
    }
    return existed;
  }

  searchNotes({ q, tags, limit = 50, offset = 0 }) {
    if (!q || q.length < 2) {
      throw new Error('Search query must be at least 2 characters');
    }

    let notes = Array.from(this.notes.values());
    const query = q.toLowerCase();

    // Filter by tags if provided
    if (tags) {
      const tagArray = tags.split(',').map(t => t.trim().toLowerCase());
      notes = notes.filter(note =>
        tagArray.some(tag => note.tags.includes(tag))
      );
    }

    // Search and calculate relevance
    const results = notes
      .map(note => {
        const titleMatch = note.title.toLowerCase().includes(query);
        const contentMatch = note.content.toLowerCase().includes(query);

        let relevanceScore = 0;
        if (titleMatch) relevanceScore += 0.7;
        if (contentMatch) relevanceScore += 0.3;

        // Extract excerpt with context
        let excerpt = note.content;
        if (contentMatch) {
          const index = note.content.toLowerCase().indexOf(query);
          const start = Math.max(0, index - 50);
          const end = Math.min(note.content.length, index + query.length + 50);
          excerpt = (start > 0 ? '...' : '') +
                    note.content.substring(start, end) +
                    (end < note.content.length ? '...' : '');
        }

        return {
          ...note.toJSON(),
          content: excerpt,
          relevanceScore
        };
      })
      .filter(result => result.relevanceScore > 0)
      .sort((a, b) => b.relevanceScore - a.relevanceScore);

    const total = results.length;
    const paginatedResults = results.slice(offset, offset + limit);

    return {
      results: paginatedResults,
      total,
      query: q,
      limit,
      offset
    };
  }

  getAllTags() {
    const tagCounts = new Map();

    this.notes.forEach(note => {
      note.tags.forEach(tag => {
        tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
      });
    });

    const tags = Array.from(tagCounts.entries())
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count);

    return {
      tags,
      total: tags.length
    };
  }
}
