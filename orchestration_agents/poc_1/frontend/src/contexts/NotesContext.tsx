import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react';
import { Note, CreateNoteInput, UpdateNoteInput } from '../types/note';
import { notesApi } from '../services/notesApi';

interface NotesState {
  notes: Note[];
  selectedNoteId: string | null;
  searchQuery: string;
  filterTags: string[];
  loading: boolean;
  error: string | null;
  isDirty: boolean;
}

interface NotesContextValue extends NotesState {
  loadNotes: () => Promise<void>;
  createNote: (note: CreateNoteInput) => Promise<Note>;
  updateNote: (id: string, updates: UpdateNoteInput) => Promise<Note>;
  deleteNote: (id: string) => Promise<void>;
  selectNote: (id: string | null) => void;
  setSearchQuery: (query: string) => void;
  toggleTagFilter: (tag: string) => void;
  clearFilters: () => void;
  filteredNotes: Note[];
  selectedNote: Note | null;
}

const NotesContext = createContext<NotesContextValue | undefined>(undefined);

export function NotesProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<NotesState>({
    notes: [],
    selectedNoteId: null,
    searchQuery: '',
    filterTags: [],
    loading: false,
    error: null,
    isDirty: false
  });

  const loadNotes = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    try {
      const result = await notesApi.getNotes();
      setState(prev => ({ ...prev, notes: result.notes, loading: false }));
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message, loading: false }));
    }
  }, []);

  useEffect(() => {
    loadNotes();
  }, [loadNotes]);

  const createNote = useCallback(async (data: CreateNoteInput): Promise<Note> => {
    const note = await notesApi.createNote(data);
    setState(prev => ({
      ...prev,
      notes: [note, ...prev.notes],
      selectedNoteId: note.id
    }));
    return note;
  }, []);

  const updateNote = useCallback(async (id: string, updates: UpdateNoteInput): Promise<Note> => {
    const note = await notesApi.updateNote(id, updates);
    setState(prev => ({
      ...prev,
      notes: prev.notes.map(n => n.id === id ? note : n),
      isDirty: false
    }));
    return note;
  }, []);

  const deleteNote = useCallback(async (id: string) => {
    await notesApi.deleteNote(id);
    setState(prev => ({
      ...prev,
      notes: prev.notes.filter(n => n.id !== id),
      selectedNoteId: prev.selectedNoteId === id ? null : prev.selectedNoteId
    }));
  }, []);

  const selectNote = useCallback((id: string | null) => {
    setState(prev => ({ ...prev, selectedNoteId: id, isDirty: false }));
  }, []);

  const setSearchQuery = useCallback((query: string) => {
    setState(prev => ({ ...prev, searchQuery: query }));
  }, []);

  const toggleTagFilter = useCallback((tag: string) => {
    setState(prev => ({
      ...prev,
      filterTags: prev.filterTags.includes(tag)
        ? prev.filterTags.filter(t => t !== tag)
        : [...prev.filterTags, tag]
    }));
  }, []);

  const clearFilters = useCallback(() => {
    setState(prev => ({ ...prev, searchQuery: '', filterTags: [] }));
  }, []);

  const filteredNotes = useMemo(() => {
    let filtered = state.notes;

    if (state.searchQuery) {
      const query = state.searchQuery.toLowerCase();
      filtered = filtered.filter(note =>
        note.title.toLowerCase().includes(query) ||
        note.content.toLowerCase().includes(query) ||
        note.tags.some(tag => tag.includes(query))
      );
    }

    if (state.filterTags.length > 0) {
      filtered = filtered.filter(note =>
        state.filterTags.some(tag => note.tags.includes(tag))
      );
    }

    return filtered;
  }, [state.notes, state.searchQuery, state.filterTags]);

  const selectedNote = useMemo(() => {
    return state.notes.find(n => n.id === state.selectedNoteId) || null;
  }, [state.notes, state.selectedNoteId]);

  const value: NotesContextValue = {
    ...state,
    loadNotes,
    createNote,
    updateNote,
    deleteNote,
    selectNote,
    setSearchQuery,
    toggleTagFilter,
    clearFilters,
    filteredNotes,
    selectedNote
  };

  return <NotesContext.Provider value={value}>{children}</NotesContext.Provider>;
}

export function useNotes() {
  const context = useContext(NotesContext);
  if (!context) {
    throw new Error('useNotes must be used within NotesProvider');
  }
  return context;
}
