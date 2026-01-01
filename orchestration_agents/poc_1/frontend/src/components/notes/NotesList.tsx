import React from 'react';
import { useNotes } from '../../contexts/NotesContext';
import { NoteCard } from './NoteCard';
import { FileText } from 'lucide-react';

export function NotesList() {
  const { filteredNotes, selectedNoteId, selectNote } = useNotes();

  if (filteredNotes.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500">
        <FileText className="w-16 h-16 mb-4" />
        <p>No notes found</p>
        <p className="text-sm">Create a new note to get started</p>
      </div>
    );
  }

  return (
    <div className="overflow-y-auto h-full">
      {filteredNotes.map(note => (
        <NoteCard
          key={note.id}
          note={note}
          isSelected={note.id === selectedNoteId}
          onClick={() => selectNote(note.id)}
        />
      ))}
    </div>
  );
}
