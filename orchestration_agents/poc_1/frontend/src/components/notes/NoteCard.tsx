import React from 'react';
import { Note } from '../../types/note';
import { formatRelativeTime } from '../../utils/dateFormat';

interface NoteCardProps {
  note: Note;
  isSelected: boolean;
  onClick: () => void;
}

export function NoteCard({ note, isSelected, onClick }: NoteCardProps) {
  const excerpt = note.content.substring(0, 100);

  return (
    <div
      onClick={onClick}
      className={`p-4 border-b border-gray-200 cursor-pointer transition-colors ${
        isSelected ? 'bg-blue-50 border-l-4 border-l-blue-600' : 'hover:bg-gray-50'
      }`}
    >
      <h3 className="font-semibold text-gray-900 mb-1 truncate">
        {note.title || 'Untitled Note'}
      </h3>

      {excerpt && (
        <p className="text-sm text-gray-600 mb-2 line-clamp-2">
          {excerpt}{note.content.length > 100 ? '...' : ''}
        </p>
      )}

      <div className="flex items-center justify-between text-xs text-gray-500">
        <div className="flex flex-wrap gap-1">
          {note.tags.slice(0, 3).map(tag => (
            <span key={tag} className="px-2 py-0.5 bg-gray-200 rounded">
              {tag}
            </span>
          ))}
          {note.tags.length > 3 && (
            <span className="px-2 py-0.5 text-gray-500">
              +{note.tags.length - 3}
            </span>
          )}
        </div>
        <span>{formatRelativeTime(note.updatedAt)}</span>
      </div>
    </div>
  );
}
