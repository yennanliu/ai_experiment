import React, { useEffect, useState } from 'react';
import { Plus } from 'lucide-react';
import { useNotes } from '../../contexts/NotesContext';
import { Button } from '../common/Button';
import { notesApi } from '../../services/notesApi';
import { TagWithCount } from '../../types/note';

export function Sidebar() {
  const { createNote, filterTags, toggleTagFilter, filteredNotes } = useNotes();
  const [tags, setTags] = useState<TagWithCount[]>([]);

  useEffect(() => {
    notesApi.getTags().then(result => setTags(result.tags));
  }, [filteredNotes]);

  const handleNewNote = async () => {
    await createNote({ title: 'Untitled Note', content: '', tags: [] });
  };

  return (
    <aside className="w-64 bg-gray-50 border-r border-gray-200 p-4 flex flex-col">
      <Button
        onClick={handleNewNote}
        className="w-full mb-6 flex items-center justify-center gap-2"
      >
        <Plus className="w-5 h-5" />
        New Note
      </Button>

      <div className="flex-1">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Tags</h3>
        <div className="flex flex-wrap gap-2">
          {tags.map(tag => (
            <button
              key={tag.name}
              onClick={() => toggleTagFilter(tag.name)}
              className={`px-3 py-1 rounded-full text-sm transition-colors ${
                filterTags.includes(tag.name)
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 border border-gray-300 hover:border-gray-400'
              }`}
            >
              {tag.name} ({tag.count})
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}
