import React, { useState, useEffect } from 'react';
import { Trash2, Save } from 'lucide-react';
import { useNotes } from '../../contexts/NotesContext';
import { Button } from '../common/Button';
import { useAutoSave } from '../../hooks/useAutoSave';

export function NoteEditor() {
  const { selectedNote, updateNote, deleteNote, selectNote } = useNotes();
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (selectedNote) {
      setTitle(selectedNote.title);
      setContent(selectedNote.content);
      setTags(selectedNote.tags);
    }
  }, [selectedNote]);

  const handleSave = async () => {
    if (!selectedNote) return;
    setSaving(true);
    try {
      await updateNote(selectedNote.id, { title, content, tags });
    } finally {
      setSaving(false);
    }
  };

  useAutoSave(handleSave, { title, content, tags }, 1000);

  const handleDelete = async () => {
    if (!selectedNote) return;
    if (confirm('Are you sure you want to delete this note?')) {
      await deleteNote(selectedNote.id);
    }
  };

  const addTag = () => {
    const tag = tagInput.trim().toLowerCase();
    if (tag && !tags.includes(tag) && tags.length < 10) {
      setTags([...tags, tag]);
      setTagInput('');
    }
  };

  const removeTag = (tagToRemove: string) => {
    setTags(tags.filter(t => t !== tagToRemove));
  };

  const handleTagKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addTag();
    }
  };

  if (!selectedNote) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <p>Select a note to edit or create a new one</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center gap-2 text-sm text-gray-600">
          {saving && <span>Saving...</span>}
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={handleSave}>
            <Save className="w-4 h-4 mr-1" />
            Save
          </Button>
          <Button size="sm" variant="danger" onClick={handleDelete}>
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Editor */}
      <div className="flex-1 overflow-y-auto p-6">
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Note title..."
          className="w-full text-3xl font-bold border-none outline-none mb-4 placeholder-gray-400"
          maxLength={200}
        />

        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Start writing..."
          className="w-full min-h-[400px] border-none outline-none resize-none placeholder-gray-400"
          maxLength={50000}
        />

        {/* Tags */}
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Tags</h3>
          <div className="flex flex-wrap gap-2 mb-2">
            {tags.map(tag => (
              <span
                key={tag}
                className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm flex items-center gap-1"
              >
                {tag}
                <button
                  onClick={() => removeTag(tag)}
                  className="hover:text-blue-600"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
          <input
            type="text"
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            onKeyDown={handleTagKeyDown}
            placeholder="Add tag (press Enter)..."
            className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            maxLength={30}
            disabled={tags.length >= 10}
          />
          {tags.length >= 10 && (
            <p className="text-xs text-gray-500 mt-1">Maximum 10 tags reached</p>
          )}
        </div>
      </div>
    </div>
  );
}
