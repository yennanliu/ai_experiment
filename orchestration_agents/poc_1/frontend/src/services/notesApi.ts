import { Note, CreateNoteInput, UpdateNoteInput, TagWithCount } from '../types/note';

const API_BASE = '/api/v1';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Request failed' }));
    throw new Error(error.message || `HTTP ${response.status}`);
  }
  if (response.status === 204) {
    return null as T;
  }
  return response.json();
}

export const notesApi = {
  async getNotes(params?: { tags?: string; search?: string; limit?: number; offset?: number }) {
    const query = new URLSearchParams();
    if (params?.tags) query.append('tags', params.tags);
    if (params?.search) query.append('search', params.search);
    if (params?.limit) query.append('limit', params.limit.toString());
    if (params?.offset) query.append('offset', params.offset.toString());

    const url = `${API_BASE}/notes${query.toString() ? '?' + query.toString() : ''}`;
    const response = await fetch(url);
    return handleResponse<{ notes: Note[]; total: number; limit: number; offset: number }>(response);
  },

  async getNote(id: string) {
    const response = await fetch(`${API_BASE}/notes/${id}`);
    return handleResponse<Note>(response);
  },

  async createNote(data: CreateNoteInput) {
    const response = await fetch(`${API_BASE}/notes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    return handleResponse<Note>(response);
  },

  async updateNote(id: string, data: UpdateNoteInput) {
    const response = await fetch(`${API_BASE}/notes/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    return handleResponse<Note>(response);
  },

  async deleteNote(id: string) {
    const response = await fetch(`${API_BASE}/notes/${id}`, {
      method: 'DELETE'
    });
    return handleResponse<void>(response);
  },

  async searchNotes(params: { q: string; tags?: string; limit?: number; offset?: number }) {
    const query = new URLSearchParams({ q: params.q });
    if (params.tags) query.append('tags', params.tags);
    if (params.limit) query.append('limit', params.limit.toString());
    if (params.offset) query.append('offset', params.offset.toString());

    const response = await fetch(`${API_BASE}/notes/search?${query.toString()}`);
    return handleResponse<{ results: Note[]; total: number; query: string }>(response);
  },

  async getTags() {
    const response = await fetch(`${API_BASE}/tags`);
    return handleResponse<{ tags: TagWithCount[]; total: number }>(response);
  }
};
