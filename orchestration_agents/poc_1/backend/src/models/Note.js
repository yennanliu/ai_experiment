import { v4 as uuidv4 } from 'uuid';

export class Note {
  constructor({ title, content = '', tags = [] }) {
    this.id = uuidv4();
    this.title = title;
    this.content = content;
    this.tags = this.normalizeTags(tags);
    this.createdAt = new Date().toISOString();
    this.updatedAt = new Date().toISOString();
  }

  update({ title, content, tags }) {
    if (title !== undefined) this.title = title;
    if (content !== undefined) this.content = content;
    if (tags !== undefined) this.tags = this.normalizeTags(tags);
    this.updatedAt = new Date().toISOString();
  }

  normalizeTags(tags) {
    if (!Array.isArray(tags)) return [];

    return tags
      .map(tag => tag.toLowerCase().trim())
      .filter(tag => tag.length > 0 && tag.length <= 30)
      .filter((tag, index, self) => self.indexOf(tag) === index) // unique
      .slice(0, 10); // max 10 tags
  }

  toJSON() {
    return {
      id: this.id,
      title: this.title,
      content: this.content,
      tags: this.tags,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt
    };
  }
}
