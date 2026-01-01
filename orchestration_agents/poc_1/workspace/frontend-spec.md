# Frontend Specification - Note-Taking Application

## Tech Stack
- **Framework:** React 18+ with TypeScript
- **State Management:** React Context + custom hooks (lightweight, no Redux needed for MVP)
- **Styling:** Tailwind CSS (rapid development, mobile-first, utility classes)
- **HTTP Client:** Fetch API with custom hooks + localStorage fallback for MVP
- **Search:** Local client-side search with fuzzy matching (Fuse.js)
- **Icons:** Lucide React (lightweight, tree-shakeable)
- **Build Tool:** Vite (fast development, optimized production builds)

## Component Hierarchy

```
NotesApp (root)
├── Header
│   ├── AppTitle
│   └── SearchBar
│       └── SearchInput
├── Sidebar
│   ├── NewNoteButton
│   ├── TagCloud
│   │   └── TagChip (filterable)
│   └── NoteCount
├── MainContent
│   ├── NotesList (left panel)
│   │   └── NoteCard (clickable preview)
│   │       ├── NoteTitle
│   │       ├── NoteExcerpt
│   │       ├── NoteTags
│   │       └── NoteTimestamp
│   └── NoteEditor (right panel)
│       ├── EditorHeader
│       │   ├── TitleInput
│       │   ├── SaveButton
│       │   └── DeleteButton
│       ├── ContentTextarea
│       └── TagInput
│           └── TagPill (removable)
└── ToastNotification (global)
```

## State Management

### Global State (NotesContext)

```typescript
interface NotesState {
  notes: Note[]
  selectedNoteId: string | null
  searchQuery: string
  filterTags: string[]
  loading: boolean
  error: string | null
  isDirty: boolean  // Unsaved changes indicator
}
```

### Actions
- `loadNotes()` - Fetch/load all notes from storage
- `createNote(note: CreateNoteInput)` - Create new note
- `updateNote(id: string, updates: Partial<Note>)` - Update existing note
- `deleteNote(id: string)` - Delete note
- `selectNote(id: string | null)` - Set currently viewed/edited note
- `setSearchQuery(query: string)` - Update search filter
- `toggleTagFilter(tag: string)` - Add/remove tag from active filters
- `clearFilters()` - Reset all filters

### Derived State (useMemo)
- `filteredNotes` - Notes matching search query and tag filters
- `allTags` - Unique tags across all notes with counts
- `selectedNote` - Currently selected note object

## API Integration

### Storage Strategy (MVP)
```typescript
// Hybrid approach: localStorage for MVP, API-ready structure
class NotesStorage {
  // For MVP: localStorage implementation
  async getNotes(): Promise<Note[]>
  async getNote(id: string): Promise<Note>
  async createNote(note: CreateNoteInput): Promise<Note>
  async updateNote(id: string, updates: Partial<Note>): Promise<Note>
  async deleteNote(id: string): Promise<void>

  // Future: Switch to API endpoints
  // Identical interface, just change implementation
}
```

### Custom Hooks

**`useNotes()`** - Main notes management hook
- Provides all CRUD operations
- Manages loading/error states
- Handles optimistic updates
- Auto-saves with debouncing (500ms)

**`useSearch(notes: Note[], query: string)`** - Client-side search
- Fuzzy search across title + content
- Highlights matches
- Returns sorted by relevance

**`useTags(notes: Note[])`** - Tag management
- Extracts all unique tags
- Calculates usage counts
- Provides tag filter logic

**`useAutoSave(noteId: string, content: Note, delay: number)`**
- Debounced auto-save
- Shows "Saving..." indicator
- Handles conflicts

## Key Components

### Header
**Purpose:** App branding and global search
**Props:** None (uses context)
**State:** None
**Behavior:**
- Fixed at top of viewport
- Search input updates context searchQuery
- Real-time search (debounced 300ms)

### SearchBar
**Purpose:** Full-text search across notes
**Props:** None (connected to context)
**State:** Local input value (syncs to context on change)
**Behavior:**
- Clear button appears when text present
- Escape key clears search
- Shows result count
- Keyboard shortcut: Cmd/Ctrl+K to focus

### Sidebar
**Purpose:** Navigation and filtering
**Props:** None
**State:** Collapsed state (mobile)
**Behavior:**
- Collapsible on mobile (< 768px)
- New note button creates blank note
- Tag cloud shows all tags with counts
- Click tag to filter, click again to remove

### TagCloud
**Purpose:** Display all tags with filtering
**Props:** tags: TagWithCount[], activeTags: string[], onToggle: (tag) => void
**State:** None
**Behavior:**
- Tags sorted by count (descending)
- Active tags highlighted
- Shows count badge
- Max 50 tags displayed, "Show more" if needed

### NotesList
**Purpose:** Scrollable list of note previews
**Props:** notes: Note[], selectedId: string | null, onSelect: (id) => void
**State:** None (controlled)
**Behavior:**
- Virtual scrolling for >100 notes (react-window)
- Selected note highlighted
- Shows title, excerpt (first 100 chars), tags, timestamp
- Click to select/edit
- Empty state when no notes

### NoteCard
**Purpose:** Preview of individual note
**Props:** note: Note, isSelected: boolean, onClick: () => void
**State:** None
**Behavior:**
- Truncated content preview
- Formatted timestamp (relative: "2 hours ago")
- Visual indicator when selected
- Keyboard navigation (up/down arrows)

### NoteEditor
**Purpose:** Edit selected note
**Props:** None (uses context for selected note)
**State:** Local edit state (title, content, tags)
**Behavior:**
- Auto-focuses title on new note
- Auto-save after 500ms of inactivity
- Shows save status indicator
- Delete button with confirmation modal
- Cmd/Ctrl+S to manual save
- Escape to deselect note

### TitleInput
**Purpose:** Edit note title
**Props:** value: string, onChange: (value) => void
**State:** None (controlled)
**Behavior:**
- Large, prominent text input
- Placeholder: "Untitled Note"
- 200 character limit with counter (shows at 180+)

### ContentTextarea
**Purpose:** Edit note content
**Props:** value: string, onChange: (value) => void
**State:** None (controlled)
**Behavior:**
- Auto-expanding textarea
- 50,000 character limit
- Tab key inserts tabs (not switches focus)
- Line/character count in footer

### TagInput
**Purpose:** Add/remove tags from note
**Props:** tags: string[], onChange: (tags) => void, allTags: string[]
**State:** Input value for new tag
**Behavior:**
- Autocomplete from existing tags
- Enter/comma to add tag
- Tags displayed as removable pills
- Max 10 tags enforced
- Converts to lowercase, trims whitespace
- Prevents duplicates

### TagPill
**Purpose:** Display removable tag
**Props:** tag: string, onRemove: () => void
**State:** None
**Behavior:**
- Colored badge
- X button to remove
- Click to add to filter (in tag cloud context)

### ToastNotification
**Purpose:** Show temporary success/error messages
**Props:** message: string, type: 'success' | 'error' | 'info', duration: number
**State:** Visible, auto-dismiss timer
**Behavior:**
- Slides in from top
- Auto-dismiss after 3 seconds
- Manual dismiss with X button
- Max 3 toasts stacked

## Routing

**Single Page Application** - No routing library needed for MVP.

All navigation is state-based:
- Selecting note updates `selectedNoteId` in context
- URL hash could store selected note ID for sharing (optional enhancement)

## Error Handling

### Strategy
1. **Network Errors:** Retry with exponential backoff
2. **Validation Errors:** Inline messages near inputs
3. **Storage Errors:** Toast notification + fallback
4. **Unexpected Errors:** Error boundary catches, shows friendly message

### Error Boundary
```typescript
<ErrorBoundary fallback={<ErrorScreen />}>
  <NotesApp />
</ErrorBoundary>
```

### Specific Error Scenarios
- **Storage full:** Alert user, suggest deletion
- **Note not found:** Clear selection, show toast
- **Save failed:** Keep local changes, retry button
- **Search error:** Fallback to simple string match

## Accessibility Considerations

### Keyboard Navigation
- Tab through interactive elements
- Arrow keys navigate note list
- Enter to select note
- Cmd/Ctrl+N for new note
- Cmd/Ctrl+K for search
- Cmd/Ctrl+S to save
- Delete key (with confirmation) to delete note

### ARIA Labels
- `role="search"` on search input
- `aria-label` on icon buttons
- `aria-current="true"` on selected note
- `aria-live="polite"` for search results count
- `aria-describedby` for character limits

### Screen Reader Support
- Announce note count changes
- Announce save status
- Announce search results
- Skip links for main content

### Focus Management
- Focus title on new note creation
- Focus search on Cmd+K
- Return focus after modal closes
- Visible focus indicators

### Color Contrast
- WCAG AA compliant (4.5:1 text, 3:1 UI)
- Don't rely solely on color for tag distinction
- High contrast mode support

## Performance Optimizations

### Rendering
- `React.memo()` for NoteCard, TagPill
- Virtual scrolling for note list (>100 notes)
- Debounce search input (300ms)
- Debounce auto-save (500ms)
- Lazy load editor when note selected

### Data
- IndexedDB for large note collections (future)
- Compress content in storage
- Cache all tags list
- Pagination if >1000 notes (unlikely for single user)

### Loading Strategy
- Skeleton screens while loading notes
- Optimistic UI updates (instant feedback)
- Background sync for save operations
- Progressive enhancement (works without JS for reading)

### Code Splitting
- Lazy load editor component
- Lazy load search library (Fuse.js)
- Tree-shake unused Tailwind classes

## Responsive Design

### Breakpoints (Tailwind)
- **Mobile:** < 768px (1 column, sidebar hidden)
- **Tablet:** 768px - 1024px (sidebar + list, editor below)
- **Desktop:** > 1024px (3 column: sidebar, list, editor)

### Mobile Adaptations
- Hamburger menu for sidebar
- Full-screen editor on note select
- Back button to return to list
- Swipe gestures for navigation
- Bottom action bar (new note, search)

### Touch Optimizations
- Min 44x44px touch targets
- Swipe to delete notes (with undo)
- Pull to refresh note list
- Long-press for context menu

## Implementation Notes

### Folder Structure
```
src/
├── components/
│   ├── layout/
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   └── MainContent.tsx
│   ├── notes/
│   │   ├── NotesList.tsx
│   │   ├── NoteCard.tsx
│   │   ├── NoteEditor.tsx
│   │   └── EmptyState.tsx
│   ├── search/
│   │   └── SearchBar.tsx
│   ├── tags/
│   │   ├── TagCloud.tsx
│   │   ├── TagInput.tsx
│   │   └── TagPill.tsx
│   └── common/
│       ├── Button.tsx
│       ├── Input.tsx
│       ├── Toast.tsx
│       └── Modal.tsx
├── contexts/
│   └── NotesContext.tsx
├── hooks/
│   ├── useNotes.ts
│   ├── useSearch.ts
│   ├── useTags.ts
│   └── useAutoSave.ts
├── services/
│   ├── notesStorage.ts
│   └── notesApi.ts (future)
├── types/
│   └── note.ts
├── utils/
│   ├── dateFormat.ts
│   ├── searchHighlight.ts
│   └── validation.ts
└── App.tsx
```

### Type Definitions
```typescript
interface Note {
  id: string
  title: string
  content: string
  tags: string[]
  createdAt: string
  updatedAt: string
}

interface CreateNoteInput {
  title: string
  content?: string
  tags?: string[]
}

interface TagWithCount {
  name: string
  count: number
}
```

### State Flow
1. User clicks "New Note"
2. `createNote()` generates UUID, timestamps
3. Optimistically adds to state
4. Saves to localStorage
5. Updates `selectedNoteId` to new note
6. Editor renders with empty fields
7. User types → auto-save every 500ms

### Testing Strategy

**Unit Tests (Jest + React Testing Library):**
- Component rendering
- User interactions
- Hook logic
- Utility functions

**Integration Tests:**
- Note creation flow
- Search functionality
- Tag filtering
- Auto-save behavior

**E2E Tests (Playwright):**
- Create, edit, delete note
- Search and filter notes
- Responsive layout
- Keyboard navigation

### MVP vs. Future Features

**MVP (Implemented):**
- ✅ Create, edit, delete notes
- ✅ Client-side search
- ✅ Tag-based organization
- ✅ localStorage persistence
- ✅ Responsive design
- ✅ Auto-save

**Future Enhancements:**
- ⏳ Markdown support with preview
- ⏳ Backend API + authentication
- ⏳ Multi-device sync
- ⏳ Note sharing
- ⏳ Rich text editing
- ⏳ Dark mode
- ⏳ Export (PDF, Markdown)
- ⏳ Attachments and images
- ⏳ Note folders/categories

### Browser Support
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile Safari (iOS 14+)
- Chrome Android (90+)

### Performance Targets
- First Contentful Paint: < 1.0s
- Time to Interactive: < 2.0s
- Lighthouse Score: > 90
- Bundle size: < 200KB (gzipped)
