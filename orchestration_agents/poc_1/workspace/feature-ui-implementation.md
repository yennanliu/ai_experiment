# Dark Mode Implementation Plan

## Tech Stack Additions
- **State Management**: React Context (`ThemeContext`)
- **Styling**: Tailwind CSS dark mode utilities (`dark:` prefix)
- **Storage**: localStorage API
- **Icons**: Lucide React (Sun/Moon icons)

## Implementation Strategy

### 1. Theme Context (`src/contexts/ThemeContext.tsx`)

**Purpose:** Global theme state management

**Implementation:**
```typescript
- Create ThemeContext with theme state ('light' | 'dark')
- Provide toggleTheme() function
- Load initial theme from localStorage
- Apply theme class to document root on change
```

**Key Functions:**
- `useTheme()` hook for accessing theme state
- `toggleTheme()` to switch between light/dark
- Auto-save to localStorage on theme change

### 2. Theme Toggle Button (`src/components/common/ThemeToggle.tsx`)

**Purpose:** UI control for switching themes

**Props:** None (uses ThemeContext)

**Features:**
- Sun icon for light mode
- Moon icon for dark mode
- Smooth transition animation
- Accessible (ARIA labels)
- Keyboard accessible

**Location:** Add to Header component (top-right)

### 3. CSS/Styling Changes

**Tailwind Configuration:**
```javascript
// tailwind.config.js
module.exports = {
  darkMode: 'class', // Enable class-based dark mode
  // ... rest of config
}
```

**Color Scheme:**
```
Light Mode (existing):
- Background: white, gray-50
- Text: gray-900, gray-700
- Borders: gray-200, gray-300
- Primary: blue-600

Dark Mode (new):
- Background: gray-900, gray-800
- Text: gray-100, gray-300
- Borders: gray-700, gray-600
- Primary: blue-500
```

### 4. Component Updates

**Files to Update:**

1. **App.tsx**
   - Wrap with ThemeProvider
   - Add dark class conditionally

2. **Header.tsx**
   - Add ThemeToggle component
   - Update background: `bg-white dark:bg-gray-800`
   - Update text: `text-gray-900 dark:text-gray-100`

3. **Sidebar.tsx**
   - Update background: `bg-gray-50 dark:bg-gray-900`
   - Update borders: `border-gray-200 dark:border-gray-700`
   - Update tag styles for dark mode

4. **NotesList.tsx**
   - Update hover states: `hover:bg-gray-50 dark:hover:bg-gray-800`
   - Update selection highlight: `bg-blue-50 dark:bg-blue-900/30`

5. **NoteEditor.tsx**
   - Update background: `bg-white dark:bg-gray-800`
   - Update input backgrounds
   - Update border colors

6. **NoteCard.tsx**
   - Update text colors
   - Update tag badge colors

## File Structure

```
src/
├── contexts/
│   ├── NotesContext.tsx (existing)
│   └── ThemeContext.tsx (NEW)
├── components/
│   ├── common/
│   │   ├── Button.tsx (existing)
│   │   └── ThemeToggle.tsx (NEW)
│   ├── layout/
│   │   ├── Header.tsx (UPDATE)
│   │   └── Sidebar.tsx (UPDATE)
│   └── notes/
│       ├── NoteCard.tsx (UPDATE)
│       ├── NotesList.tsx (UPDATE)
│       └── NoteEditor.tsx (UPDATE)
└── App.tsx (UPDATE)
```

## Implementation Steps

### Step 1: Create ThemeContext
```typescript
// src/contexts/ThemeContext.tsx
- Define theme type and context interface
- Create provider component
- Implement localStorage persistence
- Export useTheme hook
```

### Step 2: Create ThemeToggle Component
```typescript
// src/components/common/ThemeToggle.tsx
- Import Sun/Moon icons from lucide-react
- Use useTheme hook
- Render button with current theme icon
- Add smooth rotation animation
```

### Step 3: Update Tailwind Config
```javascript
// tailwind.config.js
- Set darkMode: 'class'
- Verify dark mode utilities are available
```

### Step 4: Wrap App with ThemeProvider
```typescript
// src/App.tsx
- Import ThemeProvider
- Wrap entire app
- Apply dark class to root div conditionally
```

### Step 5: Add ThemeToggle to Header
```typescript
// src/components/layout/Header.tsx
- Import ThemeToggle
- Add to right side of header
- Update header styling with dark: classes
```

### Step 6: Update All Component Styles
- Add `dark:` prefixed classes to all components
- Test each component in both themes
- Ensure proper contrast ratios

## Accessibility Considerations

**Contrast Ratios:**
- Light mode: Maintain existing AA compliance
- Dark mode: Test all text/background combinations
- Ensure all interactive elements are visible

**Keyboard Navigation:**
- Theme toggle accessible via Tab key
- Enter/Space to activate toggle

**ARIA Labels:**
- Add aria-label="Toggle dark mode"
- Announce current theme to screen readers

## Performance

**Optimization:**
- Theme state loads from localStorage on mount (synchronous)
- Theme switching uses CSS classes (instant visual change)
- No re-renders needed for unchanged components
- localStorage write is debounced/throttled if needed

**Bundle Size Impact:**
- ThemeContext: ~1KB
- ThemeToggle component: ~0.5KB
- No additional dependencies needed

## Testing Strategy

**Manual Testing:**
- [ ] Toggle switches between themes
- [ ] Theme persists after page reload
- [ ] All text is readable in both themes
- [ ] No visual glitches during switch
- [ ] Works on different screen sizes

**Edge Cases:**
- localStorage unavailable (fallback to light)
- Rapid theme toggling
- Browser back/forward navigation

## Migration Path

**Phase 1: Core Implementation**
- Add ThemeContext and ThemeToggle
- Update main layout components

**Phase 2: Component Polish**
- Update all remaining components
- Fine-tune dark mode colors

**Phase 3: Enhancement**
- Add transition animations
- Consider system theme detection

## Color Palette Reference

### Light Mode (Current)
```css
Background: #ffffff, #f9fafb
Text: #111827, #374151
Border: #e5e7eb, #d1d5db
Primary: #2563eb
```

### Dark Mode (New)
```css
Background: #111827, #1f2937
Text: #f9fafb, #d1d5db
Border: #374151, #4b5563
Primary: #3b82f6
```

## Code Examples

### ThemeContext Usage
```typescript
const { theme, toggleTheme } = useTheme();

<button onClick={toggleTheme}>
  {theme === 'light' ? <Moon /> : <Sun />}
</button>
```

### Component Styling
```typescript
<div className="bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100">
  Content
</div>
```

---

*Generated by Frontend Agent - Automated Orchestration*
