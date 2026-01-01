# Frontend Specification - Dark Mode Toggle

## Tech Stack
- Framework: React 18 with TypeScript
- State Management: React Context API + custom hook
- Styling: CSS Custom Properties (CSS Variables)
- Storage: Browser localStorage API
- Icons: Lucide React (lightweight icon library)

## Component Hierarchy

```
App (root)
├── ThemeProvider (context wrapper)
│   └── children (all app components)
└── Header/Navigation
    └── ThemeToggle (toggle button component)
```

## State Management

**ThemeContext:**
```typescript
interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
}
```

**State Flow:**
1. ThemeProvider initializes by reading localStorage
2. If no preference exists, defaults to 'light'
3. Applies theme class to document root immediately
4. Provides theme state and toggle function to all children
5. On theme change, updates localStorage and document class synchronously

**Why Context vs Other Solutions:**
- No external state library needed (lightweight feature)
- Theme needs to be accessible app-wide
- Single source of truth prevents inconsistencies
- Built-in React solution (no additional dependencies)

## Local Storage Integration

**Storage Schema:**
```typescript
{
  key: 'theme-preference',
  value: 'light' | 'dark'
}
```

**Storage Operations:**
- **Read**: On app initialization (before first render)
- **Write**: Immediately after theme toggle
- **Error Handling**: Fallback to 'light' if localStorage is unavailable (e.g., private browsing)

## Key Components

### ThemeProvider
**Purpose:** Global theme state management and persistence  
**Props:** `children: ReactNode`  
**State:** 
- `theme: 'light' | 'dark'`
- Initializes from localStorage or defaults to 'light'

**Behavior:**
```typescript
// On mount:
1. Read from localStorage('theme-preference')
2. If exists and valid → set theme
3. If not exists → default to 'light'
4. Apply theme to document.documentElement.classList
5. Prevent flash by executing before React renders

// On theme change:
1. Update state
2. Update localStorage
3. Update document root class
4. All consumers re-render with new theme
```

**Implementation Notes:**
- Use `useLayoutEffect` instead of `useEffect` to prevent flash
- Add/remove classes atomically to prevent FOUC (Flash of Unstyled Content)

### ThemeToggle
**Purpose:** User control to switch between themes  
**Props:** `className?: string` (optional positioning styles)  
**State:** None (consumes from ThemeContext)

**Behavior:**
- Displays icon representing **opposite** theme (moon in light mode, sun in dark mode)
- Click/Space/Enter triggers `toggleTheme()`
- Smooth icon transition (fade or rotate)
- Tooltip showing "Switch to dark mode" / "Switch to light mode"

**Visual States:**
- Light mode: Shows moon icon 🌙
- Dark mode: Shows sun icon ☀️
- Hover: Subtle scale or brightness change
- Active: Brief scale-down effect
- Focus: Visible focus ring (accessibility)

**Accessibility:**
```typescript
<button
  onClick={toggleTheme}
  aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
  aria-pressed={theme === 'dark'}
  title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
>
  {/* Icon */}
</button>
```

### useTheme Hook
**Purpose:** Consumer hook for accessing theme context  
**Returns:** `ThemeContextType`

**Usage Example:**
```typescript
function MyComponent() {
  const { theme, toggleTheme } = useTheme();
  
  return (
    <div className={`card ${theme}`}>
      {/* Component content */}
    </div>
  );
}
```

## CSS Architecture

**Approach:** CSS Custom Properties (Variables) + Root Class

### Color System

**CSS Variables Definition:**
```css
/* Light Theme (Default) */
:root {
  /* Backgrounds */
  --bg-primary: #ffffff;
  --bg-secondary: #f5f5f5;
  --bg-tertiary: #e5e5e5;
  
  /* Text */
  --text-primary: #1a1a1a;
  --text-secondary: #4a4a4a;
  --text-tertiary: #6a6a6a;
  
  /* Borders */
  --border-color: #d1d1d1;
  
  /* Interactive Elements */
  --button-bg: #007bff;
  --button-hover: #0056b3;
  --button-text: #ffffff;
  
  /* Shadows */
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.12);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  
  /* Accents */
  --accent-primary: #007bff;
  --accent-danger: #dc3545;
  --accent-success: #28a745;
}

/* Dark Theme */
:root.dark {
  /* Backgrounds */
  --bg-primary: #1a1a1a;
  --bg-secondary: #2d2d2d;
  --bg-tertiary: #3a3a3a;
  
  /* Text */
  --text-primary: #e5e5e5;
  --text-secondary: #b5b5b5;
  --text-tertiary: #8a8a8a;
  
  /* Borders */
  --border-color: #404040;
  
  /* Interactive Elements */
  --button-bg: #0d6efd;
  --button-hover: #0a58ca;
  --button-text: #ffffff;
  
  /* Shadows */
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.5);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.4);
  
  /* Accents */
  --accent-primary: #0d6efd;
  --accent-danger: #dc3545;
  --accent-success: #28a745;
}
```

**Transition Effects:**
```css
* {
  transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
}

/* Disable transitions on theme change to prevent slow fade on large apps */
.theme-changing * {
  transition: none !important;
}
```

**Component Usage:**
```css
.card {
  background-color: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-md);
}

.button-primary {
  background-color: var(--button-bg);
  color: var(--button-text);
}

.button-primary:hover {
  background-color: var(--button-hover);
}
```

## Theme Initialization Strategy

**Problem:** Prevent FOUC (Flash of Unstyled Content) when loading app

**Solution:** Inline script in `index.html` **before** React loads

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>App</title>
  
  <!-- Theme Initialization Script (BLOCKING) -->
  <script>
    (function() {
      try {
        const savedTheme = localStorage.getItem('theme-preference');
        const theme = savedTheme === 'dark' ? 'dark' : 'light';
        document.documentElement.classList.add(theme);
      } catch (e) {
        // localStorage unavailable, defaults to light (CSS default)
      }
    })();
  </script>
  
  <link rel="stylesheet" href="/styles.css">
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>
```

**Why This Works:**
1. Executes **synchronously** before any content renders
2. Adds theme class to `<html>` element immediately
3. CSS loads with correct theme already applied
4. React hydrates with theme already in place
5. Zero visual flash

## Error Handling

**localStorage Unavailable:**
```typescript
function getStoredTheme(): 'light' | 'dark' {
  try {
    const stored = localStorage.getItem('theme-preference');
    return stored === 'dark' ? 'dark' : 'light';
  } catch (error) {
    console.warn('localStorage unavailable, using default theme');
    return 'light';
  }
}

function setStoredTheme(theme: 'light' | 'dark'): void {
  try {
    localStorage.setItem('theme-preference', theme);
  } catch (error) {
    console.warn('Failed to persist theme preference');
    // App continues to work, just won't persist across sessions
  }
}
```

**Graceful Degradation:**
- If localStorage fails → theme still works for current session
- If CSS variables unsupported (ancient browsers) → fallback to light theme
- If JavaScript disabled → light theme via CSS defaults

## Accessibility Considerations

**Keyboard Navigation:**
- ThemeToggle is a native `<button>` element (focusable by default)
- Tab key focuses the toggle
- Space or Enter activates toggle
- Focus ring visible (via `:focus-visible`)

**ARIA Attributes:**
```typescript
<button
  onClick={toggleTheme}
  aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
  aria-pressed={theme === 'dark'}
  title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
  className="theme-toggle"
>
  {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
</button>
```

**Color Contrast:**
- All text must meet WCAG AA standards (4.5:1 for normal text, 3:1 for large text)
- Use tools like WebAIM Contrast Checker during design
- Test both themes with automated accessibility tools (Axe, Lighthouse)

**Screen Reader Announcements:**
```typescript
// Optional: Announce theme change to screen readers
function announceThemeChange(newTheme: 'light' | 'dark') {
  const announcement = document.createElement('div');
  announcement.setAttribute('role', 'status');
  announcement.setAttribute('aria-live', 'polite');
  announcement.className = 'sr-only'; // Visually hidden
  announcement.textContent = `Switched to ${newTheme} mode`;
  document.body.appendChild(announcement);
  setTimeout(() => announcement.remove(), 1000);
}
```

**Focus Management:**
- ThemeToggle remains focused after activation
- No unexpected focus jumps
- Focus ring clearly visible in both themes

## Performance Optimizations

**Theme Switching Performance:**
- Use `document.documentElement.classList` (fastest DOM operation)
- Avoid re-rendering entire app on theme change (Context only triggers consumers)
- CSS transitions limited to 0.2s (perceived as instant)

**Initial Load Performance:**
- Inline script in HTML prevents extra network request
- localStorage read is synchronous and extremely fast (~1ms)
- No additional JavaScript bundle size (Context API is built-in)

**Memoization:**
```typescript
const ThemeProvider: React.FC<Props> = ({ children }) => {
  const [theme, setThemeState] = useState<Theme>(getInitialTheme);
  
  // Memoize context value to prevent unnecessary re-renders
  const value = useMemo(() => ({
    theme,
    toggleTheme: () => {
      const newTheme = theme === 'light' ? 'dark' : 'light';
      setThemeState(newTheme);
      applyTheme(newTheme);
    },
    setTheme: (newTheme: Theme) => {
      setThemeState(newTheme);
      applyTheme(newTheme);
    }
  }), [theme]);
  
  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};
```

**Bundle Size:**
- Context API: 0kb (built-in)
- Lucide React icons: ~2kb (tree-shakeable)
- Total feature overhead: <3kb

## Implementation Notes

### Folder Structure
```
src/
├── contexts/
│   └── ThemeContext.tsx          # Context provider + hook
├── components/
│   └── ThemeToggle.tsx            # Toggle button component
├── styles/
│   ├── theme.css                  # CSS variables for both themes
│   └── transitions.css            # Theme transition styles
├── utils/
│   └── theme.ts                   # localStorage helpers
└── types/
    └── theme.ts                   # TypeScript types
```

### Implementation Steps

**Phase 1: Foundation**
1. Create TypeScript types (`theme.ts`)
2. Set up CSS variables in `theme.css`
3. Add inline script to `index.html`

**Phase 2: React Integration**
4. Build ThemeContext with provider and hook
5. Wrap App with ThemeProvider
6. Test context in browser DevTools

**Phase 3: UI Component**
7. Create ThemeToggle component
8. Add icons and styles
9. Integrate into app header/navigation

**Phase 4: Polish**
10. Add smooth transitions
11. Test accessibility (keyboard, screen readers)
12. Test in all major browsers
13. Verify no FOUC on refresh

### Testing Strategy

**Unit Tests (Jest + React Testing Library):**
```typescript
describe('ThemeContext', () => {
  it('defaults to light theme', () => {
    const { result } = renderHook(() => useTheme(), {
      wrapper: ThemeProvider
    });
    expect(result.current.theme).toBe('light');
  });
  
  it('toggles theme', () => {
    const { result } = renderHook(() => useTheme(), {
      wrapper: ThemeProvider
    });
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('dark');
  });
  
  it('persists theme to localStorage', () => {
    const { result } = renderHook(() => useTheme(), {
      wrapper: ThemeProvider
    });
    act(() => result.current.setTheme('dark'));
    expect(localStorage.getItem('theme-preference')).toBe('dark');
  });
});

describe('ThemeToggle', () => {
  it('renders with correct icon', () => {
    render(<ThemeToggle />);
    expect(screen.getByLabelText(/switch to dark mode/i)).toBeInTheDocument();
  });
  
  it('is keyboard accessible', () => {
    render(<ThemeToggle />);
    const button = screen.getByRole('button');
    button.focus();
    expect(button).toHaveFocus();
  });
});
```

**Integration Tests:**
- Test theme persistence across page reloads
- Verify all components update when theme changes
- Test fallback behavior when localStorage unavailable

**Manual Testing Checklist:**
- [ ] Theme persists after browser refresh
- [ ] No flash of unstyled content on load
- [ ] Toggle works with mouse click
- [ ] Toggle works with keyboard (Tab + Enter/Space)
- [ ] Icons update correctly
- [ ] All text remains readable in both themes
- [ ] Color contrast meets WCAG AA standards
- [ ] Works in Chrome, Firefox, Safari, Edge
- [ ] Works in private/incognito mode (localStorage may be disabled)
- [ ] Focus ring visible when navigating with keyboard
- [ ] Screen reader announces theme change

**Accessibility Audit:**
- Run Lighthouse accessibility audit (both themes)
- Run Axe DevTools (both themes)
- Test with actual screen reader (NVDA/JAWS/VoiceOver)