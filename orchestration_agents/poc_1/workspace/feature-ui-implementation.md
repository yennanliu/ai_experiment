# Frontend Specification - Dark Mode Toggle

## Tech Stack
- Framework: React 18 with TypeScript
- State Management: React Context + localStorage
- Styling: CSS Custom Properties (CSS Variables) + Tailwind CSS dark mode
- Theme Detection: `prefers-color-scheme` media query

## Component Hierarchy

```
App (ThemeProvider wrapper)
├── ThemeToggle (toggle button component)
└── [Existing app components]
```

**Note:** This is a cross-cutting feature that wraps the entire application rather than adding new page-level components.

## State Management

**Theme Context** (`contexts/ThemeContext.tsx`):
```typescript
interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
}
```

**State Flow:**
1. On app initialization, check for saved preference in localStorage
2. If no saved preference, detect system preference via `prefers-color-scheme`
3. Apply theme by setting `data-theme` attribute on document root
4. When user toggles, update context state and localStorage
5. All components consume theme via context or CSS variables

**Persistence Strategy:**
- Key: `app-theme`
- Storage: localStorage
- Fallback: System preference → light mode

## API Integration

**No API integration required** - This is a pure frontend feature with local persistence only.

## Key Components

### ThemeProvider
**Purpose:** Global theme state management and persistence
**Props:** children: ReactNode
**State:** 
- theme: 'light' | 'dark'
- mounted: boolean (prevents SSR flash)

**Behavior:**
- Initializes theme on mount from localStorage or system preference
- Applies theme to document root via `data-theme` attribute
- Saves theme changes to localStorage
- Prevents flash of wrong theme (FOUC)

**Implementation:**
```typescript
// Pseudo-code structure
useEffect(() => {
  // 1. Check localStorage
  const saved = localStorage.getItem('app-theme');
  
  // 2. Check system preference
  const systemPreference = window.matchMedia('(prefers-color-scheme: dark)').matches 
    ? 'dark' 
    : 'light';
  
  // 3. Set initial theme
  const initialTheme = saved || systemPreference;
  setTheme(initialTheme);
  document.documentElement.setAttribute('data-theme', initialTheme);
}, []);
```

### ThemeToggle
**Purpose:** User control for switching themes
**Props:** 
- className?: string (optional styling override)
- position?: 'header' | 'floating' (default: 'header')

**State:** None (controlled by ThemeContext)

**Behavior:**
- Displays sun icon in dark mode, moon icon in light mode
- Smooth icon transition animation
- Keyboard accessible (Space/Enter to toggle)
- Screen reader announces "Switch to [opposite] mode"
- Tooltip shows "Dark mode" or "Light mode"
- Ripple effect on click

**Visual States:**
- Light mode: Moon icon with subtle background
- Dark mode: Sun icon with subtle background
- Hover: Slightly larger scale + background highlight
- Focus: Visible focus ring (accessibility)
- Active: Scale down slightly for tactile feedback

### ThemeScript (Critical Inline Script)
**Purpose:** Prevent flash of unstyled content (FOUC)
**Implementation:** Inline `<script>` in `index.html` before app loads

```html
<script>
  (function() {
    const theme = localStorage.getItem('app-theme') || 
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
  })();
</script>
```

## CSS Architecture

### CSS Custom Properties Structure

**Global Variables** (`styles/themes.css`):
```css
:root[data-theme="light"] {
  /* Backgrounds */
  --color-bg-primary: #ffffff;
  --color-bg-secondary: #f7f7f7;
  --color-bg-tertiary: #eeeeee;
  
  /* Text */
  --color-text-primary: #1a1a1a;
  --color-text-secondary: #666666;
  --color-text-tertiary: #999999;
  
  /* Borders */
  --color-border: #e0e0e0;
  --color-border-hover: #cccccc;
  
  /* Interactive */
  --color-primary: #3b82f6;
  --color-primary-hover: #2563eb;
  
  /* Feedback */
  --color-success: #10b981;
  --color-error: #ef4444;
  --color-warning: #f59e0b;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
}

:root[data-theme="dark"] {
  /* Backgrounds */
  --color-bg-primary: #1a1a1a;
  --color-bg-secondary: #2d2d2d;
  --color-bg-tertiary: #404040;
  
  /* Text */
  --color-text-primary: #f5f5f5;
  --color-text-secondary: #b3b3b3;
  --color-text-tertiary: #808080;
  
  /* Borders */
  --color-border: #404040;
  --color-border-hover: #525252;
  
  /* Interactive */
  --color-primary: #60a5fa;
  --color-primary-hover: #3b82f6;
  
  /* Feedback */
  --color-success: #34d399;
  --color-error: #f87171;
  --color-warning: #fbbf24;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.4);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.5);
}
```

### Tailwind Configuration

**Update `tailwind.config.js`:**
```javascript
module.exports = {
  darkMode: 'class', // or ['class', '[data-theme="dark"]']
  theme: {
    extend: {
      colors: {
        'bg-primary': 'var(--color-bg-primary)',
        'bg-secondary': 'var(--color-bg-secondary)',
        'text-primary': 'var(--color-text-primary)',
        // ... map all CSS variables
      }
    }
  }
}
```

### Component-Level Styling

**Pattern for all components:**
```css
.component {
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
  border-color: var(--color-border);
  transition: background-color 0.2s ease, color 0.2s ease;
}
```

## Theme Transition Animation

**Strategy:** Smooth but fast transitions
- Duration: 200ms (fast enough to feel instant, smooth enough to not flash)
- Properties: background-color, color, border-color
- Easing: ease-in-out

**Global Transition** (`App.css`):
```css
* {
  transition: 
    background-color 0.2s ease-in-out,
    color 0.2s ease-in-out,
    border-color 0.2s ease-in-out;
}

/* Disable transitions on theme script to prevent FOUC */
.no-transition * {
  transition: none !important;
}
```

## Routing

No routing changes - ThemeToggle component added to header/navigation across all pages.

## Error Handling

**Graceful Degradation:**
- If localStorage is unavailable (private browsing), theme still works but won't persist
- If system preference detection fails, default to light mode
- Log errors to console for debugging, don't show to user

**Error Cases:**
```typescript
try {
  localStorage.setItem('app-theme', theme);
} catch (error) {
  console.warn('Unable to save theme preference:', error);
  // Continue without persistence
}
```

## Accessibility Considerations

### WCAG Compliance
- **Contrast Ratios:**
  - Normal text: Minimum 4.5:1 (AA) / Target 7:1 (AAA)
  - Large text: Minimum 3:1 (AA) / Target 4.5:1 (AAA)
  - UI components: Minimum 3:1

**Color Palette Validation:**
- Test all color combinations with contrast checker
- Ensure interactive elements meet contrast requirements
- Provide sufficient contrast for focus indicators

### Keyboard Navigation
- **ThemeToggle:**
  - Focusable via Tab
  - Activatable via Space or Enter
  - Clear focus indicator (2px outline)

### Screen Reader Support
```typescript
<button
  onClick={toggleTheme}
  aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
  aria-live="polite"
  aria-pressed={theme === 'dark'}
>
  {/* Icon */}
</button>
```

**Announcements:**
- When theme changes: "Dark mode activated" or "Light mode activated"
- Use aria-live region for non-disruptive announcements

### Reduced Motion Support
```css
@media (prefers-reduced-motion: reduce) {
  * {
    transition: none !important;
    animation: none !important;
  }
}
```

## Performance Optimizations

### Critical Path Optimization
1. **Inline Theme Script:** Prevent FOUC by setting theme before React hydrates
2. **Minimize Repaints:** Use CSS variables instead of re-rendering components
3. **Debounce System Preference Listener:** Only update when user changes OS setting

### Lazy Loading
- Theme assets (icons) should be inlined or use SVG sprites
- No external dependencies for theme switching

### Memoization
```typescript
const ThemeToggle = React.memo(() => {
  const { theme, toggleTheme } = useTheme();
  // Component implementation
});
```

### Bundle Size
- No external theme libraries needed
- Use built-in CSS variables
- SVG icons for sun/moon (lightweight)
- Total added JS: ~2KB (minified + gzipped)

## Testing Strategy

### Unit Tests
**ThemeContext:**
- ✓ Initializes with localStorage value
- ✓ Falls back to system preference
- ✓ Defaults to light mode if no preference
- ✓ Updates localStorage on theme change
- ✓ Applies theme attribute to document root

**ThemeToggle:**
- ✓ Renders correct icon for current theme
- ✓ Calls toggleTheme on click
- ✓ Accessible via keyboard
- ✓ Shows correct aria-label

### Integration Tests
- ✓ Theme persists across page refreshes
- ✓ Theme applies to all UI components
- ✓ No FOUC on initial load
- ✓ Smooth transition between themes

### E2E Tests (Playwright)
```typescript
test('user can toggle dark mode', async ({ page }) => {
  await page.goto('/');
  
  // Initial state (system or default)
  const initialTheme = await page.getAttribute('html', 'data-theme');
  
  // Click toggle
  await page.click('[aria-label*="Switch to"]');
  
  // Verify theme changed
  const newTheme = await page.getAttribute('html', 'data-theme');
  expect(newTheme).not.toBe(initialTheme);
  
  // Verify persistence
  await page.reload();
  const persistedTheme = await page.getAttribute('html', 'data-theme');
  expect(persistedTheme).toBe(newTheme);
});
```

### Visual Regression Tests
- Capture screenshots of key pages in both themes
- Compare against baseline to catch unintended color changes

### Accessibility Audits
- Run axe-core or Lighthouse on both themes
- Verify all contrast ratios meet WCAG AA standards

## Implementation Phases

### Phase 1: Foundation (2-3 hours)
1. Set up CSS custom properties for light/dark themes
2. Create ThemeContext with localStorage persistence
3. Implement inline theme script to prevent FOUC
4. Add ThemeProvider to App root

### Phase 2: UI Components (2-3 hours)
5. Build ThemeToggle component with animations
6. Design and implement sun/moon icons
7. Add toggle to header/navigation
8. Update existing components to use CSS variables

### Phase 3: Polish & Testing (2-3 hours)
9. Test all pages in both themes
10. Verify contrast ratios and accessibility
11. Add keyboard navigation support
12. Implement reduced motion support
13. Write unit and integration tests

### Phase 4: Documentation (1 hour)
14. Document theme usage for other developers
15. Create style guide showing both themes
16. Add theme toggle to component library/Storybook

**Total Estimate: 7-10 hours**

## Implementation Notes

### Folder Structure
```
src/
├── components/
│   └── ThemeToggle/
│       ├── ThemeToggle.tsx
│       ├── ThemeToggle.test.tsx
│       ├── ThemeToggle.module.css
│       └── icons/
│           ├── SunIcon.tsx
│           └── MoonIcon.tsx
├── contexts/
│   └── ThemeContext.tsx
├── styles/
│   ├── themes.css (CSS variables)
│   └── transitions.css (animation rules)
├── hooks/
│   └── useTheme.ts (convenience hook)
└── utils/
    └── themeHelpers.ts (localStorage, system detection)
```

### Code Examples

**ThemeProvider Implementation:**
```typescript
export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<'light' | 'dark'>('light');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem('app-theme') as 'light' | 'dark' | null;
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches 
      ? 'dark' 
      : 'light';
    
    const initialTheme = savedTheme || systemTheme;
    setThemeState(initialTheme);
    document.documentElement.setAttribute('data-theme', initialTheme);
    setMounted(true);
  }, []);

  const setTheme = (newTheme: 'light' | 'dark') => {
    setThemeState(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
    try {
      localStorage.setItem('app-theme', newTheme);
    } catch (error) {
      console.warn('Unable to save theme:', error);
    }
  };

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  // Prevent rendering until theme is determined (avoid flash)
  if (!mounted) {
    return null; // or loading skeleton
  }

  return (
    <ThemeContext.Provider value={{ theme, setTheme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
```

**Usage in Components:**
```typescript
import { useTheme } from '@/hooks/useTheme';

const MyComponent = () => {
  const { theme } = useTheme();
  
  return (
    <div className="bg-bg-primary text-text-primary">
      Current theme: {theme}
    </div>
  );
};
```

### Migration Strategy for Existing Components

**Step 1:** Identify all hardcoded colors
```bash
# Search for color values
grep -r "bg-white\|bg-gray