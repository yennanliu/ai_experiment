# Frontend Specification - Dark Mode Implementation

## Tech Stack
- **Framework:** React 18+ with TypeScript
- **State Management:** React Context API + Custom Hook
- **Styling Solution:** CSS Variables + Tailwind CSS (or CSS Modules)
- **Storage:** localStorage API
- **System Detection:** window.matchMedia API
- **Animation:** CSS transitions

## Component Hierarchy

```
App (ThemeProvider wrapper)
├── ThemeToggle (theme switch control)
│   ├── SunIcon (light mode indicator)
│   ├── MoonIcon (dark mode indicator)
│   └── SystemIcon (system preference indicator)
└── [Existing App Components]
    └── All components consume theme via context
```

## State Management

### ThemeContext Structure

```typescript
interface ThemeContextValue {
  theme: 'light' | 'dark' | 'system';
  resolvedTheme: 'light' | 'dark'; // actual applied theme
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  systemTheme: 'light' | 'dark'; // OS preference
}
```

### State Flow
1. **Initialization:**
   - Check localStorage for saved preference
   - If no preference, check system preference via `prefers-color-scheme`
   - Apply theme to document root
   
2. **Theme Change:**
   - User selects theme via ThemeToggle
   - Update context state
   - Persist to localStorage
   - Apply CSS class to document root
   
3. **System Theme Change:**
   - Listen to `prefers-color-scheme` media query changes
   - Auto-update if user has selected 'system' mode

## API Integration

**No backend API integration required** - This is a pure frontend feature using browser APIs:
- `localStorage.setItem('theme', value)`
- `localStorage.getItem('theme')`
- `window.matchMedia('(prefers-color-scheme: dark)')`

## Key Components

### ThemeProvider
**Purpose:** Global theme state management and system theme detection  
**Props:** `children: ReactNode`  
**State:**
- `theme: 'light' | 'dark' | 'system'`
- `systemTheme: 'light' | 'dark'`

**Behavior:**
- Initializes theme from localStorage or system preference
- Listens for system theme changes
- Applies theme class to `document.documentElement`
- Provides theme context to children
- Prevents FOUC by applying theme before first render

**Implementation Notes:**
```typescript
// Apply theme synchronously in useLayoutEffect
useLayoutEffect(() => {
  const root = document.documentElement;
  root.classList.remove('light', 'dark');
  root.classList.add(resolvedTheme);
  root.style.colorScheme = resolvedTheme; // for native inputs
}, [resolvedTheme]);
```

### ThemeToggle
**Purpose:** User interface for switching themes  
**Props:** 
- `position?: 'header' | 'footer' | 'floating'` (optional positioning)
- `showLabels?: boolean` (show text labels)

**State:** None (controlled by ThemeContext)

**Behavior:**
- Three-state toggle: Light → Dark → System → Light
- Visual indication of current theme
- Keyboard accessible (Space/Enter to toggle)
- Tooltips for each state
- Smooth icon transitions
- ARIA labels for screen readers

**Variants:**
- Button with icons (recommended)
- Segmented control (3 buttons)
- Dropdown menu (for more options in future)

### useTheme Hook
**Purpose:** Convenient access to theme context  
**Returns:** ThemeContextValue  
**Usage:**
```typescript
const { theme, resolvedTheme, setTheme } = useTheme();
```

## Color System Design

### CSS Variables Strategy

**Root Variables (`styles/theme.css`):**
```css
:root {
  /* Light theme (default) */
  --color-bg-primary: #ffffff;
  --color-bg-secondary: #f8f9fa;
  --color-bg-elevated: #ffffff;
  
  --color-text-primary: #1a1a1a;
  --color-text-secondary: #6b7280;
  --color-text-disabled: #9ca3af;
  
  --color-border: #e5e7eb;
  --color-border-focus: #3b82f6;
  
  --color-accent: #3b82f6;
  --color-accent-hover: #2563eb;
  
  /* Semantic colors */
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
  --color-info: #3b82f6;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
}

.dark {
  /* Dark theme overrides */
  --color-bg-primary: #0f172a;
  --color-bg-secondary: #1e293b;
  --color-bg-elevated: #334155;
  
  --color-text-primary: #f1f5f9;
  --color-text-secondary: #cbd5e1;
  --color-text-disabled: #64748b;
  
  --color-border: #334155;
  --color-border-focus: #60a5fa;
  
  --color-accent: #60a5fa;
  --color-accent-hover: #3b82f6;
  
  /* Semantic colors (adjusted for dark bg) */
  --color-success: #34d399;
  --color-warning: #fbbf24;
  --color-error: #f87171;
  --color-info: #60a5fa;
  
  /* Shadows (more pronounced in dark mode) */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.5);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.5);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.5);
}
```

### Tailwind CSS Integration

**tailwind.config.js:**
```javascript
module.exports = {
  darkMode: 'class', // Use class-based dark mode
  theme: {
    extend: {
      colors: {
        'bg-primary': 'var(--color-bg-primary)',
        'bg-secondary': 'var(--color-bg-secondary)',
        // ... map all CSS variables
      }
    }
  }
}
```

## Routing

No routing changes required - ThemeProvider wraps entire app at root level.

## Error Handling

### localStorage Failures
```typescript
try {
  localStorage.setItem('theme', theme);
} catch (error) {
  console.warn('Failed to save theme preference:', error);
  // Fallback: continue with in-memory state only
}
```

### matchMedia Unsupported
```typescript
const getSystemTheme = (): 'light' | 'dark' => {
  if (typeof window === 'undefined') return 'light';
  if (!window.matchMedia) return 'light'; // Fallback for old browsers
  
  return window.matchMedia('(prefers-color-scheme: dark)').matches 
    ? 'dark' 
    : 'light';
};
```

### SSR Considerations
- Prevent hydration mismatch by reading theme on client only
- Use suppressHydrationWarning on html element
- Apply theme class before React hydration via inline script

## Accessibility Considerations

### WCAG 2.1 AA Compliance
- **Contrast Ratios:**
  - Normal text: 4.5:1 minimum
  - Large text (18pt+): 3:1 minimum
  - UI components: 3:1 minimum
  
- **Testing Tools:**
  - Use WebAIM Contrast Checker during development
  - Automated testing with axe-core

### Keyboard Navigation
- ThemeToggle accessible via Tab key
- Space/Enter to activate
- Focus visible indicator in both themes

### Screen Reader Support
```typescript
<button
  onClick={cycleTheme}
  aria-label={`Current theme: ${theme}. Click to change theme.`}
  aria-live="polite"
>
  {/* Icon */}
</button>
```

### Reduced Motion Support
```css
@media (prefers-reduced-motion: reduce) {
  .theme-transition {
    transition: none !important;
  }
}
```

## Performance Optimizations

### FOUC Prevention
**Inline Script in index.html:**
```html
<script>
  // Execute before React loads
  (function() {
    const theme = localStorage.getItem('theme') || 'system';
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    const resolvedTheme = theme === 'system' ? systemTheme : theme;
    document.documentElement.classList.add(resolvedTheme);
  })();
</script>
```

### CSS Optimization
- Use CSS variables for instant theme switching (no JS computation)
- Single repaint when class changes on root element
- Avoid inline styles in components (use CSS classes)

### Lazy Loading
- Theme toggle component can be code-split if placed in settings
- Icons loaded on-demand

### Memoization
```typescript
const ThemeToggle = React.memo(() => {
  const { theme, setTheme } = useTheme();
  // Component logic
});
```

## Testing Strategy

### Unit Tests
**ThemeContext:**
- ✓ Initializes with localStorage value
- ✓ Falls back to system preference if no saved value
- ✓ Persists theme changes to localStorage
- ✓ Updates on system theme change (when theme='system')

**ThemeToggle:**
- ✓ Cycles through themes correctly
- ✓ Displays correct icon for each theme
- ✓ Calls setTheme with correct value

### Integration Tests
- ✓ Theme applies to all components
- ✓ Theme persists across page reloads
- ✓ System preference detected correctly
- ✓ No visual regressions in either theme

### Accessibility Tests
- ✓ All contrast ratios meet WCAG AA
- ✓ Focus indicators visible in both themes
- ✓ Screen reader announces theme changes
- ✓ Keyboard navigation functional

### Visual Regression Tests
- ✓ Screenshot comparison for key pages in both themes
- ✓ Component library (Storybook) renders correctly in both themes

## Implementation Phases

### Phase 1: Foundation (Day 1)
- [ ] Create CSS variable system
- [ ] Build ThemeProvider and context
- [ ] Implement useTheme hook
- [ ] Add FOUC prevention script
- [ ] Set up localStorage persistence

### Phase 2: UI Components (Day 2)
- [ ] Create ThemeToggle component with icons
- [ ] Integrate toggle into header/navigation
- [ ] Implement smooth transitions
- [ ] Add keyboard navigation

### Phase 3: Theme Application (Day 3)
- [ ] Audit all components for color usage
- [ ] Replace hardcoded colors with CSS variables
- [ ] Test all interactive states (hover, focus, disabled)
- [ ] Fix contrast ratio issues

### Phase 4: Polish & Testing (Day 4)
- [ ] Add accessibility features (ARIA, focus management)
- [ ] Write unit and integration tests
- [ ] Conduct visual regression testing
- [ ] Performance audit
- [ ] Browser compatibility testing

## Implementation Notes

### Folder Structure
```
src/
├── contexts/
│   └── ThemeContext.tsx          # Theme provider and context
├── hooks/
│   └── useTheme.ts               # Theme hook
├── components/
│   └── ThemeToggle/
│       ├── ThemeToggle.tsx       # Toggle component
│       ├── ThemeToggle.test.tsx  # Unit tests
│       └── icons.tsx             # Sun/Moon/System icons
├── styles/
│   ├── theme.css                 # CSS variables
│   └── transitions.css           # Theme transition styles
└── utils/
    └── theme.ts                  # Helper functions
```

### Key Helper Functions
```typescript
// utils/theme.ts

export const THEME_STORAGE_KEY = 'theme-preference';

export const saveThemePreference = (theme: Theme): void => {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (error) {
    console.warn('Failed to save theme preference:', error);
  }
};

export const loadThemePreference = (): Theme | null => {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY) as Theme;
  } catch (error) {
    return null;
  }
};

export const getSystemTheme = (): 'light' | 'dark' => {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
};
```

### Transition Styles
```css
/* styles/transitions.css */

* {
  transition: 
    background-color 200ms ease-in-out,
    border-color 200ms ease-in-out,
    color 200ms ease-in-out;
}

@media (prefers-reduced-motion: reduce) {
  * {
    transition: none !important;
  }
}
```

### Critical CSS Variables Checklist
- ✓ Background colors (primary, secondary, elevated)
- ✓ Text colors (primary, secondary, tertiary, disabled)
- ✓ Border colors (default, hover, focus)
- ✓ Accent/brand colors
- ✓ Semantic colors (success, warning, error, info)
- ✓ Shadow values
- ✓ Input/form colors
- ✓ Button states (default, hover, active, disabled)
- ✓ Link colors (default, visited, hover)
- ✓ Code/syntax highlighting (if applicable)

## Success Metrics Tracking

**Implementation Checklist:**
- [ ] 100% component coverage (all components themed)
- [ ] Zero contrast ratio violations (automated axe-core tests)
- [ ] Theme switch < 100ms (Chrome DevTools Performance tab)
- [ ] No FOUC on page load (visual inspection)
- [ ] localStorage persistence 100% reliable (error handling in place)
- [ ] System preference detection works (tested on Mac/Windows/Linux)
- [ ] Keyboard navigation fully functional (manual testing)

## Future Enhancements (Out of Current Scope)

While not part of this implementation, consider for future iterations:
- Custom color theme builder
- Per-component theme overrides
- Scheduled theme switching (night mode hours)
- Syncing theme preference across devices (requires backend)
- High contrast mode
- Additional color schemes (sepia, blue light filter)

---

**Estimated Implementation Time:** 3-4 days  
**Dependencies:** None (can be implemented independently)  
**Risk Level:** Low (isolated feature, no backend dependencies)