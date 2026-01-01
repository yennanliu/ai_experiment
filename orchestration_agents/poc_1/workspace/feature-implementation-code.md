# Dark Mode Toggle - Complete Implementation Guide

## Overview
This guide provides complete, production-ready code for implementing a dark mode toggle feature in a React 18 + TypeScript application.

---

## Step 1: HTML Setup - Prevent FOUC

### File: `public/index.html`
Add this script in the `<head>` section **before** any other scripts:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Application with dark mode support" />
    
    <!-- Critical theme script - prevents flash of unstyled content -->
    <script>
      (function() {
        try {
          const theme = localStorage.getItem('app-theme') || 
            (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
          document.documentElement.setAttribute('data-theme', theme);
          
          // Add class for Tailwind dark mode if using Tailwind
          if (theme === 'dark') {
            document.documentElement.classList.add('dark');
          }
        } catch (e) {
          console.warn('Theme initialization failed:', e);
        }
      })();
    </script>
    
    <title>Your App</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
```

---

## Step 2: CSS Theme Variables

### File: `src/styles/themes.css`

```css
/* Theme Variables - Light and Dark Mode */

:root[data-theme="light"] {
  /* Backgrounds */
  --color-bg-primary: #ffffff;
  --color-bg-secondary: #f8f9fa;
  --color-bg-tertiary: #e9ecef;
  --color-bg-elevated: #ffffff;
  --color-bg-overlay: rgba(0, 0, 0, 0.5);
  
  /* Text Colors */
  --color-text-primary: #212529;
  --color-text-secondary: #6c757d;
  --color-text-tertiary: #adb5bd;
  --color-text-inverse: #ffffff;
  
  /* Border Colors */
  --color-border-primary: #dee2e6;
  --color-border-secondary: #e9ecef;
  --color-border-hover: #adb5bd;
  --color-border-focus: #3b82f6;
  
  /* Interactive Colors */
  --color-primary: #3b82f6;
  --color-primary-hover: #2563eb;
  --color-primary-active: #1d4ed8;
  --color-secondary: #6c757d;
  --color-secondary-hover: #5a6268;
  
  /* Feedback Colors */
  --color-success: #10b981;
  --color-success-light: #d1fae5;
  --color-error: #ef4444;
  --color-error-light: #fee2e2;
  --color-warning: #f59e0b;
  --color-warning-light: #fef3c7;
  --color-info: #3b82f6;
  --color-info-light: #dbeafe;
  
  /* Shadows */
  --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07), 0 2px 4px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05);
  --shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.1), 0 10px 10px rgba(0, 0, 0, 0.04);
  
  /* Focus Ring */
  --focus-ring: 0 0 0 3px rgba(59, 130, 246, 0.3);
}

:root[data-theme="dark"] {
  /* Backgrounds */
  --color-bg-primary: #1a1a1a;
  --color-bg-secondary: #2d2d2d;
  --color-bg-tertiary: #404040;
  --color-bg-elevated: #252525;
  --color-bg-overlay: rgba(0, 0, 0, 0.7);
  
  /* Text Colors */
  --color-text-primary: #f5f5f5;
  --color-text-secondary: #b3b3b3;
  --color-text-tertiary: #808080;
  --color-text-inverse: #1a1a1a;
  
  /* Border Colors */
  --color-border-primary: #404040;
  --color-border-secondary: #333333;
  --color-border-hover: #525252;
  --color-border-focus: #60a5fa;
  
  /* Interactive Colors */
  --color-primary: #60a5fa;
  --color-primary-hover: #3b82f6;
  --color-primary-active: #2563eb;
  --color-secondary: #9ca3af;
  --color-secondary-hover: #b3b3b3;
  
  /* Feedback Colors */
  --color-success: #34d399;
  --color-success-light: #064e3b;
  --color-error: #f87171;
  --color-error-light: #7f1d1d;
  --color-warning: #fbbf24;
  --color-warning-light: #78350f;
  --color-info: #60a5fa;
  --color-info-light: #1e3a8a;
  
  /* Shadows */
  --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.3);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.5), 0 4px 6px rgba(0, 0, 0, 0.4);
  --shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.6), 0 10px 10px rgba(0, 0, 0, 0.5);
  
  /* Focus Ring */
  --focus-ring: 0 0 0 3px rgba(96, 165, 250, 0.4);
}

/* Smooth transitions for theme changes */
* {
  transition: background-color 0.2s ease-in-out,
              color 0.2s ease-in-out,
              border-color 0.2s ease-in-out,
              box-shadow 0.2s ease-in-out;
}

/* Disable transitions for users who prefer reduced motion */
@media (prefers-reduced-motion: reduce) {
  * {
    transition: none !important;
    animation: none !important;
  }
}

/* Prevent transitions during initial load */
.no-transition * {
  transition: none !important;
}
```

---

## Step 3: Global Styles

### File: `src/styles/globals.css`

```css
@import './themes.css';

/* Base Styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html,
body {
  height: 100%;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
}

#root {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Focus Styles for Accessibility */
button:focus-visible,
a:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
  outline: 2px solid var(--color-border-focus);
  outline-offset: 2px;
  box-shadow: var(--focus-ring);
}

/* Scrollbar Theming */
::-webkit-scrollbar {
  width: 12px;
  height: 12px;
}

::-webkit-scrollbar-track {
  background: var(--color-bg-secondary);
}

::-webkit-scrollbar-thumb {
  background: var(--color-border-primary);
  border-radius: 6px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--color-border-hover);
}

/* Selection Color */
::selection {
  background-color: var(--color-primary);
  color: var(--color-text-inverse);
}
```

---

## Step 4: Theme Context

### File: `src/contexts/ThemeContext.tsx`

```typescript
import React, { createContext, useState, useEffect, useCallback, ReactNode } from 'react';

export type Theme = 'light' | 'dark';

export interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

export const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

const STORAGE_KEY = 'app-theme';

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setThemeState] = useState<Theme>('light');
  const [mounted, setMounted] = useState(false);

  // Initialize theme on mount
  useEffect(() => {
    try {
      // Check localStorage first
      const savedTheme = localStorage.getItem(STORAGE_KEY) as Theme | null;
      
      // Check system preference
      const systemPreference = window.matchMedia('(prefers-color-scheme: dark)').matches 
        ? 'dark' 
        : 'light';
      
      // Use saved theme or fall back to system preference
      const initialTheme = savedTheme || systemPreference;
      
      setThemeState(initialTheme);
      applyTheme(initialTheme);
      setMounted(true);
    } catch (error) {
      console.error('Error initializing theme:', error);
      setThemeState('light');
      applyTheme('light');
      setMounted(true);
    }
  }, []);

  // Listen for system preference changes
  useEffect(() => {
    if (!mounted) return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      // Only update if user hasn't set a manual preference
      const savedTheme = localStorage.getItem(STORAGE_KEY);
      if (!savedTheme) {
        const newTheme = e.matches ? 'dark' : 'light';
        setThemeState(newTheme);
        applyTheme(newTheme);
      }
    };

    // Modern browsers
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    } 
    // Fallback for older browsers
    else if (mediaQuery.addListener) {
      mediaQuery.addListener(handleChange);
      return () => mediaQuery.removeListener(handleChange);
    }
  }, [mounted]);

  const applyTheme = (newTheme: Theme) => {
    const root = document.documentElement;
    
    // Set data-theme attribute for CSS variables
    root.setAttribute('data-theme', newTheme);
    
    // Add/remove dark class for Tailwind compatibility
    if (newTheme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  };

  const setTheme = useCallback((newTheme: Theme) => {
    try {
      setThemeState(newTheme);
      applyTheme(newTheme);
      localStorage.setItem(STORAGE_KEY, newTheme);
      
      // Announce theme change to screen readers
      const announcement = `${newTheme === 'dark' ? 'Dark' : 'Light'} mode activated`;
      announceToScreenReader(announcement);
    } catch (error) {
      console.error('Error setting theme:', error);
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  }, [theme, setTheme]);

  // Helper function to announce changes to screen readers
  const announceToScreenReader = (message: string) => {
    const announcement = document.createElement('div');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  };

  // Prevent flash of unstyled content
  if (!mounted) {
    return null;
  }

  const value: ThemeContextType = {
    theme,
    setTheme,
    toggleTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};
```

---

## Step 5: Custom Hook

### File: `src/hooks/useTheme.ts`

```typescript
import { useContext } from 'react';
import { ThemeContext, ThemeContextType } from '../contexts/ThemeContext';

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  
  return context;
};
```

---

## Step 6: Theme Toggle Component - Icons

### File: `src/components/ThemeToggle/icons/SunIcon.tsx`

```typescript
import React from 'react';

interface SunIconProps {
  className?: string;
}

export const SunIcon: React.FC<SunIconProps> = ({ className = '' }) => {
  return (
    <svg
      className={className}
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M10 3V1M10 19V17M17 10H19M1 10H3M15.657 4.343L17.071 2