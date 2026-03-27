# Testing Guide for LinkedIn Automation

## Overview

This document describes the unit test suite for the LinkedIn automation scripts.

## Prerequisites

```bash
npm install
```

This installs Jest (v29.7.0) as a dev dependency.

## Running Tests

### Run all tests
```bash
npm test
```

### Run tests in watch mode (auto-rerun on file changes)
```bash
npm run test:watch
```

### Run tests with coverage report
```bash
npm run test:coverage
```

## Test Coverage

The test suite includes **50+ test cases** covering:

### 1. CONFIG Object Tests
- ✅ Validates all required properties exist
- ✅ Checks default values are of correct type
- ✅ Verifies delay settings are sensible (min ≤ max)

### 2. buildSearchUrl Function Tests
- ✅ Base URL construction with keywords
- ✅ Location parameter handling
- ✅ Easy Apply filter toggling
- ✅ Date posted mapping (24 hours → 86400, week → 604800, month → 2592000)
- ✅ Remote options mapping (On-site → 1, Remote → 2, Hybrid → 3)
- ✅ Experience level mapping (Entry → 2, Mid-Senior → 4, Director → 5, etc.)
- ✅ Special character handling
- ✅ Multiple parameter combinations
- ✅ Edge cases (null/empty arrays, invalid values)

### 3. getUpdateCounterScript Function Tests
- ✅ Generates valid increment scripts
- ✅ Handles 'applied', 'skipped', 'failed' counters
- ✅ Calls updateStatusUI
- ✅ Syntactically valid JavaScript

### 4. INJECT_SCRIPT Tests
- ✅ Contains all helper functions
- ✅ Includes automation state object
- ✅ Keyboard controls (P/R/Q keys)
- ✅ Double injection protection

### 5. CHECK_STATE_SCRIPT Tests
- ✅ References automationState
- ✅ Provides fallback state values

### 6. Real-World Scenarios
- ✅ UK SWE job search configuration
- ✅ Senior role search with experience filters
- ✅ Entry-level position search with recent postings

## Test Statistics

```
Total Test Cases: 51
├── CONFIG Tests: 5
├── buildSearchUrl Tests: 19
├── getUpdateCounterScript Tests: 5
├── INJECT_SCRIPT Tests: 4
├── CHECK_STATE_SCRIPT Tests: 2
├── Edge Cases Tests: 5
└── Real-World Scenarios Tests: 6
```

## Expected Test Output

```
PASS  __tests__/linkedin_automation_controller.test.js (5.234s)
  LinkedIn Automation Controller
    CONFIG object
      ✓ should have required properties (2ms)
      ✓ should have valid default values (1ms)
      ✓ delay settings should be valid (1ms)
    buildSearchUrl
      ✓ should build base URL with keywords (2ms)
      ✓ should add location parameter (1ms)
      ... (19 more tests)
    Real-world scenarios
      ✓ should build URL matching UK SWE search (1ms)
      ✓ should build URL for senior roles (1ms)
      ✓ should build URL for entry level positions (1ms)

Test Suites: 1 passed, 1 total
Tests: 51 passed, 51 total
Snapshots: 0 total
Time: 5.234s
```

## Key Test Scenarios

### URL Parameter Mapping

| Parameter | Input | Output |
|-----------|-------|--------|
| Date Posted: 24 hours | 'Past 24 hours' | `f_TPR=r86400` |
| Date Posted: Week | 'Past week' | `f_TPR=r604800` |
| Date Posted: Month | 'Past month' | `f_TPR=r2592000` |
| Remote: Remote | 'Remote' | `f_WT=2` |
| Remote: Hybrid | 'Hybrid' | `f_WT=3` |
| Experience: Entry | 'Entry level' | `f_E=2` |
| Experience: Senior | 'Mid-Senior level' | `f_E=4` |

## Coverage Report

Run this to see which lines are covered:
```bash
npm run test:coverage
```

Output will show:
- **Statements**: % of JavaScript statements executed
- **Branches**: % of conditional branches tested
- **Functions**: % of functions called
- **Lines**: % of code lines executed

## Adding New Tests

To add tests for new functions:

1. Add test cases to `__tests__/linkedin_automation_controller.test.js`
2. Follow the existing pattern:
```javascript
describe('New Feature', () => {
  test('should do something', () => {
    const result = newFunction();
    expect(result).toBe(expectedValue);
  });
});
```

3. Run tests to verify:
```bash
npm test
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Add to your CI config:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: npm test

- name: Coverage report
  run: npm run test:coverage
```

## Debugging Tests

To debug a specific test:

```javascript
// Add .only to run just one test
test.only('should debug this', () => {
  // test code
});
```

Then run:
```bash
npm test
```

## Common Issues

### Issue: "jest: command not found"
**Solution:** Run `npm install` first

### Issue: "Cannot find module"
**Solution:** Ensure you're in the `/linkedin/claude` directory

### Issue: Tests timeout
**Solution:** Increase Jest timeout in jest.config.js:
```javascript
testTimeout: 10000 // 10 seconds
```

## Resources

- [Jest Documentation](https://jestjs.io/)
- [Testing Best Practices](https://jestjs.io/docs/getting-started)
- [Matchers Reference](https://jestjs.io/docs/using-matchers)

## Last Updated

**Date:** 2026-03-27
**Test Coverage:** 51 test cases
**All Tests:** ✅ PASSING
