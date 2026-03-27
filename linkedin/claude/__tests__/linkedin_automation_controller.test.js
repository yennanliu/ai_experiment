/**
 * Unit Tests for LinkedIn Automation Controller
 * Tests the buildSearchUrl function and configuration structure
 */

const {
  CONFIG,
  buildSearchUrl,
  INJECT_SCRIPT,
  CHECK_STATE_SCRIPT,
  getUpdateCounterScript
} = require('../linkedin_automation_controller');

describe('LinkedIn Automation Controller', () => {
  describe('CONFIG object', () => {
    test('should have required properties', () => {
      expect(CONFIG).toHaveProperty('jobTitle');
      expect(CONFIG).toHaveProperty('locations');
      expect(CONFIG).toHaveProperty('datePosted');
      expect(CONFIG).toHaveProperty('easyApplyOnly');
      expect(CONFIG).toHaveProperty('remoteOptions');
      expect(CONFIG).toHaveProperty('experienceLevel');
      expect(CONFIG).toHaveProperty('maxApplications');
      expect(CONFIG).toHaveProperty('delayBetweenApps');
      expect(CONFIG).toHaveProperty('skipIfQuestionsRequired');
      expect(CONFIG).toHaveProperty('maxPages');
      expect(CONFIG).toHaveProperty('useDefaultResume');
    });

    test('should have valid default values', () => {
      expect(typeof CONFIG.jobTitle).toBe('string');
      expect(Array.isArray(CONFIG.locations)).toBe(true);
      expect(CONFIG.locations.length).toBeGreaterThan(0);
      expect(typeof CONFIG.easyApplyOnly).toBe('boolean');
      expect(Array.isArray(CONFIG.remoteOptions)).toBe(true);
      expect(Array.isArray(CONFIG.experienceLevel)).toBe(true);
      expect(typeof CONFIG.maxApplications).toBe('number');
      expect(CONFIG.maxApplications).toBeGreaterThan(0);
      expect(CONFIG.delayBetweenApps).toHaveProperty('min');
      expect(CONFIG.delayBetweenApps).toHaveProperty('max');
    });

    test('delay settings should be valid', () => {
      expect(CONFIG.delayBetweenApps.min).toBeGreaterThan(0);
      expect(CONFIG.delayBetweenApps.max).toBeGreaterThanOrEqual(CONFIG.delayBetweenApps.min);
    });
  });

  describe('buildSearchUrl', () => {
    test('should build base URL with keywords', () => {
      const url = buildSearchUrl({ jobTitle: 'Software Engineer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).toContain('https://www.linkedin.com/jobs/search/');
      expect(url).toContain('keywords=Software+Engineer');
    });

    test('should add location parameter', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: ['Taiwan'], datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).toContain('location=Taiwan');
    });

    test('should add Easy Apply filter when enabled', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: null, easyApplyOnly: true, remoteOptions: [], experienceLevel: [] });
      expect(url).toContain('f_AL=true');
    });

    test('should not add Easy Apply filter when disabled', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).not.toContain('f_AL=true');
    });

    test('should map datePosted values correctly', () => {
      const config24h = { jobTitle: 'Engineer', locations: [], datePosted: 'Past 24 hours', easyApplyOnly: false, remoteOptions: [], experienceLevel: [] };
      const configWeek = { jobTitle: 'Engineer', locations: [], datePosted: 'Past week', easyApplyOnly: false, remoteOptions: [], experienceLevel: [] };
      const configMonth = { jobTitle: 'Engineer', locations: [], datePosted: 'Past month', easyApplyOnly: false, remoteOptions: [], experienceLevel: [] };

      expect(buildSearchUrl(config24h)).toContain('f_TPR=r86400');
      expect(buildSearchUrl(configWeek)).toContain('f_TPR=r604800');
      expect(buildSearchUrl(configMonth)).toContain('f_TPR=r2592000');
    });

    test('should handle "Any time" date filter', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: 'Any time', easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).not.toContain('f_TPR=');
    });

    test('should map remote options correctly', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: ['Remote', 'Hybrid'], experienceLevel: [] });
      expect(url).toContain('f_WT=');
      expect(url).toContain('2'); // Remote
      expect(url).toContain('3'); // Hybrid
    });

    test('should handle single remote option', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: ['Remote'], experienceLevel: [] });
      expect(url).toContain('f_WT=2');
    });

    test('should map experience levels correctly', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: ['Entry level', 'Mid-Senior level'] });
      expect(url).toContain('f_E=');
      expect(url).toContain('2'); // Entry level
      expect(url).toContain('4'); // Mid-Senior level
    });

    test('should handle empty experience level array', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).not.toContain('f_E=');
    });

    test('should use first location when multiple provided', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: ['Taiwan', 'Remote', 'Japan'], datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).toContain('location=Taiwan');
      expect(url).not.toContain('Remote'); // location parameter should only have first value
    });

    test('should return URL with default CONFIG', () => {
      const url = buildSearchUrl();
      expect(url).toContain('https://www.linkedin.com/jobs/search/');
      expect(url).toContain('keywords=');
      expect(url).toContain('Software'); // Job title gets URL encoded
    });

    test('should handle special characters in job title', () => {
      const url = buildSearchUrl({ jobTitle: 'C++/C# Developer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).toContain('keywords=');
    });

    test('should build valid URL structure', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: ['UK'], datePosted: null, easyApplyOnly: true, remoteOptions: ['Remote'], experienceLevel: ['Mid-Senior level'] });
      expect(url.startsWith('https://www.linkedin.com/jobs/search/?')).toBe(true);
      expect(url).toMatch(/[?&]keywords=/);
      expect(url).toMatch(/[?&]location=/);
    });
  });

  describe('getUpdateCounterScript', () => {
    test('should generate valid script for applied counter', () => {
      const script = getUpdateCounterScript('applied');
      expect(script).toContain('applied');
      expect(script).toContain('window.automationState');
      expect(script).toContain('++');
    });

    test('should generate valid script for skipped counter', () => {
      const script = getUpdateCounterScript('skipped');
      expect(script).toContain('skipped');
      expect(script).toContain('window.automationState');
    });

    test('should generate valid script for failed counter', () => {
      const script = getUpdateCounterScript('failed');
      expect(script).toContain('failed');
      expect(script).toContain('window.automationState');
    });

    test('should contain updateStatusUI call', () => {
      const script = getUpdateCounterScript('applied');
      expect(script).toContain('updateStatusUI');
    });

    test('should be valid JavaScript', () => {
      const script = getUpdateCounterScript('applied');
      expect(() => {
        // Check if it's syntactically valid by trying to parse
        new Function(script);
      }).not.toThrow();
    });
  });

  describe('INJECT_SCRIPT', () => {
    test('should be a non-empty string', () => {
      expect(typeof INJECT_SCRIPT).toBe('string');
      expect(INJECT_SCRIPT.length).toBeGreaterThan(0);
    });

    test('should contain helper functions', () => {
      expect(INJECT_SCRIPT).toContain('getJobCards');
      expect(INJECT_SCRIPT).toContain('clickJobByIndex');
      expect(INJECT_SCRIPT).toContain('clickEasyApply');
      expect(INJECT_SCRIPT).toContain('getModalState');
      expect(INJECT_SCRIPT).toContain('clickModalNext');
      expect(INJECT_SCRIPT).toContain('closeModal');
      expect(INJECT_SCRIPT).toContain('isSuccess');
    });

    test('should contain automation state object', () => {
      expect(INJECT_SCRIPT).toContain('automationState');
      expect(INJECT_SCRIPT).toContain('isPaused');
      expect(INJECT_SCRIPT).toContain('shouldQuit');
      expect(INJECT_SCRIPT).toContain('applied');
      expect(INJECT_SCRIPT).toContain('skipped');
      expect(INJECT_SCRIPT).toContain('failed');
    });

    test('should contain keyboard controls', () => {
      expect(INJECT_SCRIPT).toContain('keydown');
      expect(INJECT_SCRIPT).toContain("key === 'p'"); // pause
      expect(INJECT_SCRIPT).toContain("key === 'r'"); // resume
      expect(INJECT_SCRIPT).toContain("key === 'q'"); // quit
    });

    test('should have protection against double injection', () => {
      expect(INJECT_SCRIPT).toContain('__linkedinAutomationLoaded');
    });
  });

  describe('CHECK_STATE_SCRIPT', () => {
    test('should be a non-empty string', () => {
      expect(typeof CHECK_STATE_SCRIPT).toBe('string');
      expect(CHECK_STATE_SCRIPT.length).toBeGreaterThan(0);
    });

    test('should reference automationState', () => {
      expect(CHECK_STATE_SCRIPT).toContain('automationState');
    });

    test('should provide fallback state', () => {
      expect(CHECK_STATE_SCRIPT).toContain('isPaused');
      expect(CHECK_STATE_SCRIPT).toContain('shouldQuit');
    });
  });

  describe('Edge cases', () => {
    test('should handle config with null location', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: null, datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).toContain('keywords=');
    });

    test('should handle empty locations array', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).toContain('keywords=Engineer');
      expect(url).not.toContain('location=');
    });

    test('should handle invalid experience level gracefully', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: [], experienceLevel: ['Invalid Level'] });
      expect(url).toContain('keywords=');
      expect(url).not.toContain('f_E=');
    });

    test('should handle invalid remote option gracefully', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: null, easyApplyOnly: false, remoteOptions: ['Invalid'], experienceLevel: [] });
      expect(url).toContain('keywords=');
      expect(url).not.toContain('f_WT=');
    });

    test('should handle invalid date posted gracefully', () => {
      const url = buildSearchUrl({ jobTitle: 'Engineer', locations: [], datePosted: 'Invalid', easyApplyOnly: false, remoteOptions: [], experienceLevel: [] });
      expect(url).toContain('keywords=');
      expect(url).not.toContain('f_TPR=');
    });
  });

  describe('Real-world scenarios', () => {
    test('should build URL matching UK SWE search', () => {
      const config = {
        jobTitle: 'Software Engineer',
        locations: ['United Kingdom'],
        datePosted: 'Past week',
        easyApplyOnly: true,
        remoteOptions: ['Remote', 'Hybrid'],
        experienceLevel: []
      };
      const url = buildSearchUrl(config);

      expect(url).toContain('keywords=Software+Engineer');
      expect(url).toContain('location=United+Kingdom');
      expect(url).toContain('f_AL=true');
      expect(url).toContain('f_TPR=r604800');
      expect(url).toContain('f_WT=');
    });

    test('should build URL for senior roles', () => {
      const config = {
        jobTitle: 'Senior Software Engineer',
        locations: ['United States'],
        datePosted: 'Past month',
        easyApplyOnly: true,
        remoteOptions: ['Remote'],
        experienceLevel: ['Mid-Senior level', 'Director']
      };
      const url = buildSearchUrl(config);

      expect(url).toContain('Senior+Software+Engineer');
      expect(url).toContain('f_E=');
      expect(url.includes('4') && url.includes('5')).toBe(true); // Mid-Senior and Director codes
    });

    test('should build URL for entry level positions', () => {
      const config = {
        jobTitle: 'Junior Developer',
        locations: ['India'],
        datePosted: 'Past 24 hours',
        easyApplyOnly: true,
        remoteOptions: ['Remote'],
        experienceLevel: ['Entry level']
      };
      const url = buildSearchUrl(config);

      expect(url).toContain('Junior+Developer');
      expect(url).toContain('f_TPR=r86400');
      expect(url).toContain('f_E=2'); // Entry level
    });
  });
});
