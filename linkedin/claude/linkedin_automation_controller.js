/**
 * LinkedIn Job Application Automation Controller
 *
 * This script coordinates the automation using Chrome DevTools MCP tools.
 * It connects to an existing Chrome session (no login required).
 *
 * Usage with Claude Code:
 *   1. Open Chrome with LinkedIn logged in
 *   2. Enable remote debugging: chrome://inspect/#remote-debugging
 *   3. Run automation via Claude Code MCP tools
 *
 * Configuration:
 *   Edit the CONFIG object below to set your preferences
 */

const CONFIG = {
  // === SEARCH PARAMETERS ===
  jobTitle: 'Software Engineer',

  // Locations to search (first one is used for initial search)
  locations: ['Taiwan', 'Remote'],

  // Date posted filter
  // Options: 'Any time', 'Past month', 'Past week', 'Past 24 hours'
  datePosted: 'Past week',

  // === FILTERS ===
  // Only apply to Easy Apply jobs
  easyApplyOnly: true,

  // Remote work preference
  // Options: 'On-site', 'Remote', 'Hybrid'
  remoteOptions: ['Remote', 'Hybrid'],

  // Experience level (leave empty for all)
  // Options: 'Internship', 'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive'
  experienceLevel: [],

  // === AUTOMATION SETTINGS ===
  // Maximum number of applications per session
  maxApplications: 50,

  // Delay between applications (milliseconds)
  delayBetweenApps: {
    min: 3000,
    max: 6000
  },

  // Skip jobs that require additional screening questions
  skipIfQuestionsRequired: false,

  // Maximum pages to process
  maxPages: 10,

  // === RESUME/COVER LETTER ===
  // Use default resume uploaded to LinkedIn
  useDefaultResume: true
};

/**
 * Build LinkedIn job search URL from configuration
 */
function buildSearchUrl(config = CONFIG) {
  const baseUrl = 'https://www.linkedin.com/jobs/search/';
  const params = new URLSearchParams();

  // Keywords
  params.set('keywords', config.jobTitle);

  // Location
  if (config.locations && config.locations.length > 0) {
    params.set('location', config.locations[0]);
  }

  // Easy Apply
  if (config.easyApplyOnly) {
    params.set('f_AL', 'true');
  }

  // Date posted
  const dateMap = {
    'Past 24 hours': 'r86400',
    'Past week': 'r604800',
    'Past month': 'r2592000',
    'Any time': ''
  };
  if (config.datePosted && dateMap[config.datePosted]) {
    params.set('f_TPR', dateMap[config.datePosted]);
  }

  // Remote filter
  const remoteMap = { 'On-site': '1', 'Remote': '2', 'Hybrid': '3' };
  if (config.remoteOptions && config.remoteOptions.length > 0) {
    const values = config.remoteOptions.map(o => remoteMap[o]).filter(Boolean).join(',');
    if (values) params.set('f_WT', values);
  }

  // Experience level
  const expMap = {
    'Internship': '1', 'Entry level': '2', 'Associate': '3',
    'Mid-Senior level': '4', 'Director': '5', 'Executive': '6'
  };
  if (config.experienceLevel && config.experienceLevel.length > 0) {
    const values = config.experienceLevel.map(e => expMap[e]).filter(Boolean).join(',');
    if (values) params.set('f_E', values);
  }

  return `${baseUrl}?${params.toString()}`;
}

/**
 * Inject automation helper scripts into page
 */
const INJECT_SCRIPT = `
(function() {
  // Prevent double injection
  if (window.__linkedinAutomationLoaded) return 'Already loaded';

  // Global state
  window.__linkedinAutomationLoaded = true;
  window.automationState = {
    isPaused: false,
    shouldQuit: false,
    applied: 0,
    skipped: 0,
    failed: 0
  };

  // Keyboard controls
  document.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    if (key === 'p') {
      window.automationState.isPaused = true;
      updateStatusUI('PAUSED');
    } else if (key === 'r') {
      window.automationState.isPaused = false;
      updateStatusUI('RUNNING');
    } else if (key === 'q') {
      window.automationState.shouldQuit = true;
      updateStatusUI('STOPPED');
    }
  });

  // Status indicator
  function createStatusUI() {
    let el = document.getElementById('linkedin-auto-status');
    if (el) return el;

    el = document.createElement('div');
    el.id = 'linkedin-auto-status';
    el.style.cssText = 'position:fixed;top:20px;right:20px;background:#0077B5;color:white;padding:15px 20px;border-radius:12px;font-family:system-ui;font-size:14px;z-index:999999;box-shadow:0 4px 20px rgba(0,0,0,0.3);min-width:200px;';
    document.body.appendChild(el);
    updateStatusUI('RUNNING');
    return el;
  }

  function updateStatusUI(status) {
    const el = document.getElementById('linkedin-auto-status');
    if (!el) return;

    const colors = {RUNNING:'#0077B5',PAUSED:'#FF9800',STOPPED:'#f44336',COMPLETED:'#4CAF50'};
    const icons = {RUNNING:'🤖',PAUSED:'⏸️',STOPPED:'🛑',COMPLETED:'✅'};
    const s = window.automationState;

    el.style.background = colors[status] || colors.RUNNING;
    el.innerHTML = '<div style="font-weight:600;margin-bottom:8px;">' + icons[status] + ' AUTOMATION ' + status + '</div>' +
      '<div style="font-size:12px;opacity:0.9;">P=Pause R=Resume Q=Quit</div>' +
      '<div style="font-size:12px;margin-top:8px;border-top:1px solid rgba(255,255,255,0.3);padding-top:8px;">Applied: ' + s.applied + ' | Skipped: ' + s.skipped + ' | Failed: ' + s.failed + '</div>';
  }

  window.updateStatusUI = updateStatusUI;

  // Helper functions
  window.getJobCards = function() {
    const cards = [];
    const selectors = ['.job-card-container', '.jobs-search-results__list-item', '[data-job-id]'];

    let elements = [];
    for (const sel of selectors) {
      elements = document.querySelectorAll(sel);
      if (elements.length > 0) break;
    }

    elements.forEach((card, i) => {
      const titleEl = card.querySelector('a[class*="job-card"], .job-card-list__title');
      const companyEl = card.querySelector('.job-card-container__company-name');
      const jobId = card.getAttribute('data-job-id') || 'job-' + i;
      const applied = card.textContent.includes('Applied') || card.querySelector('[class*="applied"]');

      cards.push({
        index: i,
        jobId: jobId,
        title: titleEl?.textContent?.trim() || 'Unknown',
        company: companyEl?.textContent?.trim() || 'Unknown',
        applied: !!applied
      });
    });

    return cards;
  };

  window.clickJobByIndex = function(index) {
    const cards = getJobCards();
    if (index >= cards.length) return { success: false, error: 'Index out of range' };

    const card = document.querySelectorAll('.job-card-container, .jobs-search-results__list-item')[index];
    const link = card?.querySelector('a');
    if (link) {
      link.click();
      return { success: true, job: cards[index] };
    }
    return { success: false, error: 'Could not click card' };
  };

  window.findEasyApplyButton = function() {
    const buttons = document.querySelectorAll('button');
    for (const btn of buttons) {
      const text = btn.textContent.toLowerCase();
      if ((text.includes('easy apply') || text.includes('簡易應徵')) && !btn.disabled) {
        return btn;
      }
    }
    return null;
  };

  window.clickEasyApply = function() {
    const btn = findEasyApplyButton();
    if (!btn) return { success: false, error: 'Easy Apply button not found' };
    btn.click();
    return { success: true };
  };

  window.getModalState = function() {
    const modal = document.querySelector('.jobs-easy-apply-modal, [role="dialog"]');
    if (!modal || modal.offsetParent === null) return { open: false };

    const text = modal.textContent.toLowerCase();
    let step = 'unknown';

    if (text.includes('contact')) step = 'contact';
    else if (text.includes('resume')) step = 'resume';
    else if (text.includes('question')) step = 'questions';
    else if (text.includes('review')) step = 'review';
    else if (text.includes('submitted') || text.includes('success')) step = 'submitted';

    // Find next/submit button
    const buttons = modal.querySelectorAll('button');
    let nextBtn = null;
    for (const btn of buttons) {
      const t = btn.textContent.toLowerCase();
      if (t.includes('next') || t.includes('submit') || t.includes('review') || t.includes('continue')) {
        if (!btn.disabled) nextBtn = btn;
      }
    }

    return { open: true, step, hasNextButton: !!nextBtn };
  };

  window.clickModalNext = function() {
    const modal = document.querySelector('.jobs-easy-apply-modal, [role="dialog"]');
    if (!modal) return { success: false, error: 'No modal' };

    const texts = ['submit application', 'submit', 'next', 'review', 'continue'];
    const buttons = modal.querySelectorAll('button');

    for (const text of texts) {
      for (const btn of buttons) {
        if (btn.textContent.toLowerCase().includes(text) && !btn.disabled) {
          btn.click();
          return { success: true, clicked: btn.textContent.trim() };
        }
      }
    }

    return { success: false, error: 'No clickable button found' };
  };

  window.closeModal = function() {
    const closeBtn = document.querySelector('button[aria-label="Dismiss"], .artdeco-modal__dismiss');
    if (closeBtn) {
      closeBtn.click();
      return { success: true };
    }
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', keyCode: 27 }));
    return { success: true, method: 'escape' };
  };

  window.handleDiscardDialog = function() {
    const buttons = document.querySelectorAll('button');
    for (const btn of buttons) {
      if (btn.textContent.toLowerCase().includes('discard')) {
        btn.click();
        return true;
      }
    }
    return false;
  };

  window.isSuccess = function() {
    const text = document.body.textContent.toLowerCase();
    return text.includes('application submitted') || text.includes('已送出申請');
  };

  // Initialize
  createStatusUI();
  console.log('✅ LinkedIn Automation ready! (P=Pause, R=Resume, Q=Quit)');

  return 'Automation helpers loaded';
})();
`;

/**
 * Script to check automation state
 */
const CHECK_STATE_SCRIPT = `
(function() {
  return window.automationState || { isPaused: false, shouldQuit: false, applied: 0, skipped: 0, failed: 0 };
})();
`;

/**
 * Script to increment counters
 */
function getUpdateCounterScript(type) {
  return `
(function() {
  if (window.automationState) {
    window.automationState.${type}++;
    window.updateStatusUI?.('RUNNING');
  }
  return window.automationState;
})();
`;
}

// Export for use with MCP tools
module.exports = {
  CONFIG,
  buildSearchUrl,
  INJECT_SCRIPT,
  CHECK_STATE_SCRIPT,
  getUpdateCounterScript
};

// Log configuration on load
console.log('LinkedIn Automation Controller');
console.log('==============================');
console.log('Configuration:');
console.log(`  Job Title: ${CONFIG.jobTitle}`);
console.log(`  Location: ${CONFIG.locations.join(', ')}`);
console.log(`  Date Posted: ${CONFIG.datePosted}`);
console.log(`  Easy Apply Only: ${CONFIG.easyApplyOnly}`);
console.log(`  Remote Options: ${CONFIG.remoteOptions.join(', ')}`);
console.log(`  Max Applications: ${CONFIG.maxApplications}`);
console.log('');
console.log('Search URL:');
console.log(buildSearchUrl());
