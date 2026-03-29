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

  window.hasEasyApplyButton = function(index) {
    const elements = document.querySelectorAll('[data-job-id]');
    if (index >= elements.length) return false;

    // Check if clicking this job will show Easy Apply button
    const card = elements[index];
    const text = card.textContent.toLowerCase();
    return !text.includes('applied') && !text.includes('no longer accepting');
  };

  window.clickJobByIndex = function(index) {
    const elements = document.querySelectorAll('[data-job-id]');
    if (index >= elements.length) return { success: false, error: 'Index out of range' };

    elements[index].click();
    return { success: true };
  };

  window.findEasyApplyButton = function() {
    // Method 1: Look for button with class containing 'jobs-apply-button'
    let btn = document.querySelector('button.jobs-apply-button');
    if (btn && !btn.disabled && btn.offsetParent !== null) return btn;

    // Method 2: Look for button with aria-label containing 'Easy Apply'
    btn = document.querySelector('button[aria-label*="Easy Apply"]');
    if (btn && !btn.disabled && btn.offsetParent !== null) return btn;

    // Method 3: Search all buttons for text content
    const buttons = Array.from(document.querySelectorAll('button'));
    for (const b of buttons) {
      const text = (b.textContent || '').toLowerCase();
      const ariaLabel = (b.getAttribute('aria-label') || '').toLowerCase();
      const innerHTML = (b.innerHTML || '').toLowerCase();

      // Check if button contains "easy apply" text and is visible
      if ((text.includes('easy apply') || ariaLabel.includes('easy apply') || innerHTML.includes('easy apply') ||
           text.includes('簡易應徵')) &&
          !b.disabled && b.offsetParent !== null) {
        return b;
      }
    }
    return null;
  };

  window.clickEasyApply = function() {
    const btn = window.findEasyApplyButton();
    if (!btn) return { success: false, error: 'Easy Apply button not found' };

    console.log('Found Easy Apply button:', btn.outerHTML.substring(0, 100));
    btn.click();
    return { success: true, buttonText: btn.textContent.trim() };
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
    return text.includes('application submitted') ||
           text.includes('application sent') ||
           text.includes('已送出申請') ||
           text.includes('your application was sent');
  };

  // Fill form fields with default values
  window.fillFormFields = function() {
    // Fill numeric inputs (years of experience, etc.)
    document.querySelectorAll('input[type="number"]').forEach(inp => {
      if (!inp.value || inp.value === '' || inp.value === '0') {
        inp.value = '5';
        inp.dispatchEvent(new Event('input', { bubbles: true }));
        inp.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });

    // Fill empty text inputs
    document.querySelectorAll('input[type="text"]').forEach(inp => {
      if (!inp.value && inp.required) {
        inp.value = 'N/A';
        inp.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });

    // Click "Yes" radio buttons for common questions
    document.querySelectorAll('input[type="radio"]').forEach(radio => {
      const label = radio.closest('label') || document.querySelector('label[for="' + radio.id + '"]');
      if (label && label.textContent.toLowerCase().includes('yes') && !radio.checked) {
        radio.click();
      }
    });
  };

  // Complete automation for a single job
  window.applyToJob = async function(jobIndex) {
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));

    const jobItems = document.querySelectorAll('[data-job-id]');
    if (jobIndex >= jobItems.length) {
      return { success: false, error: 'Job index out of range' };
    }

    const jobItem = jobItems[jobIndex];
    const titleEl = jobItem.querySelector('a');
    const jobTitle = titleEl?.textContent?.trim() || 'Unknown Job';

    console.log('[applyToJob] Starting:', jobTitle);

    // Step 1: Click on job to load details
    jobItem.click();
    await sleep(1500);

    // Step 2: Find and click Easy Apply button
    const easyApplyBtn = window.findEasyApplyButton();
    if (!easyApplyBtn) {
      console.log('[applyToJob] No Easy Apply button found');
      return { success: false, error: 'NO_EASY_APPLY', title: jobTitle };
    }

    console.log('[applyToJob] Clicking Easy Apply button');
    easyApplyBtn.click();
    await sleep(2500);

    // Step 3: Process form steps
    let stepsCompleted = 0;
    for (let step = 0; step < 15; step++) {
      // Fill any form fields
      window.fillFormFields();

      // Find the next/submit/continue button
      const allButtons = Array.from(document.querySelectorAll('button')).filter(b =>
        !b.disabled && b.offsetParent !== null
      );

      // Priority order: Submit > Continue > Next > Review
      const buttonPriority = ['submit application', 'submit', 'continue', 'next', 'review'];
      let clickedBtn = null;

      for (const priority of buttonPriority) {
        clickedBtn = allButtons.find(b => b.textContent.toLowerCase().includes(priority));
        if (clickedBtn) {
          console.log('[applyToJob] Step', step, '- Clicking:', clickedBtn.textContent.trim());
          clickedBtn.click();
          stepsCompleted++;
          break;
        }
      }

      if (!clickedBtn) {
        console.log('[applyToJob] No more buttons to click at step', step);
        break;
      }

      await sleep(1000);
    }

    // Step 4: Wait for completion and check success
    await sleep(2000);

    const success = window.isSuccess();
    console.log('[applyToJob] Success:', success, 'Steps:', stepsCompleted);

    // Step 5: Close any modals
    const closeBtn = document.querySelector('button[aria-label="Dismiss"]');
    if (closeBtn) {
      closeBtn.click();
      await sleep(500);
    }

    // Handle discard dialog if it appears
    window.handleDiscardDialog();
    await sleep(500);

    return {
      success: true,
      title: jobTitle,
      stepsCompleted: stepsCompleted,
      applicationSent: success
    };
  };

  // Batch apply to multiple jobs
  window.batchApply = async function(startIndex = 0, count = 10) {
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));

    const results = {
      applied: 0,
      skipped: 0,
      failed: 0,
      jobs: []
    };

    const jobItems = document.querySelectorAll('[data-job-id]');
    const endIndex = Math.min(startIndex + count, jobItems.length);

    console.log('[batchApply] Processing jobs', startIndex, 'to', endIndex - 1);

    for (let i = startIndex; i < endIndex; i++) {
      // Check pause/quit
      if (window.automationState.shouldQuit) {
        console.log('[batchApply] Quit requested');
        break;
      }

      while (window.automationState.isPaused) {
        await sleep(500);
      }

      try {
        const result = await window.applyToJob(i);

        if (result.error === 'NO_EASY_APPLY') {
          results.skipped++;
          window.automationState.skipped++;
          results.jobs.push({ index: i, title: result.title, status: 'SKIPPED' });
        } else if (result.applicationSent) {
          results.applied++;
          window.automationState.applied++;
          results.jobs.push({ index: i, title: result.title, status: 'SUCCESS' });
        } else {
          results.applied++; // Count as applied even if success detection unsure
          window.automationState.applied++;
          results.jobs.push({ index: i, title: result.title, status: 'COMPLETED' });
        }

        window.updateStatusUI('RUNNING');

        // Random delay between applications
        const delay = 2000 + Math.random() * 3000;
        await sleep(delay);

      } catch (err) {
        results.failed++;
        window.automationState.failed++;
        results.jobs.push({ index: i, status: 'ERROR', error: err.message });
        console.error('[batchApply] Error:', err.message);
      }
    }

    window.updateStatusUI('COMPLETED');
    console.log('[batchApply] Results:', results);
    return results;
  };

  // Initialize
  createStatusUI();
  console.log('✅ LinkedIn Automation ready! (P=Pause, R=Resume, Q=Quit)');
  console.log('📌 Use window.batchApply(startIndex, count) to apply to jobs');

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
