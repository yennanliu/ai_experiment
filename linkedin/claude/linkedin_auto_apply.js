/**
 * LinkedIn Job Application Automation
 *
 * Uses chrome-cdp-skill to connect to existing browser session (no login needed)
 *
 * Features:
 * - Press 'P' to PAUSE automation
 * - Press 'R' to RESUME automation
 * - Press 'Q' to QUIT automation
 * - Configurable job search parameters
 *
 * Prerequisites:
 * 1. Chrome remote debugging enabled (chrome://inspect/#remote-debugging)
 * 2. Already logged into LinkedIn in Chrome
 * 3. Node.js 22+ installed
 */

// Configuration
const CONFIG = {
  // Search parameters
  jobTitle: 'Software Engineer',
  locations: ['Taiwan', 'Remote'],
  datePosted: 'Past week', // Options: 'Any time', 'Past month', 'Past week', 'Past 24 hours'

  // Easy Apply filter (recommended)
  easyApplyOnly: true,

  // Experience level (optional)
  experienceLevel: [], // ['Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive']

  // Remote options
  remoteOptions: ['Remote', 'Hybrid'], // ['On-site', 'Remote', 'Hybrid']

  // Automation settings
  maxApplications: 50,
  delayBetweenApps: { min: 3000, max: 6000 }, // milliseconds

  // Resume/Cover Letter (if pre-uploaded to LinkedIn)
  useDefaultResume: true,

  // Safety
  skipIfQuestionsRequired: false, // Skip jobs that require additional questions
};

// Global automation state
let automationState = {
  isPaused: false,
  shouldQuit: false,
  applicationsCount: 0,
  skippedCount: 0,
  failedCount: 0
};

/**
 * Build LinkedIn job search URL from config
 */
function buildSearchUrl(config) {
  const baseUrl = 'https://www.linkedin.com/jobs/search/';
  const params = new URLSearchParams();

  // Keywords
  params.set('keywords', config.jobTitle);

  // Location (use first location)
  if (config.locations && config.locations.length > 0) {
    params.set('location', config.locations[0]);
  }

  // Easy Apply filter
  if (config.easyApplyOnly) {
    params.set('f_AL', 'true');
  }

  // Date posted filter
  const dateFilterMap = {
    'Past 24 hours': 'r86400',
    'Past week': 'r604800',
    'Past month': 'r2592000',
    'Any time': ''
  };
  if (dateFilterMap[config.datePosted]) {
    params.set('f_TPR', dateFilterMap[config.datePosted]);
  }

  // Remote filter
  const remoteMap = {
    'On-site': '1',
    'Remote': '2',
    'Hybrid': '3'
  };
  if (config.remoteOptions && config.remoteOptions.length > 0) {
    const remoteValues = config.remoteOptions
      .map(opt => remoteMap[opt])
      .filter(Boolean)
      .join(',');
    if (remoteValues) {
      params.set('f_WT', remoteValues);
    }
  }

  // Experience level
  const expMap = {
    'Internship': '1',
    'Entry level': '2',
    'Associate': '3',
    'Mid-Senior level': '4',
    'Director': '5',
    'Executive': '6'
  };
  if (config.experienceLevel && config.experienceLevel.length > 0) {
    const expValues = config.experienceLevel
      .map(exp => expMap[exp])
      .filter(Boolean)
      .join(',');
    if (expValues) {
      params.set('f_E', expValues);
    }
  }

  return `${baseUrl}?${params.toString()}`;
}

/**
 * Create status indicator overlay on page
 */
function createStatusIndicator() {
  // Remove existing indicator
  const existing = document.getElementById('linkedin-automation-status');
  if (existing) existing.remove();

  const statusBox = document.createElement('div');
  statusBox.id = 'linkedin-automation-status';
  statusBox.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: linear-gradient(135deg, #0077B5 0%, #00A0DC 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14px;
    font-weight: 600;
    z-index: 999999;
    box-shadow: 0 4px 20px rgba(0,119,181,0.4);
    min-width: 220px;
  `;
  statusBox.innerHTML = `
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
      <span style="font-size: 20px;">🤖</span>
      <span>AUTOMATION RUNNING</span>
    </div>
    <div style="font-size: 11px; opacity: 0.9;">
      P = Pause | R = Resume | Q = Quit
    </div>
    <div id="automation-stats" style="margin-top: 10px; font-size: 12px; border-top: 1px solid rgba(255,255,255,0.3); padding-top: 8px;">
      Applied: 0 | Skipped: 0 | Failed: 0
    </div>
  `;
  document.body.appendChild(statusBox);
  return statusBox;
}

/**
 * Update status indicator
 */
function updateStatusIndicator(status, stats = {}) {
  const statusBox = document.getElementById('linkedin-automation-status');
  if (!statusBox) return;

  const colors = {
    'RUNNING': 'linear-gradient(135deg, #0077B5 0%, #00A0DC 100%)',
    'PAUSED': 'linear-gradient(135deg, #FF9800 0%, #FFB74D 100%)',
    'COMPLETED': 'linear-gradient(135deg, #4CAF50 0%, #81C784 100%)',
    'ERROR': 'linear-gradient(135deg, #f44336 0%, #e57373 100%)'
  };

  const icons = {
    'RUNNING': '🤖',
    'PAUSED': '⏸️',
    'COMPLETED': '✅',
    'ERROR': '❌'
  };

  statusBox.style.background = colors[status] || colors['RUNNING'];

  const statsDiv = statusBox.querySelector('#automation-stats');
  if (statsDiv && stats) {
    statsDiv.textContent = `Applied: ${stats.applied || 0} | Skipped: ${stats.skipped || 0} | Failed: ${stats.failed || 0}`;
  }

  const headerDiv = statusBox.querySelector('div');
  if (headerDiv) {
    headerDiv.innerHTML = `
      <span style="font-size: 20px;">${icons[status]}</span>
      <span>AUTOMATION ${status}</span>
    `;
  }
}

/**
 * Setup keyboard controls for pause/resume/quit
 */
function setupKeyboardControls() {
  // Remove existing listener
  if (window._linkedinAutomationKeyHandler) {
    document.removeEventListener('keydown', window._linkedinAutomationKeyHandler);
  }

  window.linkedinAutomationControl = {
    paused: false,
    quit: false
  };

  window._linkedinAutomationKeyHandler = (e) => {
    const key = e.key.toLowerCase();

    if (key === 'p') {
      window.linkedinAutomationControl.paused = true;
      console.log('⏸️ AUTOMATION PAUSED');
      updateStatusIndicator('PAUSED');
    } else if (key === 'r') {
      window.linkedinAutomationControl.paused = false;
      console.log('▶️ AUTOMATION RESUMED');
      updateStatusIndicator('RUNNING');
    } else if (key === 'q') {
      window.linkedinAutomationControl.quit = true;
      console.log('🛑 AUTOMATION QUIT');
    }
  };

  document.addEventListener('keydown', window._linkedinAutomationKeyHandler);
  console.log('✅ Keyboard controls initialized (P=Pause, R=Resume, Q=Quit)');
}

/**
 * Get list of job cards on current page
 */
function getJobCards() {
  const cards = [];

  // LinkedIn job cards selector (may need adjustment based on page structure)
  const jobCardSelectors = [
    '.job-card-container',
    '.jobs-search-results__list-item',
    '[data-job-id]',
    '.scaffold-layout__list-item'
  ];

  let jobElements = [];
  for (const selector of jobCardSelectors) {
    jobElements = document.querySelectorAll(selector);
    if (jobElements.length > 0) break;
  }

  jobElements.forEach((card, index) => {
    const titleEl = card.querySelector('.job-card-list__title, a[class*="job-card"], .artdeco-entity-lockup__title');
    const companyEl = card.querySelector('.job-card-container__company-name, .artdeco-entity-lockup__subtitle');
    const locationEl = card.querySelector('.job-card-container__metadata-item, .job-card-container__metadata-wrapper');

    const jobId = card.getAttribute('data-job-id') ||
                  card.querySelector('[data-job-id]')?.getAttribute('data-job-id') ||
                  `job-${index}`;

    const alreadyApplied = card.textContent.includes('Applied') ||
                           card.textContent.includes('已應徵') ||
                           card.querySelector('[class*="applied"]');

    cards.push({
      index,
      element: card,
      jobId,
      title: titleEl?.textContent?.trim() || 'Unknown Title',
      company: companyEl?.textContent?.trim() || 'Unknown Company',
      location: locationEl?.textContent?.trim() || '',
      alreadyApplied: !!alreadyApplied
    });
  });

  return cards;
}

/**
 * Click on a job card to view details
 */
function clickJobCard(index) {
  const cards = getJobCards();
  if (index >= cards.length) {
    throw new Error(`Job index ${index} out of range (${cards.length} jobs)`);
  }

  const card = cards[index];
  const clickTarget = card.element.querySelector('a') || card.element;
  clickTarget.click();

  return {
    success: true,
    job: {
      title: card.title,
      company: card.company,
      jobId: card.jobId
    }
  };
}

/**
 * Find and click Easy Apply button
 */
function findEasyApplyButton() {
  const buttonSelectors = [
    'button.jobs-apply-button',
    'button[data-control-name="jobdetails_topcard_inapply"]',
    '.jobs-apply-button--top-card',
    'button:has-text("Easy Apply")',
    'button:has-text("簡易應徵")'
  ];

  // Try multiple selectors
  for (const selector of buttonSelectors) {
    try {
      const button = document.querySelector(selector);
      if (button && button.textContent.toLowerCase().includes('easy apply')) {
        return button;
      }
    } catch (e) {
      // Selector syntax error, skip
    }
  }

  // Fallback: find button by text content
  const allButtons = document.querySelectorAll('button');
  for (const button of allButtons) {
    const text = button.textContent.toLowerCase();
    if (text.includes('easy apply') || text.includes('簡易應徵')) {
      return button;
    }
  }

  return null;
}

/**
 * Click Easy Apply button to open application modal
 */
function clickEasyApply() {
  const button = findEasyApplyButton();

  if (!button) {
    return { success: false, reason: 'Easy Apply button not found' };
  }

  button.click();
  return { success: true, message: 'Easy Apply button clicked' };
}

/**
 * Check if application modal is open
 */
function isApplicationModalOpen() {
  const modalSelectors = [
    '.jobs-easy-apply-modal',
    '[data-test-modal]',
    '.artdeco-modal--layer-default',
    '[role="dialog"]'
  ];

  for (const selector of modalSelectors) {
    const modal = document.querySelector(selector);
    if (modal && modal.offsetParent !== null) {
      // Check if it's an application modal
      const modalText = modal.textContent.toLowerCase();
      if (modalText.includes('apply') || modalText.includes('resume') ||
          modalText.includes('submit') || modalText.includes('應徵')) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Get current step in the application flow
 */
function getApplicationStep() {
  const modal = document.querySelector('.jobs-easy-apply-modal, [role="dialog"]');
  if (!modal) return null;

  const text = modal.textContent.toLowerCase();

  // Check for different steps
  if (text.includes('contact info') || text.includes('聯絡資訊')) {
    return 'contact';
  } else if (text.includes('resume') || text.includes('履歷')) {
    return 'resume';
  } else if (text.includes('additional questions') || text.includes('其他問題')) {
    return 'questions';
  } else if (text.includes('review') || text.includes('檢閱')) {
    return 'review';
  } else if (text.includes('submitted') || text.includes('已送出')) {
    return 'submitted';
  }

  return 'unknown';
}

/**
 * Click Next/Continue/Submit button in modal
 */
function clickNextButton() {
  const buttonTexts = ['Next', 'Continue', 'Submit application', 'Submit', 'Review',
                       '下一步', '繼續', '送出申請', '送出', '檢閱'];

  const buttons = document.querySelectorAll('.jobs-easy-apply-modal button, [role="dialog"] button');

  for (const button of buttons) {
    const text = button.textContent.trim();
    if (buttonTexts.some(t => text.toLowerCase().includes(t.toLowerCase()))) {
      // Don't click disabled buttons
      if (!button.disabled && button.offsetParent !== null) {
        button.click();
        return { success: true, buttonText: text };
      }
    }
  }

  return { success: false, reason: 'Next button not found' };
}

/**
 * Close application modal (dismiss/cancel)
 */
function closeModal() {
  const closeSelectors = [
    'button[data-test-modal-close-btn]',
    '.artdeco-modal__dismiss',
    'button[aria-label="Dismiss"]',
    'button[aria-label="關閉"]'
  ];

  for (const selector of closeSelectors) {
    const closeBtn = document.querySelector(selector);
    if (closeBtn) {
      closeBtn.click();
      return { success: true };
    }
  }

  // Try ESC key
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', keyCode: 27 }));
  return { success: true, method: 'escape' };
}

/**
 * Handle discard confirmation dialog
 */
function handleDiscardDialog() {
  const discardBtn = Array.from(document.querySelectorAll('button'))
    .find(btn => btn.textContent.toLowerCase().includes('discard'));

  if (discardBtn) {
    discardBtn.click();
    return true;
  }
  return false;
}

/**
 * Check if application was successful
 */
function isApplicationSuccessful() {
  const successIndicators = [
    '.jobs-apply-success',
    '.artdeco-inline-feedback--success',
    '[data-test-modal-container] .artdeco-modal__content'
  ];

  for (const selector of successIndicators) {
    const el = document.querySelector(selector);
    if (el) {
      const text = el.textContent.toLowerCase();
      if (text.includes('submitted') || text.includes('success') ||
          text.includes('已送出') || text.includes('成功')) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Scroll to load more jobs
 */
function scrollToLoadMore() {
  const jobsList = document.querySelector('.jobs-search-results-list, .scaffold-layout__list');
  if (jobsList) {
    jobsList.scrollTop = jobsList.scrollHeight;
    return true;
  }
  window.scrollTo(0, document.body.scrollHeight);
  return true;
}

/**
 * Go to next page of results
 */
function goToNextPage() {
  const nextBtn = document.querySelector('button[aria-label="Next"], button[aria-label="下一頁"]');
  if (nextBtn && !nextBtn.disabled) {
    nextBtn.click();
    return true;
  }

  // Try pagination links
  const currentPage = document.querySelector('[aria-current="true"], .artdeco-pagination__indicator--number.active');
  if (currentPage) {
    const nextPage = currentPage.nextElementSibling;
    if (nextPage && nextPage.querySelector('button')) {
      nextPage.querySelector('button').click();
      return true;
    }
  }

  return false;
}

// Export functions for browser console usage
if (typeof window !== 'undefined') {
  window.linkedinApply = {
    CONFIG,
    buildSearchUrl,
    createStatusIndicator,
    updateStatusIndicator,
    setupKeyboardControls,
    getJobCards,
    clickJobCard,
    findEasyApplyButton,
    clickEasyApply,
    isApplicationModalOpen,
    getApplicationStep,
    clickNextButton,
    closeModal,
    handleDiscardDialog,
    isApplicationSuccessful,
    scrollToLoadMore,
    goToNextPage
  };

  console.log('✅ LinkedIn Automation Helper loaded!');
  console.log('Available at: window.linkedinApply');
  console.log('\nQuick start:');
  console.log('  1. linkedinApply.setupKeyboardControls()');
  console.log('  2. linkedinApply.createStatusIndicator()');
  console.log('  3. linkedinApply.getJobCards()');
}

module.exports = {
  CONFIG,
  buildSearchUrl,
  createStatusIndicator,
  updateStatusIndicator,
  setupKeyboardControls,
  getJobCards,
  clickJobCard,
  findEasyApplyButton,
  clickEasyApply,
  isApplicationModalOpen,
  getApplicationStep,
  clickNextButton,
  closeModal,
  handleDiscardDialog,
  isApplicationSuccessful,
  scrollToLoadMore,
  goToNextPage
};
