/**
 * LinkedIn Job Application Automation - Runner Script
 *
 * This script provides the automation flow for LinkedIn job applications.
 * It's designed to be run step-by-step using Chrome DevTools MCP tools.
 *
 * IMPORTANT: This uses chrome-cdp-skill to connect to your existing
 * Chrome browser session. No login handling needed!
 *
 * Usage:
 *   1. Enable Chrome remote debugging (chrome://inspect/#remote-debugging)
 *   2. Log into LinkedIn in Chrome
 *   3. Run this automation via Claude Code MCP tools
 */

// ============================================================================
// CONFIGURATION - Edit these values for your search
// ============================================================================

const CONFIG = {
  // Search Parameters
  jobTitle: 'Software Engineer',
  locations: ['Taiwan', 'Remote'],
  datePosted: 'Past week', // Options: 'Any time', 'Past month', 'Past week', 'Past 24 hours'

  // Filters
  easyApplyOnly: true,
  remoteOptions: ['Remote', 'Hybrid'], // Options: 'On-site', 'Remote', 'Hybrid'
  experienceLevel: [], // Options: 'Entry level', 'Mid-Senior level', 'Director', etc.

  // Automation Settings
  maxApplications: 50,
  delayMin: 3000,
  delayMax: 6000,
  maxPages: 10
};

// ============================================================================
// URL BUILDER
// ============================================================================

function buildSearchUrl(config = CONFIG) {
  const params = new URLSearchParams();

  params.set('keywords', config.jobTitle);

  if (config.locations?.length > 0) {
    params.set('location', config.locations[0]);
  }

  if (config.easyApplyOnly) {
    params.set('f_AL', 'true');
  }

  const dateMap = {
    'Past 24 hours': 'r86400',
    'Past week': 'r604800',
    'Past month': 'r2592000'
  };
  if (config.datePosted && dateMap[config.datePosted]) {
    params.set('f_TPR', dateMap[config.datePosted]);
  }

  const remoteMap = { 'On-site': '1', 'Remote': '2', 'Hybrid': '3' };
  if (config.remoteOptions?.length > 0) {
    const values = config.remoteOptions.map(o => remoteMap[o]).filter(Boolean).join(',');
    if (values) params.set('f_WT', values);
  }

  return `https://www.linkedin.com/jobs/search/?${params.toString()}`;
}

// ============================================================================
// INJECTION SCRIPT - Run this in the browser via MCP evaluate_script
// ============================================================================

const INJECT_AUTOMATION_HELPERS = `
(function() {
  // Prevent double injection
  if (window.__linkedinAutoLoaded) return { status: 'already_loaded' };
  window.__linkedinAutoLoaded = true;

  // State
  window.autoState = {
    paused: false,
    quit: false,
    applied: 0,
    skipped: 0,
    failed: 0
  };

  // Keyboard controls
  document.addEventListener('keydown', (e) => {
    if (e.key === 'p' || e.key === 'P') {
      window.autoState.paused = true;
      updateUI('PAUSED');
    } else if (e.key === 'r' || e.key === 'R') {
      window.autoState.paused = false;
      updateUI('RUNNING');
    } else if (e.key === 'q' || e.key === 'Q') {
      window.autoState.quit = true;
      updateUI('STOPPED');
    }
  });

  // Status UI
  function createUI() {
    let el = document.getElementById('li-auto-ui');
    if (el) return el;

    el = document.createElement('div');
    el.id = 'li-auto-ui';
    el.style.cssText = 'position:fixed;top:20px;right:20px;background:#0077B5;color:white;padding:15px;border-radius:10px;font-family:system-ui;font-size:13px;z-index:999999;box-shadow:0 4px 15px rgba(0,0,0,0.3);min-width:180px;';
    document.body.appendChild(el);
    return el;
  }

  function updateUI(status) {
    const el = createUI();
    const s = window.autoState;
    const colors = {RUNNING:'#0077B5',PAUSED:'#FF9800',STOPPED:'#f44336',DONE:'#4CAF50'};
    el.style.background = colors[status] || colors.RUNNING;
    el.innerHTML =
      '<div style="font-weight:bold;margin-bottom:6px;">🤖 ' + status + '</div>' +
      '<div style="font-size:11px;opacity:0.8;margin-bottom:8px;">P=Pause R=Resume Q=Quit</div>' +
      '<div style="font-size:12px;">✅ ' + s.applied + ' | ⏭ ' + s.skipped + ' | ❌ ' + s.failed + '</div>';
  }
  window.updateUI = updateUI;

  // Get all job cards
  window.getJobs = function() {
    const cards = [];
    const els = document.querySelectorAll('.job-card-container, [data-job-id], .jobs-search-results__list-item');

    els.forEach((el, i) => {
      const title = el.querySelector('a[class*="job-card"], .job-card-list__title')?.textContent?.trim() || 'Unknown';
      const company = el.querySelector('.job-card-container__company-name, .artdeco-entity-lockup__subtitle')?.textContent?.trim() || '';
      const applied = el.textContent.includes('Applied') || !!el.querySelector('[class*="applied"]');
      const jobId = el.getAttribute('data-job-id') || 'job-' + i;

      cards.push({ index: i, jobId, title, company, applied });
    });

    return cards;
  };

  // Click a job card by index
  window.clickJob = function(idx) {
    const els = document.querySelectorAll('.job-card-container, [data-job-id], .jobs-search-results__list-item');
    if (idx >= els.length) return { ok: false, err: 'Index out of range' };

    const link = els[idx].querySelector('a');
    if (link) { link.click(); return { ok: true }; }
    els[idx].click();
    return { ok: true };
  };

  // Find Easy Apply button
  window.getEasyApplyBtn = function() {
    const btns = document.querySelectorAll('button');
    for (const btn of btns) {
      const txt = btn.textContent.toLowerCase();
      if ((txt.includes('easy apply') || txt.includes('簡易')) && !btn.disabled && btn.offsetParent) {
        return btn;
      }
    }
    return null;
  };

  // Click Easy Apply
  window.clickEasyApply = function() {
    const btn = getEasyApplyBtn();
    if (!btn) return { ok: false, err: 'No Easy Apply button' };
    btn.click();
    return { ok: true };
  };

  // Check modal state
  window.getModalInfo = function() {
    const modal = document.querySelector('.jobs-easy-apply-modal, [role="dialog"]');
    if (!modal || !modal.offsetParent) return { open: false };

    const txt = modal.textContent.toLowerCase();
    let step = 'unknown';
    if (txt.includes('contact')) step = 'contact';
    else if (txt.includes('resume')) step = 'resume';
    else if (txt.includes('question')) step = 'questions';
    else if (txt.includes('review')) step = 'review';
    else if (txt.includes('submitted') || txt.includes('success')) step = 'done';

    // Check for required fields
    const hasRequired = modal.querySelectorAll('[required], [aria-required="true"]').length > 0;

    return { open: true, step, hasRequired };
  };

  // Click next/submit in modal
  window.clickNext = function() {
    const modal = document.querySelector('.jobs-easy-apply-modal, [role="dialog"]');
    if (!modal) return { ok: false, err: 'No modal' };

    const priority = ['submit application', 'submit', 'next', 'review', 'continue', '送出', '下一步'];
    const btns = modal.querySelectorAll('button');

    for (const word of priority) {
      for (const btn of btns) {
        if (btn.textContent.toLowerCase().includes(word) && !btn.disabled && btn.offsetParent) {
          btn.click();
          return { ok: true, clicked: btn.textContent.trim() };
        }
      }
    }
    return { ok: false, err: 'No button found' };
  };

  // Close modal
  window.closeModal = function() {
    const close = document.querySelector('button[aria-label="Dismiss"], .artdeco-modal__dismiss');
    if (close) { close.click(); return { ok: true }; }
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    return { ok: true, method: 'esc' };
  };

  // Handle discard dialog
  window.discardDraft = function() {
    const btns = document.querySelectorAll('button');
    for (const btn of btns) {
      if (btn.textContent.toLowerCase().includes('discard')) {
        btn.click();
        return true;
      }
    }
    return false;
  };

  // Check if application succeeded
  window.checkSuccess = function() {
    const txt = document.body.textContent.toLowerCase();
    return txt.includes('application submitted') || txt.includes('已送出');
  };

  // Update counters
  window.incApplied = function() { window.autoState.applied++; updateUI('RUNNING'); return window.autoState; };
  window.incSkipped = function() { window.autoState.skipped++; updateUI('RUNNING'); return window.autoState; };
  window.incFailed = function() { window.autoState.failed++; updateUI('RUNNING'); return window.autoState; };

  // Initialize
  createUI();
  updateUI('RUNNING');
  console.log('✅ LinkedIn automation ready! Press P/R/Q to control.');

  return { status: 'loaded', jobs: getJobs().length };
})();
`;

// ============================================================================
// MCP TOOL COMMANDS - Copy and run these in Claude Code
// ============================================================================

const MCP_COMMANDS = `
// ============================================================================
// STEP-BY-STEP MCP COMMANDS FOR LINKEDIN AUTOMATION
// Run these in Claude Code to automate LinkedIn job applications
// ============================================================================

// STEP 0: List Chrome pages to find LinkedIn
mcp__chrome-devtools__list_pages()

// STEP 1: Navigate to LinkedIn Jobs (edit URL as needed)
mcp__chrome-devtools__navigate_page({
  type: 'url',
  url: '${buildSearchUrl(CONFIG)}'
})

// STEP 2: Wait for page to load
mcp__chrome-devtools__wait_for({ text: ['Easy Apply', 'jobs'] })

// STEP 3: Take snapshot to see current state
mcp__chrome-devtools__take_snapshot()

// STEP 4: Inject automation helpers
mcp__chrome-devtools__evaluate_script({
  function: \`${INJECT_AUTOMATION_HELPERS.replace(/`/g, '\\`')}\`
})

// STEP 5: Get list of jobs
mcp__chrome-devtools__evaluate_script({
  function: "() => window.getJobs()"
})

// STEP 6: Click on first job
mcp__chrome-devtools__evaluate_script({
  function: "() => window.clickJob(0)"
})

// STEP 7: Wait and click Easy Apply
mcp__chrome-devtools__wait_for({ time: 1 })
mcp__chrome-devtools__evaluate_script({
  function: "() => window.clickEasyApply()"
})

// STEP 8: Check modal state
mcp__chrome-devtools__evaluate_script({
  function: "() => window.getModalInfo()"
})

// STEP 9: Click through modal steps (repeat until done)
mcp__chrome-devtools__evaluate_script({
  function: "() => window.clickNext()"
})

// STEP 10: Check if successful
mcp__chrome-devtools__evaluate_script({
  function: "() => window.checkSuccess()"
})

// STEP 11: Increment counter and close
mcp__chrome-devtools__evaluate_script({
  function: "() => { window.incApplied(); window.closeModal(); return window.autoState; }"
})

// STEP 12: Check automation state (for pause/quit)
mcp__chrome-devtools__evaluate_script({
  function: "() => window.autoState"
})
`;

// ============================================================================
// AUTOMATION LOOP - For reference / manual execution
// ============================================================================

async function automationLoop(page) {
  // This is a reference implementation
  // In practice, run steps via MCP tools

  console.log('Starting LinkedIn automation...');
  console.log(`Search URL: ${buildSearchUrl(CONFIG)}`);

  let totalApplied = 0;
  let currentPage = 1;

  while (totalApplied < CONFIG.maxApplications && currentPage <= CONFIG.maxPages) {
    // 1. Get jobs on current page
    const jobs = await page.evaluate(() => window.getJobs());
    console.log(`Page ${currentPage}: Found ${jobs.length} jobs`);

    // 2. Process each job
    for (let i = 0; i < jobs.length && totalApplied < CONFIG.maxApplications; i++) {
      const job = jobs[i];

      // Check pause/quit
      const state = await page.evaluate(() => window.autoState);
      if (state.quit) {
        console.log('Quit requested');
        return;
      }
      while (state.paused) {
        await new Promise(r => setTimeout(r, 1000));
      }

      // Skip if already applied
      if (job.applied) {
        console.log(`Skipping (applied): ${job.title}`);
        await page.evaluate(() => window.incSkipped());
        continue;
      }

      console.log(`Applying to: ${job.title} at ${job.company}`);

      try {
        // Click job
        await page.evaluate((idx) => window.clickJob(idx), i);
        await new Promise(r => setTimeout(r, 1500));

        // Click Easy Apply
        const easyApply = await page.evaluate(() => window.clickEasyApply());
        if (!easyApply.ok) {
          console.log(`  Skip: ${easyApply.err}`);
          await page.evaluate(() => window.incSkipped());
          continue;
        }

        await new Promise(r => setTimeout(r, 1000));

        // Complete modal steps
        let maxSteps = 10;
        while (maxSteps-- > 0) {
          const modal = await page.evaluate(() => window.getModalInfo());

          if (!modal.open) break;
          if (modal.step === 'done') break;

          const next = await page.evaluate(() => window.clickNext());
          if (!next.ok) break;

          await new Promise(r => setTimeout(r, 1000));
        }

        // Check success
        const success = await page.evaluate(() => window.checkSuccess());
        if (success) {
          console.log(`  SUCCESS`);
          await page.evaluate(() => window.incApplied());
          totalApplied++;
        } else {
          console.log(`  FAILED`);
          await page.evaluate(() => window.incFailed());
        }

        // Close modal
        await page.evaluate(() => window.closeModal());
        await new Promise(r => setTimeout(r, 500));
        await page.evaluate(() => window.discardDraft());

      } catch (err) {
        console.log(`  ERROR: ${err.message}`);
        await page.evaluate(() => window.incFailed());
        await page.evaluate(() => { window.closeModal(); window.discardDraft(); });
      }

      // Random delay
      const delay = CONFIG.delayMin + Math.random() * (CONFIG.delayMax - CONFIG.delayMin);
      await new Promise(r => setTimeout(r, delay));
    }

    // Go to next page
    const hasNext = await page.evaluate(() => {
      const btn = document.querySelector('button[aria-label="Next"]');
      if (btn && !btn.disabled) { btn.click(); return true; }
      return false;
    });

    if (!hasNext) break;
    currentPage++;
    await new Promise(r => setTimeout(r, 2000));
  }

  console.log(`\nAutomation complete!`);
  console.log(`Applied: ${totalApplied}`);
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  CONFIG,
  buildSearchUrl,
  INJECT_AUTOMATION_HELPERS,
  MCP_COMMANDS,
  automationLoop
};

// Print info on load
console.log('='.repeat(60));
console.log('LinkedIn Job Application Automation');
console.log('='.repeat(60));
console.log('\nConfiguration:');
console.log(`  Job Title: ${CONFIG.jobTitle}`);
console.log(`  Locations: ${CONFIG.locations.join(', ')}`);
console.log(`  Date Posted: ${CONFIG.datePosted}`);
console.log(`  Easy Apply Only: ${CONFIG.easyApplyOnly}`);
console.log(`  Max Applications: ${CONFIG.maxApplications}`);
console.log('\nSearch URL:');
console.log(`  ${buildSearchUrl(CONFIG)}`);
console.log('\nSee CLAUDE.md for detailed MCP tool usage.');
console.log('='.repeat(60));
