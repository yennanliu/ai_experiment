/**
 * LinkedIn Job Application - Standalone Browser Script
 *
 * INSTRUCTIONS:
 * 1. Open LinkedIn job search: https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=Singapore&f_AL=true&f_TPR=r2592000
 * 2. Open browser console (F12 or Cmd+Option+J)
 * 3. Paste this ENTIRE script and press Enter
 * 4. The script will run automatically until it reaches 200 applications
 * 5. You can pause/resume or check status anytime
 *
 * COMMANDS:
 * - pauseAutomation()  - Pause the automation
 * - resumeAutomation() - Resume the automation
 * - checkStatus()      - Check current progress
 * - resetProgress()    - Reset and start from 0 (careful!)
 */

(function() {
  'use strict';

  // Configuration
  const CONFIG = {
    TARGET_APPLICATIONS: 100,
    JOBS_PER_BATCH: 5,
    DELAY_BETWEEN_JOBS: 2500,
    TIMEOUT_PER_JOB: 120000,
    BASE_URL: 'https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=UK&f_AL=true&f_TPR=r604800&f_WT=2,3'
  };

  // Persistent state in localStorage
  const getState = () => {
    const saved = localStorage.getItem('linkedinAutoApplyState');
    return saved ? JSON.parse(saved) : {
      applied: 0,
      skipped: 0,
      failed: 0,
      currentPage: 0,
      isPaused: false,
      shouldStop: false
    };
  };

  const setState = (updates) => {
    const current = getState();
    const newState = { ...current, ...updates };
    localStorage.setItem('linkedinAutoApplyState', JSON.stringify(newState));
    return newState;
  };

  // Initialize state
  let state = getState();

  // Create status UI
  function createStatusUI() {
    let el = document.getElementById('linkedin-auto-status');
    if (el) return el;

    el = document.createElement('div');
    el.id = 'linkedin-auto-status';
    el.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: linear-gradient(135deg, #0077B5 0%, #00A0DC 100%);
      color: white;
      padding: 20px;
      border-radius: 12px;
      font-family: system-ui, -apple-system, sans-serif;
      font-size: 14px;
      z-index: 999999;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      min-width: 280px;
      backdrop-filter: blur(10px);
    `;
    document.body.appendChild(el);
    return el;
  }

  function updateStatusUI() {
    const el = createStatusUI();
    const pct = Math.round((state.applied / CONFIG.TARGET_APPLICATIONS) * 100);
    const status = state.isPaused ? 'PAUSED' : state.shouldStop ? 'STOPPED' : 'RUNNING';
    const statusColors = { RUNNING: '#4CAF50', PAUSED: '#FF9800', STOPPED: '#f44336' };

    el.innerHTML = `
      <div style="font-weight:700;font-size:16px;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
        <span style="width:12px;height:12px;border-radius:50%;background:${statusColors[status]};"></span>
        ${status}
      </div>
      <div style="font-size:13px;opacity:0.95;line-height:1.6;">
        <div style="margin-bottom:8px;">
          <strong>${state.applied}</strong> / ${CONFIG.TARGET_APPLICATIONS} <span style="opacity:0.8;">(${pct}%)</span>
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.2);padding-top:8px;margin-top:8px;">
          <div>✅ Applied: ${state.applied}</div>
          <div>⏭️  Skipped: ${state.skipped}</div>
          <div>❌ Failed: ${state.failed}</div>
          <div>📄 Page: ${state.currentPage + 1}</div>
        </div>
      </div>
      <div style="margin-top:12px;font-size:11px;opacity:0.7;border-top:1px solid rgba(255,255,255,0.2);padding-top:8px;">
        Console: pauseAutomation() | resumeAutomation()
      </div>
    `;
  }

  // Helper functions
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  function findEasyApplyButton() {
    let btn = document.querySelector('button.jobs-apply-button');
    if (btn && !btn.disabled && btn.offsetParent !== null) return btn;

    btn = document.querySelector('button[aria-label*="Easy Apply"]');
    if (btn && !btn.disabled && btn.offsetParent !== null) return btn;

    const buttons = Array.from(document.querySelectorAll('button'));
    for (const b of buttons) {
      const text = (b.textContent || '').toLowerCase();
      if (text.includes('easy apply') && !b.disabled && b.offsetParent !== null) return b;
    }
    return null;
  }

  function fillFormFields() {
    document.querySelectorAll('input[type="number"]').forEach(inp => {
      if (!inp.value || inp.value === '' || inp.value === '0') {
        inp.value = '5';
        inp.dispatchEvent(new Event('input', { bubbles: true }));
        inp.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });

    document.querySelectorAll('input[type="radio"]').forEach(radio => {
      const label = radio.closest('label') || document.querySelector(`label[for="${radio.id}"]`);
      if (label && label.textContent.toLowerCase().includes('yes') && !radio.checked) {
        radio.click();
      }
    });
  }

  async function applyToJob(jobIndex) {
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('TIMEOUT')), CONFIG.TIMEOUT_PER_JOB)
    );

    const applyPromise = (async () => {
      const jobItems = document.querySelectorAll('[data-job-id]');
      if (jobIndex >= jobItems.length) {
        return { success: false, error: 'OUT_OF_RANGE' };
      }

      const jobItem = jobItems[jobIndex];
      const titleEl = jobItem.querySelector('a');
      const jobTitle = titleEl?.textContent?.trim() || 'Unknown Job';

      if (jobItem.textContent.includes('Applied')) {
        return { success: false, error: 'ALREADY_APPLIED', title: jobTitle };
      }

      jobItem.click();
      await sleep(1500);

      const easyApplyBtn = findEasyApplyButton();
      if (!easyApplyBtn) {
        return { success: false, error: 'NO_EASY_APPLY', title: jobTitle };
      }

      easyApplyBtn.click();
      await sleep(2500);

      let stepsCompleted = 0;
      for (let step = 0; step < 15; step++) {
        fillFormFields();

        const buttons = Array.from(document.querySelectorAll('button')).filter(b =>
          !b.disabled && b.offsetParent !== null
        );

        const priority = ['submit application', 'submit', 'continue', 'next', 'review'];
        let clickedBtn = null;

        for (const p of priority) {
          clickedBtn = buttons.find(b => b.textContent.toLowerCase().includes(p));
          if (clickedBtn) {
            clickedBtn.click();
            stepsCompleted++;
            break;
          }
        }

        if (!clickedBtn) break;
        await sleep(1000);
      }

      await sleep(2000);

      const closeBtn = document.querySelector('button[aria-label="Dismiss"]');
      if (closeBtn) {
        closeBtn.click();
        await sleep(500);
      }

      return { success: true, title: jobTitle };
    })();

    try {
      return await Promise.race([applyPromise, timeoutPromise]);
    } catch (err) {
      // Cleanup on timeout
      const closeBtn = document.querySelector('button[aria-label="Dismiss"]');
      if (closeBtn) closeBtn.click();

      const discardBtn = Array.from(document.querySelectorAll('button')).find(b =>
        b.textContent.toLowerCase().includes('discard')
      );
      if (discardBtn) discardBtn.click();

      return { success: false, error: 'TIMEOUT' };
    }
  }

  async function processBatch() {
    console.log(`\n🚀 Processing batch on page ${state.currentPage + 1}...`);

    const jobItems = document.querySelectorAll('[data-job-id]');
    const batchSize = Math.min(CONFIG.JOBS_PER_BATCH, jobItems.length);

    for (let i = 0; i < batchSize; i++) {
      if (state.shouldStop || state.isPaused) {
        console.log('⏸️  Automation paused or stopped');
        return false;
      }

      if (state.applied >= CONFIG.TARGET_APPLICATIONS) {
        console.log(`🎯 Target reached! ${state.applied} applications submitted.`);
        state = setState({ shouldStop: true });
        return false;
      }

      const result = await applyToJob(i);

      if (result.error) {
        state = setState({ skipped: state.skipped + 1 });
        console.log(`⏭️  [${i}] Skipped: ${result.title || 'Unknown'} (${result.error})`);
      } else {
        state = setState({ applied: state.applied + 1 });
        console.log(`✅ [${i}] Applied: ${result.title}`);
      }

      updateStatusUI();
      await sleep(CONFIG.DELAY_BETWEEN_JOBS);
    }

    return true;
  }

  async function navigateToNextPage() {
    state = setState({ currentPage: state.currentPage + 1 });
    const start = state.currentPage * 25;
    const url = `${CONFIG.BASE_URL}&start=${start}`;

    console.log(`📄 Navigating to page ${state.currentPage + 1}...`);
    window.location.href = url;
    await sleep(3000);
  }

  async function runAutomation() {
    console.log(`
╔═══════════════════════════════════════════════════════╗
║      LinkedIn Auto-Apply Automation Started           ║
╠═══════════════════════════════════════════════════════╣
║  Target: ${CONFIG.TARGET_APPLICATIONS} applications                           ║
║  Current: ${state.applied} applied, ${state.skipped} skipped               ║
╚═══════════════════════════════════════════════════════╝
    `);

    updateStatusUI();

    while (!state.shouldStop && state.applied < CONFIG.TARGET_APPLICATIONS) {
      if (state.isPaused) {
        await sleep(1000);
        continue;
      }

      const shouldContinue = await processBatch();

      if (!shouldContinue) break;

      // Navigate to next page after batch
      await navigateToNextPage();
      await sleep(3000); // Wait for page load
    }

    console.log(`
╔═══════════════════════════════════════════════════════╗
║           Automation Complete!                        ║
╠═══════════════════════════════════════════════════════╣
║  ✅ Applied: ${state.applied}                                  ║
║  ⏭️  Skipped: ${state.skipped}                                  ║
║  ❌ Failed: ${state.failed}                                   ║
╚═══════════════════════════════════════════════════════╝
    `);
  }

  // Global functions for user control
  window.pauseAutomation = () => {
    state = setState({ isPaused: true });
    updateStatusUI();
    console.log('⏸️  Automation PAUSED');
  };

  window.resumeAutomation = () => {
    state = setState({ isPaused: false });
    updateStatusUI();
    console.log('▶️  Automation RESUMED');
  };

  window.checkStatus = () => {
    console.log(`
    Status:
      Applied: ${state.applied}/${CONFIG.TARGET_APPLICATIONS}
      Skipped: ${state.skipped}
      Failed: ${state.failed}
      Current Page: ${state.currentPage + 1}
      Paused: ${state.isPaused}
    `);
    return state;
  };

  window.resetProgress = () => {
    if (confirm('Are you sure you want to reset all progress?')) {
      localStorage.removeItem('linkedinAutoApplyState');
      state = getState();
      updateStatusUI();
      console.log('🔄 Progress reset');
    }
  };

  // Start automation
  console.log('✅ Automation script loaded!');
  console.log('Starting in 3 seconds...');
  console.log('Use pauseAutomation() to pause, resumeAutomation() to resume');

  setTimeout(() => {
    runAutomation();
  }, 3000);

})();
