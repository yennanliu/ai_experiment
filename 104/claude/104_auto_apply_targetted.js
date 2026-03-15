/**
 * 104.com.tw Job Application Automation with Manual Controls
 *
 * Features:
 * - Press 'P' to PAUSE automation
 * - Press 'R' to RESUME automation
 * - Press 'Q' to QUIT automation
 * - Visual indicator shows current status
 *
 * This prevents interference with manual control
 */

// Global state
let automationState = {
  isPaused: false,
  shouldQuit: false,
  currentJob: 0,
  statusElement: null
};

/**
 * Create status indicator on page
 */
function createStatusIndicator(page, { target = 0, current = 0 }) {
  return page.evaluate(({ target, current }) => {
    // Remove existing indicator if any
    const existing = document.getElementById('automation-status');
    if (existing) existing.remove();

    // Create status box
    const statusBox = document.createElement('div');
    statusBox.id = 'automation-status';
    statusBox.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #4CAF50;
      color: white;
      padding: 15px 20px;
      border-radius: 8px;
      font-family: monospace;
      font-size: 14px;
      font-weight: bold;
      z-index: 999999;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      cursor: pointer;
      user-select: none;
    `;
    statusBox.innerHTML = `
      🤖 AUTOMATION RUNNING<br>
      <small style="font-size: 11px;">Progress: ${current}/${target}</small><br>
      <small style="font-size: 11px;">P=Pause | R=Resume | Q=Quit</small>
    `;
    document.body.appendChild(statusBox);

    return true;
  }, { target, current });
}

/**
 * Update status indicator
 */
function updateStatus(page, { status, target = 0, current = 0 }) {
  return page.evaluate(({ status, target, current }) => {
    const statusBox = document.getElementById('automation-status');
    if (!statusBox) return;

    if (status === 'RUNNING') {
      statusBox.style.background = '#4CAF50';
      statusBox.innerHTML = `
        🤖 AUTOMATION RUNNING<br>
        <small style="font-size: 11px;">Progress: ${current}/${target}</small><br>
        <small style="font-size: 11px;">P=Pause | R=Resume | Q=Quit</small>
      `;
    } else if (status === 'PAUSED') {
      statusBox.style.background = '#FF9800';
      statusBox.innerHTML = `
        ⏸️ AUTOMATION PAUSED<br>
        <small style="font-size: 11px;">Progress: ${current}/${target}</small><br>
        <small style="font-size: 11px;">Press R to Resume | Q to Quit</small>
      `;
    } else if (status === 'COMPLETED') {
      statusBox.style.background = '#2196F3';
      statusBox.innerHTML = `
        ✅ AUTOMATION COMPLETED<br>
        <small style="font-size: 11px;">Applied to ${current - successfulSoFar} new jobs.</small><br>
        <small style="font-size: 11px;">Total: ${current}/${target}</small>
      `;
    }
  }, { status, target, current });
}

/**
 * Setup keyboard controls
 */
function setupKeyboardControls(page) {
  return page.evaluate(() => {
    window.automationControl = {
      paused: false,
      quit: false
    };

    document.addEventListener('keydown', (e) => {
      if (e.key.toLowerCase() === 'p') {
        window.automationControl.paused = true;
        console.log('⏸️  AUTOMATION PAUSED by user');
      } else if (e.key.toLowerCase() === 'r') {
        window.automationControl.paused = false;
        console.log('▶️  AUTOMATION RESUMED by user');
      } else if (e.key.toLowerCase() === 'q') {
        window.automationControl.quit = true;
        console.log('🛑 AUTOMATION QUIT by user');
      }
    });

    return true;
  });
}

/**
 * Check if user wants to pause/quit
 */
async function checkUserControl(page) {
  const control = await page.evaluate(() => {
    return {
      paused: window.automationControl?.paused || false,
      quit: window.automationControl?.quit || false
    };
  });

  if (control.quit) {
    throw new Error('USER_QUIT');
  }

  if (control.paused) {
    await updateStatus(page, 'PAUSED');
    console.log('⏸️  Automation paused. Press R to resume...');

    // Wait until resumed
    while (true) {
      await page.waitForTimeout(1000);
      const newControl = await page.evaluate(() => ({
        paused: window.automationControl?.paused || false,
        quit: window.automationControl?.quit || false
      }));

      if (newControl.quit) {
        throw new Error('USER_QUIT');
      }

      if (!newControl.paused) {
        await updateStatus(page, 'RUNNING');
        console.log('▶️  Automation resumed!');
        break;
      }
    }
  }
}

/**
 * Main automation with controls
 */
async function autoApplyWithTarget(page, { searchUrl, startPage = 1, maxPages = 100, target = 0, successfulSoFar = 0 }) {
  console.log('🚀 Starting Automation with Manual Controls...\n');
  console.log('Controls:');
  console.log('  P = Pause automation');
  console.log('  R = Resume automation');
  console.log('  Q = Quit automation');
  console.log('\n');

  const results = {
    totalJobs: 0,
    successful: 0,
    failed: 0,
    skipped: 0,
    pages: []
  };

  try {
    // Setup controls
    await createStatusIndicator(page, { target, current: successfulSoFar });
    await setupKeyboardControls(page);
    await updateStatus(page, { status: 'RUNNING', target, current: successfulSoFar });

    for (let pageNum = startPage; pageNum <= maxPages; pageNum++) {
      if (target > 0 && successfulSoFar + results.successful >= target) {
        console.log(`\n🎯 Target of ${target} reached. Stopping.`);
        break;
      }
      // Check if user wants to quit
      await checkUserControl(page);

      console.log(`\n╔════════════════════════════════════════╗`);
      console.log(`║         PAGE ${pageNum} | Progress: ${successfulSoFar + results.successful}/${target}        ║`);
      console.log(`╚════════════════════════════════════════╝\n`);

      const pageResults = {
        pageNumber: pageNum,
        jobs: [],
        successful: 0,
        failed: 0,
        skipped: 0
      };

      // Navigate to page
      const pageUrl = `${searchUrl}&page=${pageNum}`;
      await page.goto(pageUrl, { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(2000);

      // Re-setup controls after navigation
      await setupKeyboardControls(page);

      const jobCount = await page.evaluate(() => {
        return document.querySelectorAll('.apply-button__button').length;
      });

      if (jobCount === 0) {
        console.log('⚠️  No jobs found. Stopping.');
        break;
      }

      console.log(`Found ${jobCount} jobs\n`);

      // Process each job
      for (let jobIndex = 0; jobIndex < jobCount; jobIndex++) {
        // Check for pause/quit before each job
        await checkUserControl(page);

        if (target > 0 && successfulSoFar + results.successful >= target) {
          console.log(`🎯 Target of ${target} reached. Stopping job processing.`);
          break;
        }

        console.log(`\n━━━━ Job ${jobIndex + 1}/${jobCount} ━━━━`);

        try {
          // Get job info
          const jobInfo = await page.evaluate((index) => {
            const containers = document.querySelectorAll('[class*="job-list-container"]');
            if (index >= containers.length) return null;

            const container = containers[index];
            const titleLink = container.querySelector('a[href*="/job/"]');
            const title = titleLink ? titleLink.textContent.trim() : 'Unknown';

            const companyEl = container.querySelector('[class*="company"]');
            const company = companyEl ? companyEl.textContent.trim() : 'Unknown';

            const alreadyApplied = container.textContent.includes('已應徵');

            return {
              title: title.substring(0, 60),
              company: company.substring(0, 40),
              alreadyApplied
            };
          }, jobIndex);

          if (!jobInfo) break;

          console.log(`📋 ${jobInfo.title}`);

          if (jobInfo.alreadyApplied) {
            console.log('⏭️  SKIPPED');
            pageResults.skipped++;
            await page.waitForTimeout(1000);
            continue;
          }

          // Apply
          await page.evaluate((index) => {
            document.querySelectorAll('.apply-button__button')[index].click();
          }, jobIndex);
          await page.waitForTimeout(1000);

          // Switch to new tab
          const pages = page.context().pages();
          const newTab = pages[pages.length - 1];
          await newTab.bringToFront();
          await newTab.waitForTimeout(1000);

          // Select cover letter
          await newTab.evaluate(() => {
            const span = Array.from(document.querySelectorAll('span'))
              .find(el => el.textContent === '系統預設' && el.tagName === 'SPAN');
            if (span?.parentElement) span.parentElement.click();
          });
          await newTab.waitForTimeout(500);

          await newTab.evaluate(() => {
            const options = document.querySelectorAll('.multiselect__option');
            options.forEach(opt => {
              if (opt.textContent.trim() === '自訂推薦信1') opt.click();
            });
          });
          await newTab.waitForTimeout(500);

          // Submit
          await newTab.evaluate(() => {
            const buttons = document.querySelectorAll('button');
            buttons.forEach(btn => {
              if (btn.textContent.includes('確認送出')) btn.click();
            });
          });
          await newTab.waitForTimeout(2000);

          // Verify
          const finalUrl = newTab.url();
          if (finalUrl.includes('/job/apply/done/')) {
            console.log('✅ SUCCESS');
            pageResults.successful++;
            await updateStatus(page, { status: 'RUNNING', target, current: successfulSoFar + results.successful + pageResults.successful });
          } else {
            throw new Error('Not successful');
          }

          // Cleanup
          await newTab.close();
          await page.bringToFront();
          await page.waitForTimeout(500);

          // Re-setup controls after returning to main page
          await setupKeyboardControls(page);

        } catch (error) {
          console.error(`❌ FAILED: ${error.message}`);
          pageResults.failed++;

          try {
            const pages = page.context().pages();
            if (pages.length > 1) await pages[pages.length - 1].close();
            await page.bringToFront();
            await setupKeyboardControls(page);
          } catch (e) {}
        }

        // Random delay
        const delay = 2000 + Math.random() * 2000;
        await page.waitForTimeout(delay);
      }

      // Page summary
      console.log(`\nPage ${pageNum}: ✅${pageResults.successful} ❌${pageResults.failed} ⏭️${pageResults.skipped}\n`);

      results.pages.push(pageResults);
      results.totalJobs += pageResults.jobs.length;
      results.successful += pageResults.successful;
      results.failed += pageResults.failed;
      results.skipped += pageResults.skipped;

      // Delay before next page
      await page.waitForTimeout(5000);
    }

    // Completed
    await updateStatus(page, { status: 'COMPLETED', target, current: successfulSoFar + results.successful });

  } catch (error) {
    if (error.message === 'USER_QUIT') {
      console.log('\n🛑 Automation stopped by user');
      await updateStatus(page, { status: 'COMPLETED', target, current: successfulSoFar + results.successful });
    } else {
      throw error;
    }
  }

  // Final summary
  console.log('\n');
  console.log('╔══════════════════════════════════════════════╗');
  console.log('║     AUTOMATION RUN COMPLETE (FOR KEYWORD)    ║');
  console.log('╚══════════════════════════════════════════════╝');
  console.log(`✅ Successful: ${results.successful}`);
  console.log(`❌ Failed: ${results.failed}`);
  console.log(`⏭️  Skipped: ${results.skipped}`);
  console.log(`📄 Pages: ${results.pages.length}\n`);

  return results;
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { autoApplyWithTarget };
}
