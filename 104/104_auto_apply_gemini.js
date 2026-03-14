#!/usr/bin/env node
/**
 * 104.com.tw Job Application Automation Script
 * Compatible with Gemini CLI / Claude Code / Any CLI that can run Node.js
 *
 * Usage:
 *   node 104_auto_apply_gemini.js [options]
 *
 * Options:
 *   --start-page <num>   Starting page number (default: 2)
 *   --max-pages <num>    Maximum pages to process (default: 5)
 *   --headless           Run in headless mode (default: false)
 *   --help               Show help message
 *
 * Features:
 *   - Press 'P' to PAUSE automation
 *   - Press 'R' to RESUME automation
 *   - Press 'Q' to QUIT automation
 *   - Visual status indicator on page
 *
 * Prerequisites:
 *   - Node.js installed
 *   - Playwright installed: npm install playwright
 *   - Must be logged into 104.com.tw before running
 */

const { chromium } = require('playwright');

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    startPage: 2,
    maxPages: 5,
    headless: false,
    help: false
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--start-page':
        config.startPage = parseInt(args[++i], 10) || 2;
        break;
      case '--max-pages':
        config.maxPages = parseInt(args[++i], 10) || 5;
        break;
      case '--headless':
        config.headless = true;
        break;
      case '--help':
      case '-h':
        config.help = true;
        break;
    }
  }

  return config;
}

function showHelp() {
  console.log(`
104.com.tw Job Application Automation
======================================

Usage:
  node 104_auto_apply_gemini.js [options]

Options:
  --start-page <num>   Starting page number (default: 2)
  --max-pages <num>    Maximum pages to process (default: 5)
  --headless           Run in headless mode (default: false)
  --help, -h           Show this help message

Keyboard Controls (during automation):
  P = Pause automation
  R = Resume automation
  Q = Quit automation

Example:
  node 104_auto_apply_gemini.js --start-page 3 --max-pages 10

Note: You must be logged into 104.com.tw before running this script.
`);
}

// Create status indicator on page
async function createStatusIndicator(page) {
  return page.evaluate(() => {
    const existing = document.getElementById('automation-status');
    if (existing) existing.remove();

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
    `;
    statusBox.innerHTML = `
      🤖 AUTOMATION RUNNING<br>
      <small style="font-size: 11px;">Press P=Pause | R=Resume | Q=Quit</small>
    `;
    document.body.appendChild(statusBox);
    return true;
  });
}

// Update status indicator
async function updateStatus(page, status, extra = '') {
  return page.evaluate(({ status, extra }) => {
    const statusBox = document.getElementById('automation-status');
    if (!statusBox) return;

    const configs = {
      RUNNING: {
        bg: '#4CAF50',
        html: `🤖 AUTOMATION RUNNING ${extra}<br><small style="font-size: 11px;">Press P=Pause | R=Resume | Q=Quit</small>`
      },
      PAUSED: {
        bg: '#FF9800',
        html: `⏸️ AUTOMATION PAUSED<br><small style="font-size: 11px;">Press R to Resume | Q to Quit</small>`
      },
      COMPLETED: {
        bg: '#2196F3',
        html: `✅ AUTOMATION COMPLETED ${extra}<br><small style="font-size: 11px;">You can close this window</small>`
      }
    };

    const config = configs[status];
    if (config) {
      statusBox.style.background = config.bg;
      statusBox.innerHTML = config.html;
    }
  }, { status, extra });
}

// Setup keyboard controls
async function setupKeyboardControls(page) {
  return page.evaluate(() => {
    if (window.automationControlSetup) return true;

    window.automationControl = {
      paused: false,
      quit: false
    };

    document.addEventListener('keydown', (e) => {
      const key = e.key.toLowerCase();
      if (key === 'p') {
        window.automationControl.paused = true;
        console.log('⏸️ AUTOMATION PAUSED by user');
      } else if (key === 'r') {
        window.automationControl.paused = false;
        console.log('▶️ AUTOMATION RESUMED by user');
      } else if (key === 'q') {
        window.automationControl.quit = true;
        console.log('🛑 AUTOMATION QUIT by user');
      }
    });

    window.automationControlSetup = true;
    return true;
  });
}

// Check user control state
async function checkUserControl(page) {
  const control = await page.evaluate(() => ({
    paused: window.automationControl?.paused || false,
    quit: window.automationControl?.quit || false
  }));

  if (control.quit) {
    throw new Error('USER_QUIT');
  }

  if (control.paused) {
    await updateStatus(page, 'PAUSED');
    console.log('⏸️ Automation paused. Press R to resume...');

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
        console.log('▶️ Automation resumed!');
        break;
      }
    }
  }
}

// Random delay helper
function randomDelay(min = 2000, max = 4000) {
  return min + Math.random() * (max - min);
}

// Main automation function
async function autoApplyJobs(config) {
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════╗');
  console.log('║   104.com.tw Job Application Automation            ║');
  console.log('╚════════════════════════════════════════════════════╝');
  console.log('\nKeyboard Controls:');
  console.log('  P = Pause automation');
  console.log('  R = Resume automation');
  console.log('  Q = Quit automation\n');
  console.log(`Starting from page ${config.startPage}, processing ${config.maxPages} pages\n`);

  const browser = await chromium.launch({
    headless: config.headless,
    args: ['--start-maximized']
  });

  const context = await browser.newContext({
    viewport: null,
    storageState: undefined // Will use existing session if available
  });

  const page = await context.newPage();

  const results = {
    successful: 0,
    failed: 0,
    skipped: 0,
    pagesProcessed: 0
  };

  const BASE_URL = 'https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&jobsource=joblist_search&keyword=%20%20%20%20%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15&remoteWork=1,2';

  try {
    // Navigate to first page
    const firstPageUrl = `${BASE_URL}&page=${config.startPage}`;
    console.log('Navigating to 104.com.tw...');
    await page.goto(firstPageUrl, { waitUntil: 'networkidle', timeout: 60000 });

    // Check if logged in
    const isLoggedIn = await page.evaluate(() => {
      return !document.querySelector('a[href*="login"]') ||
             document.querySelector('[class*="member"]') !== null;
    });

    if (!isLoggedIn) {
      console.log('\n⚠️ WARNING: You may not be logged in!');
      console.log('Please log in manually in the browser window.');
      console.log('Press any key in the browser after logging in...\n');
      await page.waitForTimeout(30000); // Wait 30 seconds for manual login
    }

    // Setup controls
    await createStatusIndicator(page);
    await setupKeyboardControls(page);
    await updateStatus(page, 'RUNNING');

    // Process pages
    for (let pageNum = config.startPage; pageNum < config.startPage + config.maxPages; pageNum++) {
      await checkUserControl(page);

      console.log(`\n╔════════════════════════════════════════╗`);
      console.log(`║         PAGE ${pageNum}                        ║`);
      console.log(`╚════════════════════════════════════════╝\n`);

      // Navigate to page
      const pageUrl = `${BASE_URL}&page=${pageNum}`;
      await page.goto(pageUrl, { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(2000);

      // Re-setup controls after navigation
      await setupKeyboardControls(page);
      await createStatusIndicator(page);

      // Count jobs
      const jobCount = await page.evaluate(() => {
        return document.querySelectorAll('.apply-button__button').length;
      });

      if (jobCount === 0) {
        console.log('⚠️ No jobs found on this page. Stopping.');
        break;
      }

      console.log(`Found ${jobCount} jobs\n`);

      let pageSuccessful = 0;
      let pageFailed = 0;
      let pageSkipped = 0;

      // Process each job
      for (let jobIndex = 0; jobIndex < jobCount; jobIndex++) {
        await checkUserControl(page);

        console.log(`\n━━━━ Job ${jobIndex + 1}/${jobCount} ━━━━`);

        try {
          // Get job info
          const jobInfo = await page.evaluate((index) => {
            const containers = document.querySelectorAll('[class*="job-list-container"], article');
            const container = containers[index] || document.body;

            const titleLink = container.querySelector('a[href*="/job/"]');
            const title = titleLink ? titleLink.textContent.trim() : 'Unknown';

            const companyEl = container.querySelector('[class*="company"]');
            const company = companyEl ? companyEl.textContent.trim() : 'Unknown';

            const alreadyApplied = container.textContent.includes('已應徵') ||
                                   container.textContent.includes('今日已應徵');

            return {
              title: title.substring(0, 60),
              company: company.substring(0, 40),
              alreadyApplied
            };
          }, jobIndex);

          console.log(`📋 ${jobInfo.title}`);
          console.log(`🏢 ${jobInfo.company}`);

          if (jobInfo.alreadyApplied) {
            console.log('⏭️ SKIPPED (already applied)');
            pageSkipped++;
            await page.waitForTimeout(500);
            continue;
          }

          // Click apply button
          await page.evaluate((index) => {
            const buttons = document.querySelectorAll('.apply-button__button');
            if (buttons[index]) {
              buttons[index].click();
            }
          }, jobIndex);

          await page.waitForTimeout(1500);

          // Switch to new tab
          const pages = page.context().pages();
          if (pages.length < 2) {
            throw new Error('New tab did not open');
          }

          const newTab = pages[pages.length - 1];
          await newTab.bringToFront();
          await newTab.waitForTimeout(2000);

          // Check if already on success page
          if (newTab.url().includes('/job/apply/done/')) {
            console.log('✅ SUCCESS (already applied)');
            pageSuccessful++;
            await newTab.close();
            await page.bringToFront();
            await setupKeyboardControls(page);
            continue;
          }

          // Select cover letter
          await newTab.evaluate(() => {
            const spans = Array.from(document.querySelectorAll('span'));
            const systemDefault = spans.find(el => el.textContent === '系統預設');
            if (systemDefault?.parentElement) {
              systemDefault.parentElement.click();
            }
          });
          await newTab.waitForTimeout(500);

          await newTab.evaluate(() => {
            const options = document.querySelectorAll('.multiselect__option');
            options.forEach(opt => {
              if (opt.textContent.trim() === '自訂推薦信1') {
                opt.click();
              }
            });
          });
          await newTab.waitForTimeout(500);

          // Click submit
          await newTab.evaluate(() => {
            const buttons = document.querySelectorAll('button');
            buttons.forEach(btn => {
              if (btn.textContent.includes('確認送出')) {
                btn.click();
              }
            });
          });
          await newTab.waitForTimeout(3000);

          // Verify success
          const finalUrl = newTab.url();
          if (finalUrl.includes('/job/apply/done/')) {
            console.log('✅ SUCCESS');
            pageSuccessful++;
          } else {
            throw new Error('Application may not have completed');
          }

          // Close tab and return
          await newTab.close();
          await page.bringToFront();
          await setupKeyboardControls(page);

        } catch (error) {
          console.log(`❌ FAILED: ${error.message}`);
          pageFailed++;

          // Cleanup any open tabs
          try {
            const allPages = page.context().pages();
            for (let i = allPages.length - 1; i > 0; i--) {
              await allPages[i].close();
            }
            await page.bringToFront();
            await setupKeyboardControls(page);
          } catch (e) {
            // Ignore cleanup errors
          }
        }

        // Random delay between jobs
        const delay = randomDelay();
        await page.waitForTimeout(delay);
      }

      // Page summary
      console.log(`\n📊 Page ${pageNum} Summary: ✅${pageSuccessful} ❌${pageFailed} ⏭️${pageSkipped}`);

      results.successful += pageSuccessful;
      results.failed += pageFailed;
      results.skipped += pageSkipped;
      results.pagesProcessed++;

      // Delay before next page
      await page.waitForTimeout(3000);
    }

    await updateStatus(page, 'COMPLETED', `(✅${results.successful} ❌${results.failed})`);

  } catch (error) {
    if (error.message === 'USER_QUIT') {
      console.log('\n🛑 Automation stopped by user');
    } else {
      console.error('\n❌ Error:', error.message);
    }
  }

  // Final summary
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════╗');
  console.log('║           AUTOMATION COMPLETE                    ║');
  console.log('╚══════════════════════════════════════════════════╝');
  console.log(`✅ Successful: ${results.successful}`);
  console.log(`❌ Failed: ${results.failed}`);
  console.log(`⏭️  Skipped: ${results.skipped}`);
  console.log(`📄 Pages Processed: ${results.pagesProcessed}`);
  console.log('\n');

  // Keep browser open for review
  console.log('Browser will stay open. Close it manually when done.');

  return results;
}

// Main entry point
async function main() {
  const config = parseArgs();

  if (config.help) {
    showHelp();
    process.exit(0);
  }

  try {
    await autoApplyJobs(config);
  } catch (error) {
    console.error('Fatal error:', error.message);
    process.exit(1);
  }
}

main();
