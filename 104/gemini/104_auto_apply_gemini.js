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
 *   --skip-login         Skip automatic login (default: false)
 *   --use-existing       Connect to existing Chrome browser (reuse session)
 *   --cdp-port <num>     Chrome DevTools Protocol port (default: 9222)
 *   --help               Show help message
 *
 * Environment Variables (.env file):
 *   JOB_EMAIL            Your 104.com.tw email
 *   JOB_PASSWORD         Your 104.com.tw password
 *   COVER_LETTER         Cover letter name (default: 自訂推薦信1)
 *
 * Features:
 *   - Connect to existing Chrome browser (--use-existing)
 *   - Automatic login using .env credentials
 *   - Press 'P' to PAUSE automation
 *   - Press 'R' to RESUME automation
 *   - Press 'Q' to QUIT automation
 *   - Visual status indicator on page
 *
 * Prerequisites:
 *   - Node.js installed
 *   - Run: npm install (to install playwright and dotenv)
 *   - For --use-existing: Start Chrome with remote debugging enabled
 */

require('dotenv').config();
const { chromium } = require('playwright');
const path = require('path');

// Load credentials from .env
const CREDENTIALS = {
  email: process.env.JOB_EMAIL,
  password: process.env.JOB_PASSWORD,
  coverLetter: process.env.COVER_LETTER || '自訂推薦信1'
};

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    startPage: 2,
    maxPages: 5,
    headless: false,
    skipLogin: false,
    useExisting: false,
    cdpPort: 9222,
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
      case '--skip-login':
        config.skipLogin = true;
        break;
      case '--use-existing':
        config.useExisting = true;
        break;
      case '--cdp-port':
        config.cdpPort = parseInt(args[++i], 10) || 9222;
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
  --skip-login         Skip automatic login (default: false)
  --use-existing       Connect to existing Chrome browser (reuse login session)
  --cdp-port <num>     Chrome DevTools Protocol port (default: 9222)
  --help, -h           Show this help message

Environment Variables (.env file):
  JOB_EMAIL            Your 104.com.tw login email
  JOB_PASSWORD         Your 104.com.tw password
  COVER_LETTER         Cover letter name (default: 自訂推薦信1)

Keyboard Controls (during automation):
  P = Pause automation
  R = Resume automation
  Q = Quit automation

Examples:
  # New browser with auto-login
  node 104_auto_apply_gemini.js --start-page 3 --max-pages 10

  # Use existing Chrome (must start Chrome with remote debugging first)
  node 104_auto_apply_gemini.js --use-existing --start-page 2 --max-pages 5

Using Existing Chrome Browser (--use-existing):
  Step 1: Close Chrome completely
  Step 2: Start Chrome with remote debugging:
    macOS:
      /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222
    Windows:
      "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222
    Linux:
      google-chrome --remote-debugging-port=9222
  Step 3: Log into 104.com.tw in that browser
  Step 4: Run: node 104_auto_apply_gemini.js --use-existing

Setup:
  1. Copy .env.example to .env
  2. Fill in your credentials in .env
  3. Run: npm install
  4. Run: node 104_auto_apply_gemini.js
`);
}

// Login to 104.com.tw
async function login(page) {
  console.log('\n🔐 Starting login process...');

  if (!CREDENTIALS.email || !CREDENTIALS.password) {
    console.log('⚠️ No credentials found in .env file');
    console.log('Please create a .env file with JOB_EMAIL and JOB_PASSWORD');
    return false;
  }

  try {
    // Navigate to login page
    await page.goto('https://www.104.com.tw/member/login', {
      waitUntil: 'networkidle',
      timeout: 30000
    });
    await page.waitForTimeout(2000);

    // Check if already logged in
    const isAlreadyLoggedIn = await page.evaluate(() => {
      return window.location.href.includes('/jobs/') ||
             document.querySelector('[class*="member-name"]') !== null ||
             document.querySelector('.member-header') !== null;
    });

    if (isAlreadyLoggedIn) {
      console.log('✅ Already logged in!');
      return true;
    }

    console.log('📧 Entering email...');

    // Find and fill email field
    const emailInput = await page.waitForSelector('input[type="text"], input[name="username"], input[placeholder*="mail"], input[placeholder*="ID"]', {
      timeout: 10000
    });

    if (emailInput) {
      await emailInput.fill(CREDENTIALS.email);
      await page.waitForTimeout(500);
    } else {
      // Try alternative method
      await page.evaluate((email) => {
        const inputs = document.querySelectorAll('input');
        for (const input of inputs) {
          if (input.type === 'text' || input.type === 'email' ||
              input.placeholder?.includes('mail') || input.placeholder?.includes('ID')) {
            input.value = email;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            break;
          }
        }
      }, CREDENTIALS.email);
    }

    // Click continue/next button
    console.log('➡️ Clicking continue...');
    await page.waitForTimeout(500);

    const continueClicked = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button'));
      const continueBtn = buttons.find(btn =>
        btn.textContent.includes('繼續') ||
        btn.textContent.includes('Continue') ||
        btn.textContent.includes('下一步')
      );
      if (continueBtn) {
        continueBtn.click();
        return true;
      }
      return false;
    });

    if (!continueClicked) {
      // Try pressing Enter
      await page.keyboard.press('Enter');
    }

    await page.waitForTimeout(2000);

    // Enter password
    console.log('🔑 Entering password...');

    const passwordInput = await page.waitForSelector('input[type="password"]', {
      timeout: 10000
    });

    if (passwordInput) {
      await passwordInput.fill(CREDENTIALS.password);
      await page.waitForTimeout(500);
    }

    // Click login button
    console.log('🚪 Clicking login...');

    const loginClicked = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button'));
      const loginBtn = buttons.find(btn =>
        btn.textContent.includes('登入') ||
        btn.textContent.includes('Login') ||
        btn.textContent.includes('Sign in')
      );
      if (loginBtn) {
        loginBtn.click();
        return true;
      }
      return false;
    });

    if (!loginClicked) {
      await page.keyboard.press('Enter');
    }

    // Wait for login to complete
    console.log('⏳ Waiting for login to complete...');
    await page.waitForTimeout(5000);

    // Check for 2FA
    const needs2FA = await page.evaluate(() => {
      const pageText = document.body.innerText;
      return pageText.includes('驗證碼') ||
             pageText.includes('verification') ||
             pageText.includes('確認碼') ||
             document.querySelector('input[maxlength="6"]') !== null;
    });

    if (needs2FA) {
      console.log('\n⚠️ 2FA REQUIRED!');
      console.log('Please enter the verification code in the browser.');
      console.log('Waiting 60 seconds for manual 2FA completion...\n');

      // Wait for 2FA completion
      let attempts = 0;
      while (attempts < 60) {
        await page.waitForTimeout(1000);
        attempts++;

        const currentUrl = page.url();
        const is2FAComplete = !currentUrl.includes('login') &&
                              !currentUrl.includes('verify');

        if (is2FAComplete) {
          console.log('✅ 2FA completed!');
          break;
        }

        if (attempts % 10 === 0) {
          console.log(`⏳ Still waiting for 2FA... (${60 - attempts}s remaining)`);
        }
      }
    }

    // Verify login success
    await page.waitForTimeout(2000);
    const loginSuccess = await page.evaluate(() => {
      return !window.location.href.includes('login') ||
             document.querySelector('[class*="member"]') !== null;
    });

    if (loginSuccess) {
      console.log('✅ Login successful!\n');
      return true;
    } else {
      console.log('❌ Login may have failed. Please check the browser.\n');
      return false;
    }

  } catch (error) {
    console.log(`❌ Login error: ${error.message}`);
    return false;
  }
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
      },
      LOGIN: {
        bg: '#9C27B0',
        html: `🔐 LOGGING IN...<br><small style="font-size: 11px;">Please wait</small>`
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

// Check if logged in
async function isLoggedIn(page) {
  return page.evaluate(() => {
    // Check for login button (means NOT logged in)
    const loginBtn = document.querySelector('a[href*="login"]');
    const memberArea = document.querySelector('[class*="member"]');
    const memberHeader = document.querySelector('.member-header');

    // If there's a member area or no login button, we're logged in
    return memberArea !== null || memberHeader !== null || loginBtn === null;
  });
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
  console.log(`Starting from page ${config.startPage}, processing ${config.maxPages} pages`);
  console.log(`Cover Letter: ${CREDENTIALS.coverLetter}`);
  console.log(`Mode: ${config.useExisting ? 'Using existing Chrome browser' : 'New browser'}\n`);

  let browser;
  let context;
  let page;
  let shouldCloseBrowser = true;

  if (config.useExisting) {
    // Connect to existing Chrome browser via CDP
    console.log(`🔌 Connecting to existing Chrome on port ${config.cdpPort}...`);
    try {
      browser = await chromium.connectOverCDP(`http://127.0.0.1:${config.cdpPort}`);
      console.log('✅ Connected to existing Chrome browser!\n');

      // Get existing contexts
      const contexts = browser.contexts();
      if (contexts.length > 0) {
        context = contexts[0];
        const pages = context.pages();

        // Use existing page or create new one
        if (pages.length > 0) {
          page = pages[0];
          console.log(`📄 Using existing tab: ${page.url()}\n`);
        } else {
          page = await context.newPage();
        }
      } else {
        context = await browser.newContext();
        page = await context.newPage();
      }

      shouldCloseBrowser = false; // Don't close user's browser
    } catch (error) {
      console.log(`\n❌ Failed to connect to existing Chrome: ${error.message}`);
      console.log('\nMake sure Chrome is running with remote debugging enabled:');
      console.log('  1. Close Chrome completely');
      console.log('  2. Run: /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222');
      console.log('  3. Log into 104.com.tw');
      console.log('  4. Run this script again with --use-existing\n');
      process.exit(1);
    }
  } else {
    // Launch new browser
    browser = await chromium.launch({
      headless: config.headless,
      args: ['--start-maximized']
    });

    context = await browser.newContext({
      viewport: null
    });

    page = await context.newPage();
  }

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
    console.log('🌐 Navigating to 104.com.tw...');
    await page.goto(firstPageUrl, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(2000);

    // Check login status and login if needed
    const loggedIn = await isLoggedIn(page);

    if (config.useExisting) {
      // Using existing browser - just verify login status
      if (loggedIn) {
        console.log('✅ Already logged in (existing browser session)!\n');
      } else {
        console.log('\n⚠️ Not logged in! Please log into 104.com.tw in this browser.');
        console.log('Waiting 30 seconds for you to log in...\n');
        await page.waitForTimeout(30000);

        // Navigate back to job search after login
        await page.goto(firstPageUrl, { waitUntil: 'networkidle', timeout: 60000 });
        await page.waitForTimeout(2000);
      }
    } else if (!loggedIn && !config.skipLogin) {
      const loginResult = await login(page);

      if (!loginResult) {
        console.log('\n⚠️ Automatic login failed.');
        console.log('Please log in manually in the browser window.');
        console.log('Waiting 30 seconds for manual login...\n');
        await page.waitForTimeout(30000);
      }

      // Navigate back to job search after login
      await page.goto(firstPageUrl, { waitUntil: 'networkidle', timeout: 60000 });
      await page.waitForTimeout(2000);
    } else if (!loggedIn && config.skipLogin) {
      console.log('\n⚠️ Not logged in and --skip-login is set.');
      console.log('Please log in manually in the browser window.');
      console.log('Waiting 30 seconds for manual login...\n');
      await page.waitForTimeout(30000);
    } else {
      console.log('✅ Already logged in!\n');
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

          // Click apply button and wait for new tab
          const [newTab] = await Promise.all([
            page.context().waitForEvent('popup'),
            page.evaluate((index) => {
              const buttons = document.querySelectorAll('.apply-button__button');
              if (buttons[index]) {
                buttons[index].click();
              }
            }, jobIndex)
          ]);

          await newTab.bringToFront();
          await newTab.waitForTimeout(2000);

          try {
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

            // Use cover letter from config
            const coverLetterName = CREDENTIALS.coverLetter;
            await newTab.evaluate((coverLetter) => {
              const options = document.querySelectorAll('.multiselect__option');
              options.forEach(opt => {
                if (opt.textContent.trim() === coverLetter) {
                  opt.click();
                }
              });
            }, coverLetterName);
            await newTab.waitForTimeout(1000);

            // Click submit
            await newTab.evaluate(() => {
              const buttons = document.querySelectorAll('button');
              buttons.forEach(btn => {
                if (btn.textContent.includes('確認送出') || btn.textContent.includes('確定')) {
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
          } finally {
            // Close tab and return
            if (!newTab.isClosed()) {
              await newTab.close();
            }
            await page.bringToFront();
            await setupKeyboardControls(page);
          }

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

  if (config.useExisting) {
    console.log('Browser connection closed. Your Chrome browser remains open.');
    // Disconnect from CDP (doesn't close the browser)
    await browser.close();
  } else {
    console.log('Browser will stay open. Close it manually when done.');
  }

  return results;
}

// Main entry point
async function main() {
  const config = parseArgs();

  if (config.help) {
    showHelp();
    process.exit(0);
  }

  // Validate .env file exists
  if (!CREDENTIALS.email || !CREDENTIALS.password) {
    console.log('\n⚠️ WARNING: Credentials not found in .env file');
    console.log('Create a .env file with:');
    console.log('  JOB_EMAIL=your_email@example.com');
    console.log('  JOB_PASSWORD=your_password');
    console.log('\nOr copy .env.example to .env and fill in your credentials.\n');
  }

  try {
    await autoApplyJobs(config);
  } catch (error) {
    console.error('Fatal error:', error.message);
    process.exit(1);
  }
}

main();
