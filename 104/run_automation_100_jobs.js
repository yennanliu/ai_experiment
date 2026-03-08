/**
 * Runner script for 104.com.tw automation - Apply to 100 jobs
 *
 * This script:
 * - Launches Chromium browser
 * - Navigates to 104.com.tw
 * - Runs automation to apply to ~100 jobs (5-6 pages × ~20 jobs/page)
 * - Supports manual controls (P=Pause, R=Resume, Q=Quit)
 */

const { chromium } = require('playwright');
const { autoApplyWithControls } = require('./104_auto_apply_with_controls.js');

async function main() {
  console.log('╔═══════════════════════════════════════════════════════╗');
  console.log('║   104.com.tw Job Application Automation              ║');
  console.log('║   Target: Apply to ~100 jobs                          ║');
  console.log('╚═══════════════════════════════════════════════════════╝\n');

  const browser = await chromium.launch({
    headless: false, // Show browser for 2FA/manual login if needed
    slowMo: 100 // Slightly slow down for stability
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
  });

  const page = await context.newPage();

  try {
    // Navigate to the job search page
    console.log('📍 Navigating to 104.com.tw job search...\n');
    const startUrl = 'https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&jobsource=joblist_search&keyword=%20%20%20%20%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15&remoteWork=1,2&page=2';

    await page.goto(startUrl, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(3000);

    // Check if login is needed
    const needsLogin = await page.evaluate(() => {
      return document.body.textContent.includes('登入') ||
             document.body.textContent.includes('Login');
    });

    if (needsLogin) {
      console.log('⚠️  Login may be required. Please complete login manually if prompted.');
      console.log('⏳ Waiting 30 seconds for manual login...\n');
      await page.waitForTimeout(30000);
    }

    // Run automation for 6 pages (120 jobs target, accounting for skipped jobs)
    console.log('🚀 Starting automation...\n');
    console.log('📋 Target: 6 pages (~120 jobs, expecting ~100 successful applications)');
    console.log('⌨️  Controls: P=Pause | R=Resume | Q=Quit\n');

    const startPage = 2;  // Starting from page 2
    const maxPages = 7;   // Process pages 2-7 (6 pages total)

    const results = await autoApplyWithControls(page, startPage, maxPages);

    // Display final results
    console.log('\n');
    console.log('╔═══════════════════════════════════════════════════════╗');
    console.log('║             AUTOMATION COMPLETED                      ║');
    console.log('╚═══════════════════════════════════════════════════════╝');
    console.log(`\n📊 Final Results:`);
    console.log(`   ✅ Successful applications: ${results.successful}`);
    console.log(`   ❌ Failed applications: ${results.failed}`);
    console.log(`   ⏭️  Skipped (already applied): ${results.skipped}`);
    console.log(`   📄 Pages processed: ${results.pages.length}`);
    console.log(`   📝 Total jobs encountered: ${results.successful + results.failed + results.skipped}`);
    console.log('\n');

    // Check if we hit the 100 job target
    if (results.successful >= 100) {
      console.log('🎯 Target achieved! Applied to 100+ jobs successfully!\n');
    } else {
      console.log(`📈 Applied to ${results.successful} jobs. To reach 100, continue with more pages.\n`);
    }

  } catch (error) {
    console.error('\n❌ Error occurred:', error.message);
    console.error(error.stack);
  } finally {
    console.log('🔚 Closing browser in 5 seconds...');
    await page.waitForTimeout(5000);
    await browser.close();
  }
}

// Run the automation
main().catch(console.error);
