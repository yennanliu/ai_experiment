/**
 * Runner script for 104.com.tw automation - Apply to 2000 jobs
 *
 * This script:
 * - Uses a series of different keywords to find new jobs.
 * - Aims for a cumulative target of 2000 successful applications.
 * - Stops automatically when the target is reached.
 * - Supports manual controls (P=Pause, R=Resume, Q=Quit).
 */

const { chromium } = require('playwright');
const { autoApplyWithTarget } = require('./104_auto_apply_targetted.js');

const GRAND_TARGET = 2000;
// Keywords that are less likely to have been exhausted in previous runs.
const KEYWORDS = [
    'AI Engineer',
    'Machine Learning Engineer',
    'Site Reliability Engineer',
    'SRE',
    'Node.js',
    'React',
    'Vue',
    'Angular',
    'iOS',
    'Android',
    'Senior Software Engineer',
    'Principal Engineer',
    'Tech Lead',
    'Web3',
    'Blockchain',
    'Security Engineer',
    'Firmware Engineer',
    'Embedded Engineer',
    'Game Developer'
];

async function main() {
  console.log('╔═══════════════════════════════════════════════════════╗');
  console.log('║   104.com.tw Job Application Automation              ║');
  console.log(`║   Target: Apply to ${GRAND_TARGET} jobs                          ║`);
  console.log('╚═══════════════════════════════════════════════════════╝\n');

  const browser = await chromium.launch({
    headless: false,
    slowMo: 100
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
  });

  const page = await context.newPage();

  let totalSuccessful = 0;
  const grandTotalResults = { successful: 0, failed: 0, skipped: 0, pagesProcessed: 0 };

  try {
    console.log('📍 Navigating to 104.com.tw to check login status...\n');
    await page.goto('https://www.104.com.tw/jobs/search/', { waitUntil: 'domcontentloaded', timeout: 90000 });
    await page.waitForTimeout(3000);

    const needsLogin = await page.evaluate(() => {
      return document.body.textContent.includes('登入') && !document.body.textContent.includes('登出');
    });

    if (needsLogin) {
      console.log('⚠️  Login required. Please log in to your 104.com.tw account in the browser.');
      console.log('⏳ Waiting for you to log in... The script will resume in 60 seconds or when you navigate away from the login page.');
      await page.waitForNavigation({ timeout: 60000 }).catch(() => {
        console.log('Login wait timeout. Assuming login was successful and continuing...');
      });
    } else {
        console.log('✅ Already logged in.');
    }


    for (const keyword of KEYWORDS) {
      if (totalSuccessful >= GRAND_TARGET) {
        console.log(`\n🎉 Grand target of ${GRAND_TARGET} reached! Automation stopping.`);
        break;
      }

      console.log(`\n${'='.repeat(70)}`);
      console.log(`🚀 Starting automation for keyword: "${keyword}"`);
      console.log(`   Current Progress: ${totalSuccessful} / ${GRAND_TARGET}`);
      console.log(`${'='.repeat(70)}\n`);

      const remainingTarget = GRAND_TARGET - totalSuccessful;
      const searchUrl = `https://www.104.com.tw/jobs/search/?keyword=${encodeURIComponent(keyword)}&order=15&jobsource=joblist_search&remoteWork=1,2`;

      const results = await autoApplyWithTarget(page, {
        searchUrl: searchUrl,
        startPage: 1,
        maxPages: 100, // Process up to 100 pages per keyword
        target: GRAND_TARGET, // Pass the grand target
        successfulSoFar: totalSuccessful
      });

      totalSuccessful += results.successful;
      grandTotalResults.successful += results.successful;
      grandTotalResults.failed += results.failed;
      grandTotalResults.skipped += results.skipped;
      grandTotalResults.pagesProcessed += results.pages.length;
    }

  } catch (error) {
    console.error('\n❌ A critical error occurred:', error.message);
  } finally {
    console.log('\n\n╔═══════════════════════════════════════════════════════╗');
    console.log('║             FINAL AUTOMATION SUMMARY                  ║');
    console.log('╚═══════════════════════════════════════════════════════╝');
    console.log(`\n📊 Grand Total Results:`);
    console.log(`   ✅ Successful applications: ${grandTotalResults.successful}`);
    console.log(`   ❌ Failed applications: ${grandTotalResults.failed}`);
    console.log(`   ⏭️  Skipped (already applied): ${grandTotalResults.skipped}`);
    console.log(`   📄 Total pages processed: ${grandTotalResults.pagesProcessed}`);
    console.log(`\n🎯 Target of ${GRAND_TARGET} jobs was ${totalSuccessful >= GRAND_TARGET ? 'ACHIEVED!' : 'NOT REACHED.'}`);
    console.log('\n🔚 Closing browser in 10 seconds...');
    await page.waitForTimeout(10000);
    await browser.close();
  }
}

main().catch(console.error);
