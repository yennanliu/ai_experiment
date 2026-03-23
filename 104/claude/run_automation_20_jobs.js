#!/usr/bin/env node

/**
 * Script to apply to 20 SWE jobs on 104.com.tw
 */

const { chromium } = require('playwright');
const { autoApplyWithControls } = require('./104_auto_apply_with_controls.js');

async function main() {
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    console.log('🚀 Starting 104 SWE Job Application Automation...\n');

    // Navigate to 104 login page first (user needs to login manually)
    await page.goto('https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&jobsource=joblist_search&keyword=%20%20%20%20%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15&remoteWork=1,2&page=1', { waitUntil: 'networkidle' });

    console.log('✅ Page loaded. If not logged in, please login now in the browser.');
    console.log('Press Enter in this terminal once you are logged in and ready...\n');

    // Wait for user to login (read from stdin)
    await new Promise(resolve => {
      process.stdin.once('data', () => resolve());
    });

    console.log('\n📋 Starting automation to apply to ~20 SWE jobs...\n');

    // Run automation on pages 1-2 (approximately 40-44 jobs, we'll apply to ~20)
    const results = await autoApplyWithControls(page, 1, 2);

    console.log('\n✨ Automation complete!');
    console.log('Results:', results);

  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    // Keep browser open for review
    console.log('\n💡 Browser will stay open. Close it manually when done.');
    // Don't close yet - let user review results
    // await browser.close();
  }
}

main().catch(console.error);
