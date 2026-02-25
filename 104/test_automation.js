/**
 * Test Automation Script for 104.com.tw
 *
 * This script applies to only the FIRST 3 jobs for testing purposes
 * Use this to verify everything works before running full automation
 *
 * Safe for testing - limited scope
 */

// Test Configuration
const TEST_CONFIG = {
  maxJobs: 3, // Only process first 3 jobs
  coverLetter: '自訂推薦信1',
  delayBetweenJobs: 3000, // 3 seconds delay
  verbose: true // Detailed logging
};

// Test Results
const testResults = {
  jobs: [],
  successful: 0,
  failed: 0,
  skipped: 0
};

/**
 * Run automation test on first 3 jobs
 */
async function runTest(page) {
  console.log('🧪 Starting TEST automation (3 jobs only)...\n');

  try {
    // Get apply buttons count
    const totalButtons = await page.evaluate(() => {
      return document.querySelectorAll('.apply-button__button').length;
    });

    console.log(`Found ${totalButtons} jobs on this page`);
    console.log(`Will test first ${Math.min(TEST_CONFIG.maxJobs, totalButtons)} jobs\n`);

    const jobsToProcess = Math.min(TEST_CONFIG.maxJobs, totalButtons);

    // Process each job
    for (let i = 0; i < jobsToProcess; i++) {
      console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
      console.log(`   TEST JOB ${i + 1}/${jobsToProcess}`);
      console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

      try {
        await testSingleJob(page, i);
        console.log('\n⏳ Waiting 3 seconds before next job...');
        await page.waitForTimeout(TEST_CONFIG.delayBetweenJobs);
      } catch (error) {
        console.error(`\n❌ Test job ${i + 1} failed:`, error.message);
        testResults.failed++;
      }
    }

    // Print test summary
    printTestSummary();

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

/**
 * Test a single job application
 */
async function testSingleJob(page, jobIndex) {
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

  if (!jobInfo) {
    throw new Error('Job not found');
  }

  console.log(`📋 Job: ${jobInfo.title}`);
  console.log(`🏢 Company: ${jobInfo.company}`);

  if (jobInfo.alreadyApplied) {
    console.log('⏭️  SKIPPED: Already applied');
    testResults.skipped++;
    testResults.jobs.push({ ...jobInfo, status: 'SKIPPED' });
    return;
  }

  // Step 1: Click apply button
  console.log('\n[1/5] Clicking apply button...');
  await page.evaluate((index) => {
    const buttons = document.querySelectorAll('.apply-button__button');
    buttons[index].click();
  }, jobIndex);

  await page.waitForTimeout(1000);

  // Step 2: Switch to new tab
  console.log('[2/5] Switching to application form tab...');
  const pages = page.context().pages();
  const newTab = pages[pages.length - 1];
  await newTab.bringToFront();
  await newTab.waitForTimeout(1000);

  try {
    const url = newTab.url();

    if (!url.includes('apply=form')) {
      throw new Error('Application form did not open');
    }

    console.log('[3/5] Opening cover letter dropdown...');

    // Step 3: Select cover letter
    await newTab.evaluate(() => {
      const elements = Array.from(document.querySelectorAll('*'));
      const systemDefault = elements.find(el =>
        el.textContent === '系統預設' && el.tagName === 'SPAN'
      );

      if (systemDefault && systemDefault.parentElement) {
        systemDefault.parentElement.click();
      } else {
        throw new Error('Dropdown not found');
      }
    });

    await newTab.waitForTimeout(500);

    console.log('[3/5] Selecting cover letter...');
    await newTab.evaluate((coverLetterName) => {
      const options = document.querySelectorAll('.multiselect__option');
      let found = false;

      options.forEach(option => {
        if (option.textContent.trim() === coverLetterName) {
          option.click();
          found = true;
        }
      });

      if (!found) {
        throw new Error(`Cover letter "${coverLetterName}" not found`);
      }
    }, TEST_CONFIG.coverLetter);

    console.log(`    ✓ Selected: ${TEST_CONFIG.coverLetter}`);

    await newTab.waitForTimeout(500);

    // Step 4: Submit
    console.log('[4/5] Submitting application...');
    await newTab.evaluate(() => {
      const buttons = document.querySelectorAll('button');
      let found = false;

      buttons.forEach(btn => {
        if (btn.textContent.includes('確認送出')) {
          btn.click();
          found = true;
        }
      });

      if (!found) {
        throw new Error('Submit button not found');
      }
    });

    await newTab.waitForTimeout(2000);

    // Step 5: Verify success
    console.log('[5/5] Verifying submission...');
    const finalUrl = newTab.url();

    if (finalUrl.includes('/job/apply/done/')) {
      console.log('\n✅ SUCCESS: Application submitted!');
      testResults.successful++;
      testResults.jobs.push({ ...jobInfo, status: 'SUCCESS' });
    } else {
      throw new Error('Success page not reached');
    }

  } catch (error) {
    console.error('\n❌ FAILED:', error.message);
    testResults.failed++;
    testResults.jobs.push({ ...jobInfo, status: 'FAILED', error: error.message });
  } finally {
    // Close new tab and return to search page
    await newTab.close();
    await page.bringToFront();
    await page.waitForTimeout(500);
  }
}

/**
 * Print test summary
 */
function printTestSummary() {
  console.log('\n');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('          TEST SUMMARY                  ');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log(`Total Tested: ${testResults.jobs.length}`);
  console.log(`✅ Successful: ${testResults.successful}`);
  console.log(`❌ Failed: ${testResults.failed}`);
  console.log(`⏭️  Skipped: ${testResults.skipped}`);
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  if (testResults.successful === testResults.jobs.length - testResults.skipped) {
    console.log('🎉 All tests passed! Ready for full automation.');
  } else if (testResults.failed > 0) {
    console.log('⚠️  Some tests failed. Review errors before full automation.');
  }

  console.log('\nDetailed Results:');
  console.table(testResults.jobs);
}

// Export for browser use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runTest };
}

if (typeof window !== 'undefined') {
  window.test104 = { runTest };
  console.log('✅ Test automation loaded! Run: test104.runTest(page)');
}
