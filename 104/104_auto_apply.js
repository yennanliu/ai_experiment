/**
 * 104.com.tw Job Application Automation Script
 *
 * This script automates job applications on 104.com.tw
 * - Navigates through job search pages
 * - Applies to jobs using "自訂推薦信1" cover letter
 * - Logs all results to a JSON file
 *
 * Usage: Run through Playwright MCP tools
 */

// Configuration
const CONFIG = {
  // Search URL (modify as needed)
  searchUrl: 'https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&jobsource=joblist_search&keyword=%20%20%20%20%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15&page=1&remoteWork=1,2',

  // Cover letter to use
  coverLetter: '自訂推薦信1',

  // Delays (in milliseconds)
  delayBetweenJobs: { min: 2000, max: 4000 },
  delayAfterClick: 500,
  delayAfterSubmit: 2000,

  // Limits
  maxJobsPerPage: 100, // Safety limit
  maxPages: 10, // Maximum pages to process

  // Logging
  logFile: 'application_log.json'
};

// Results tracking
const results = {
  startTime: new Date().toISOString(),
  totalAttempted: 0,
  successful: 0,
  failed: 0,
  skipped: 0,
  jobs: []
};

/**
 * Main automation function
 */
async function autoApplyJobs(page) {
  console.log('🚀 Starting 104.com.tw Job Application Automation...\n');

  let currentPage = 1;
  let hasNextPage = true;

  while (hasNextPage && currentPage <= CONFIG.maxPages) {
    console.log(`\n========== PAGE ${currentPage} ==========`);

    try {
      // Navigate to search page
      await navigateToPage(page, currentPage);

      // Get all apply buttons on the page
      const applyButtonsCount = await getApplyButtonsCount(page);
      console.log(`Found ${applyButtonsCount} jobs on this page\n`);

      if (applyButtonsCount === 0) {
        console.log('No jobs found on this page. Stopping.');
        break;
      }

      // Process each job
      for (let jobIndex = 0; jobIndex < Math.min(applyButtonsCount, CONFIG.maxJobsPerPage); jobIndex++) {
        console.log(`\n--- Job ${jobIndex + 1}/${applyButtonsCount} ---`);

        try {
          await processJob(page, jobIndex);
        } catch (error) {
          console.error(`❌ Error processing job ${jobIndex + 1}:`, error.message);
          results.failed++;
          results.jobs.push({
            page: currentPage,
            index: jobIndex + 1,
            status: 'FAILED',
            error: error.message,
            timestamp: new Date().toISOString()
          });
        }

        // Random delay between jobs
        await randomDelay(CONFIG.delayBetweenJobs.min, CONFIG.delayBetweenJobs.max);
      }

      // Check for next page
      hasNextPage = await hasNextPageAvailable(page);

      if (hasNextPage) {
        currentPage++;
        console.log(`\n✅ Moving to page ${currentPage}...`);
      } else {
        console.log('\n✅ No more pages available.');
      }

    } catch (error) {
      console.error(`❌ Error on page ${currentPage}:`, error.message);
      break;
    }
  }

  // Save results
  await saveResults();

  // Print summary
  printSummary();
}

/**
 * Navigate to search page
 */
async function navigateToPage(page, pageNumber) {
  const url = CONFIG.searchUrl.replace(/page=\d+/, `page=${pageNumber}`);
  await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(1000);
}

/**
 * Get count of apply buttons on the page
 */
async function getApplyButtonsCount(page) {
  return await page.evaluate(() => {
    const buttons = document.querySelectorAll('.apply-button__button');
    return buttons.length;
  });
}

/**
 * Process a single job application
 */
async function processJob(page, jobIndex) {
  results.totalAttempted++;

  // Get job info before clicking
  const jobInfo = await getJobInfo(page, jobIndex);
  console.log(`📋 Job: ${jobInfo.title}`);
  console.log(`🏢 Company: ${jobInfo.company}`);

  // Check if already applied
  if (jobInfo.alreadyApplied) {
    console.log('⏭️  SKIPPED: Already applied');
    results.skipped++;
    results.jobs.push({
      ...jobInfo,
      status: 'SKIPPED',
      reason: 'Already applied',
      timestamp: new Date().toISOString()
    });
    return;
  }

  // Click apply button (opens new tab)
  await clickApplyButton(page, jobIndex);
  await page.waitForTimeout(CONFIG.delayAfterClick);

  // Get all tabs and switch to the new one
  const pages = page.context().pages();
  const newTab = pages[pages.length - 1];
  await newTab.bringToFront();
  await newTab.waitForTimeout(1000);

  try {
    // Check if we're on the apply form page
    const currentUrl = newTab.url();

    if (!currentUrl.includes('apply=form')) {
      throw new Error('Apply form did not open');
    }

    // Select cover letter
    await selectCoverLetter(newTab);
    await newTab.waitForTimeout(CONFIG.delayAfterClick);

    // Submit application
    await submitApplication(newTab);
    await newTab.waitForTimeout(CONFIG.delayAfterSubmit);

    // Verify success
    const finalUrl = newTab.url();

    if (finalUrl.includes('/job/apply/done/')) {
      console.log('✅ SUCCESS: Application submitted');
      results.successful++;
      results.jobs.push({
        ...jobInfo,
        status: 'SUCCESS',
        timestamp: new Date().toISOString()
      });
    } else {
      throw new Error('Success page not reached');
    }

  } catch (error) {
    console.error('❌ FAILED:', error.message);
    results.failed++;
    results.jobs.push({
      ...jobInfo,
      status: 'FAILED',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  } finally {
    // Close the new tab and return to search page
    await newTab.close();
    await page.bringToFront();
    await page.waitForTimeout(500);
  }
}

/**
 * Get job information from the listing
 */
async function getJobInfo(page, jobIndex) {
  return await page.evaluate((index) => {
    const containers = document.querySelectorAll('[class*="job-list-container"]');

    if (index >= containers.length) {
      return {
        title: 'Unknown',
        company: 'Unknown',
        alreadyApplied: false,
        index: index + 1
      };
    }

    const container = containers[index];
    const titleLink = container.querySelector('a[href*="/job/"]');
    const title = titleLink ? titleLink.textContent.trim() : 'Unknown';
    const url = titleLink ? titleLink.href : '';

    // Try to find company name
    const companyElement = container.querySelector('[class*="company"]');
    const company = companyElement ? companyElement.textContent.trim() : 'Unknown';

    // Check if already applied
    const containerText = container.textContent;
    const alreadyApplied = containerText.includes('今日已應徵') ||
                          containerText.includes('已應徵') ||
                          containerText.includes('近日已應徵');

    return {
      title: title.substring(0, 100),
      company: company.substring(0, 50),
      url: url,
      alreadyApplied: alreadyApplied,
      index: index + 1
    };
  }, jobIndex);
}

/**
 * Click the apply button at the specified index
 */
async function clickApplyButton(page, jobIndex) {
  await page.evaluate((index) => {
    const buttons = document.querySelectorAll('.apply-button__button');

    if (index < buttons.length) {
      buttons[index].click();
      return true;
    }

    throw new Error('Apply button not found');
  }, jobIndex);
}

/**
 * Select cover letter from dropdown
 */
async function selectCoverLetter(page) {
  // Click dropdown to open it
  await page.evaluate(() => {
    const elements = Array.from(document.querySelectorAll('*'));
    const systemDefault = elements.find(el =>
      el.textContent === '系統預設' && el.tagName === 'SPAN'
    );

    if (systemDefault && systemDefault.parentElement) {
      systemDefault.parentElement.click();
    } else {
      throw new Error('Cover letter dropdown not found');
    }
  });

  await page.waitForTimeout(500);

  // Select the cover letter option
  await page.evaluate((coverLetterName) => {
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
  }, CONFIG.coverLetter);

  console.log(`📝 Selected cover letter: ${CONFIG.coverLetter}`);
}

/**
 * Submit the application
 */
async function submitApplication(page) {
  await page.evaluate(() => {
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
}

/**
 * Check if next page is available
 */
async function hasNextPageAvailable(page) {
  return await page.evaluate(() => {
    // Look for next page button/link
    const nextButton = Array.from(document.querySelectorAll('button, a')).find(el => {
      const text = el.textContent.trim();
      return text.includes('下一頁') || text.includes('next') ||
             (el.getAttribute('aria-label') && el.getAttribute('aria-label').includes('next'));
    });

    // Check if button is disabled
    if (nextButton) {
      return !nextButton.disabled &&
             !nextButton.classList.contains('disabled') &&
             !nextButton.hasAttribute('disabled');
    }

    return false;
  });
}

/**
 * Random delay
 */
async function randomDelay(min, max) {
  const delay = Math.floor(Math.random() * (max - min + 1)) + min;
  console.log(`⏳ Waiting ${(delay / 1000).toFixed(1)}s...`);
  await new Promise(resolve => setTimeout(resolve, delay));
}

/**
 * Save results to JSON file
 */
async function saveResults() {
  const fs = require('fs');

  results.endTime = new Date().toISOString();
  results.duration = Math.floor((new Date(results.endTime) - new Date(results.startTime)) / 1000);

  try {
    fs.writeFileSync(CONFIG.logFile, JSON.stringify(results, null, 2));
    console.log(`\n💾 Results saved to ${CONFIG.logFile}`);
  } catch (error) {
    console.error('❌ Failed to save results:', error.message);
  }
}

/**
 * Print summary
 */
function printSummary() {
  console.log('\n');
  console.log('========================================');
  console.log('           AUTOMATION SUMMARY          ');
  console.log('========================================');
  console.log(`⏱️  Duration: ${results.duration}s`);
  console.log(`📊 Total Attempted: ${results.totalAttempted}`);
  console.log(`✅ Successful: ${results.successful}`);
  console.log(`❌ Failed: ${results.failed}`);
  console.log(`⏭️  Skipped: ${results.skipped}`);
  console.log('========================================\n');

  if (results.successful > 0) {
    console.log('🎉 Congratulations! Applications submitted successfully!');
  }
}

// Export for use with Playwright MCP
module.exports = { autoApplyJobs, CONFIG };
