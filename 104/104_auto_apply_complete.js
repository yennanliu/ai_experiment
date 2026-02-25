/**
 * 104.com.tw Job Application Automation Script
 *
 * This script automates job applications on 104.com.tw using Playwright
 *
 * USAGE:
 * Run this with Playwright's page context:
 * await autoApply104Jobs(page);
 *
 * REQUIREMENTS:
 * - User must be logged in to 104.com.tw before running
 * - Cover letter "自訂推薦信1" must exist in user's profile
 * - Account credentials: ***REMOVED*** / ***REMOVED***
 */

async function autoApply104Jobs(page, options = {}) {
  const {
    startPage = 6,           // Starting page number
    maxPages = 5,            // Maximum number of pages to process
    delayMin = 2000,         // Minimum delay between applications (ms)
    delayMax = 4000,         // Maximum delay between applications (ms)
    coverLetter = '自訂推薦信1'  // Cover letter to use
  } = options;

  const results = {
    success: [],
    skipped: [],
    failed: [],
    totalProcessed: 0
  };

  /**
   * Apply to a single job
   * Returns: { status: 'success'|'skipped'|'failed', reason?, job }
   */
  async function applyToJob(job) {
    console.log(`\n🔍 Processing: ${job.title}`);

    try {
      // Step 1: Navigate to job detail page
      await page.goto(job.url, { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(2000);

      // Step 2: Check if already applied
      const pageText = await page.evaluate(() => document.body.textContent);
      if (pageText.includes('已應徵') || pageText.includes('今日已應徵')) {
        console.log(`   ⚠️  SKIPPED: Already applied`);
        return { status: 'skipped', reason: 'Already applied', job };
      }

      // Step 3: Find and click the apply button
      const applyClicked = await page.evaluate(() => {
        const allElements = Array.from(document.querySelectorAll('button, a, div'));
        const applyBtn = allElements.find(el => {
          const text = el.textContent || '';
          // Match "應徵" but not "已應徵" or "人應徵"
          return (text.includes('我要應徵') || text.trim() === '應徵') &&
                 !text.includes('已應徵') &&
                 !text.includes('人應徵') &&
                 el.offsetParent !== null; // Must be visible
        });

        if (applyBtn) {
          applyBtn.click();
          return true;
        }
        return false;
      });

      if (!applyClicked) {
        console.log(`   ⚠️  SKIPPED: No apply button found`);
        return { status: 'skipped', reason: 'No apply button', job };
      }

      await page.waitForTimeout(2000);

      // Step 4: Verify application form opened
      const currentUrl = page.url();
      if (!currentUrl.includes('apply=form')) {
        console.log(`   ⚠️  SKIPPED: Apply form not opened`);
        return { status: 'skipped', reason: 'Apply form not opened', job };
      }

      // Step 5: Open cover letter dropdown
      const dropdownOpened = await page.evaluate(() => {
        const dropdowns = Array.from(document.querySelectorAll('div'));
        const dropdown = dropdowns.find(el => {
          const text = el.textContent || '';
          return text.includes('系統預設') || text.includes('自訂推薦信');
        });

        if (dropdown) {
          const clickableElement = dropdown.querySelector('.multiselect__select') ||
                                   dropdown.querySelector('[class*="select"]') ||
                                   dropdown;
          clickableElement.click();
          return true;
        }
        return false;
      });

      if (!dropdownOpened) {
        console.log(`   ⚠️  SKIPPED: Cover letter dropdown not found`);
        return { status: 'skipped', reason: 'Cover letter dropdown not found', job };
      }

      await page.waitForTimeout(1500);

      // Step 6: Select cover letter
      const coverLetterName = coverLetter;
      const optionSelected = await page.evaluate((letterName) => {
        const options = Array.from(document.querySelectorAll('span, div, li'));
        const option = options.find(el => el.textContent.trim() === letterName);
        if (option) {
          option.click();
          return true;
        }
        return false;
      }, coverLetterName);

      if (!optionSelected) {
        console.log(`   ⚠️  SKIPPED: Cover letter "${coverLetterName}" not found`);
        return { status: 'skipped', reason: `Cover letter "${coverLetterName}" not found`, job };
      }

      await page.waitForTimeout(1000);

      // Step 7: Submit application
      const submitted = await page.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const submitButton = buttons.find(el => el.textContent.includes('確認送出'));
        if (submitButton) {
          submitButton.click();
          return true;
        }
        return false;
      });

      if (!submitted) {
        console.log(`   ⚠️  SKIPPED: Submit button not found`);
        return { status: 'skipped', reason: 'Submit button not found', job };
      }

      await page.waitForTimeout(3000);

      // Step 8: Verify success
      const finalUrl = page.url();
      if (finalUrl.includes('/job/apply/done/')) {
        console.log(`   ✅ SUCCESS: Application submitted`);
        return { status: 'success', job };
      } else {
        console.log(`   ❌ FAILED: Submit confirmation not received`);
        return { status: 'failed', reason: 'Submit confirmation not received', job };
      }

    } catch (error) {
      console.log(`   ❌ FAILED: ${error.message}`);
      return { status: 'failed', reason: error.message, job };
    }
  }

  /**
   * Collect job listings from current page
   */
  async function collectJobsFromPage(pageNum) {
    const searchUrl = `https://www.104.com.tw/jobs/search/?page=${pageNum}&keyword=++++%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&jobsource=joblist_search&order=15&remoteWork=1,2&area=6001001000,6001002000`;

    await page.goto(searchUrl, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(3000);

    const jobData = await page.evaluate(() => {
      const links = [];
      const jobElements = document.querySelectorAll('a[href*="/job/"]');
      const seenUrls = new Set();

      jobElements.forEach((linkElement) => {
        const jobUrl = linkElement.href;
        if (seenUrls.has(jobUrl)) return;
        seenUrls.add(jobUrl);

        const container = linkElement.closest('article') || linkElement.parentElement?.parentElement;
        const jobTitle = linkElement.textContent.trim();
        const containerText = container ? container.textContent : '';

        // Filter out already applied or closed jobs
        const alreadyApplied = containerText.includes('今日已應徵') || containerText.includes('已應徵');
        const cantApply = containerText.includes('無法應徵') || containerText.includes('關閉職缺');

        if (jobUrl.includes('/job/') && !alreadyApplied && !cantApply) {
          // Extract company name if available
          const companyElement = container?.querySelector('[class*="company"]');
          const company = companyElement ? companyElement.textContent.trim() : '';

          links.push({
            url: jobUrl,
            title: jobTitle.substring(0, 100),
            company: company.substring(0, 50)
          });
        }
      });

      return links;
    });

    return jobData;
  }

  /**
   * Main automation loop
   */
  console.log(`\n${'='.repeat(70)}`);
  console.log(`🚀 104.com.tw Auto-Apply Automation`);
  console.log(`   Start Page: ${startPage}`);
  console.log(`   Max Pages: ${maxPages}`);
  console.log(`   Cover Letter: ${coverLetter}`);
  console.log(`${'='.repeat(70)}\n`);

  let currentPage = startPage;
  const endPage = startPage + maxPages - 1;

  while (currentPage <= endPage) {
    console.log(`\n📄 [Page ${currentPage}]`);

    // Collect jobs from current page
    const jobs = await collectJobsFromPage(currentPage);
    console.log(`   Found ${jobs.length} jobs to process`);

    if (jobs.length === 0) {
      console.log(`   No more jobs found. Stopping automation.`);
      break;
    }

    // Process each job
    for (let i = 0; i < jobs.length; i++) {
      const job = jobs[i];
      console.log(`\n   [${i + 1}/${jobs.length}]`);

      const result = await applyToJob(job);

      // Store result
      if (result.status === 'success') {
        results.success.push(result.job);
      } else if (result.status === 'skipped') {
        results.skipped.push({ job: result.job, reason: result.reason });
      } else {
        results.failed.push({ job: result.job, reason: result.reason });
      }

      results.totalProcessed++;

      // Random delay to avoid detection
      const delay = delayMin + Math.random() * (delayMax - delayMin);
      console.log(`   ⏱️  Waiting ${(delay / 1000).toFixed(1)}s before next job...`);
      await page.waitForTimeout(delay);
    }

    // Move to next page
    currentPage++;
  }

  // Print final summary
  console.log(`\n${'='.repeat(70)}`);
  console.log(`📊 Final Summary`);
  console.log(`${'='.repeat(70)}`);
  console.log(`   Total Processed: ${results.totalProcessed}`);
  console.log(`   ✅ Successfully Applied: ${results.success.length}`);
  console.log(`   ⚠️  Skipped: ${results.skipped.length}`);
  console.log(`   ❌ Failed: ${results.failed.length}`);
  console.log(`${'='.repeat(70)}\n`);

  // Print details
  if (results.success.length > 0) {
    console.log(`\n✅ Successfully Applied (${results.success.length}):`);
    results.success.forEach((job, i) => {
      console.log(`   ${i + 1}. ${job.title}`);
      if (job.company) console.log(`      @ ${job.company}`);
    });
  }

  if (results.skipped.length > 0) {
    console.log(`\n⚠️  Skipped Jobs (${results.skipped.length}):`);
    results.skipped.slice(0, 10).forEach((item, i) => {
      console.log(`   ${i + 1}. ${item.job.title}`);
      console.log(`      Reason: ${item.reason}`);
    });
    if (results.skipped.length > 10) {
      console.log(`   ... and ${results.skipped.length - 10} more`);
    }
  }

  if (results.failed.length > 0) {
    console.log(`\n❌ Failed Applications (${results.failed.length}):`);
    results.failed.forEach((item, i) => {
      console.log(`   ${i + 1}. ${item.job.title}`);
      console.log(`      Reason: ${item.reason}`);
    });
  }

  return results;
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { autoApply104Jobs };
}

// Browser console usage example:
// await autoApply104Jobs(page, { startPage: 6, maxPages: 3 });
