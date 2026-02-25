// 104.com.tw Complete Auto-Apply Automation Script
// Processes all jobs across multiple pages

async function autoApply104Jobs(page) {
  const results = {
    success: [],
    skipped: [],
    failed: [],
    appliedJobs: []
  };

  // Function to apply to a single job
  async function applyToJob(job) {
    console.log(`\n🔍 [${job.title}]`);

    try {
      // Navigate to job detail page
      await page.goto(job.url, { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(3000);

      // Check if already applied
      const pageText = await page.evaluate(() => document.body.textContent);
      if (pageText.includes('已應徵') || pageText.includes('今日已應徵')) {
        console.log(`   ⚠️  SKIPPED: Already applied`);
        return { status: 'skipped', reason: 'Already applied', job };
      }

      // Find and click apply button
      const applyClicked = await page.evaluate(() => {
        // Look for apply button
        const allElements = Array.from(document.querySelectorAll('button, a, div'));
        const applyBtn = allElements.find(el => {
          const text = el.textContent || '';
          return (text.includes('我要應徵') || (text.trim() === '應徵')) &&
                 !text.includes('已應徵') &&
                 !text.includes('人應徵') &&
                 el.offsetParent !== null; // Element must be visible
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

      await page.waitForTimeout(3000);

      // Check if URL changed to apply form
      const currentUrl = page.url();
      if (!currentUrl.includes('apply=form')) {
        console.log(`   ⚠️  SKIPPED: Apply form not opened (URL: ${currentUrl})`);
        return { status: 'skipped', reason: 'Apply form not opened', job };
      }

      // Select cover letter "自訂推薦信1"
      const coverLetterSelected = await page.evaluate(() => {
        // Open dropdown
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

      if (!coverLetterSelected) {
        console.log(`   ⚠️  SKIPPED: Cover letter dropdown not found`);
        return { status: 'skipped', reason: 'Cover letter dropdown not found', job };
      }

      await page.waitForTimeout(1500);

      // Select the cover letter option
      const optionSelected = await page.evaluate(() => {
        const options = Array.from(document.querySelectorAll('span, div, li'));
        const option = options.find(el => el.textContent.trim() === '自訂推薦信1');
        if (option) {
          option.click();
          return true;
        }
        return false;
      });

      if (!optionSelected) {
        console.log(`   ⚠️  SKIPPED: Cover letter option not found`);
        return { status: 'skipped', reason: 'Cover letter option not found', job };
      }

      await page.waitForTimeout(1000);

      // Submit application
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

      // Check if successful
      const finalUrl = page.url();
      if (finalUrl.includes('/job/apply/done/')) {
        console.log(`   ✅ SUCCESS`);
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

  // Collect jobs from current page
  async function collectJobsFromPage(pageNum) {
    await page.goto(`https://www.104.com.tw/jobs/search/?page=${pageNum}&keyword=++++%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&jobsource=joblist_search&order=15&remoteWork=1,2&area=6001001000,6001002000`);
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

        const alreadyApplied = containerText.includes('今日已應徵') || containerText.includes('已應徵');
        const cantApply = containerText.includes('無法應徵') || containerText.includes('關閉職缺');

        if (jobUrl.includes('/job/') && !alreadyApplied && !cantApply) {
          links.push({
            url: jobUrl,
            title: jobTitle.substring(0, 80)
          });
        }
      });

      return links;
    });

    return jobData;
  }

  // Main automation loop
  console.log(`\n${'='.repeat(70)}`);
  console.log(`🚀 104.com.tw Auto-Apply Automation Started`);
  console.log(`${'='.repeat(70)}\n`);

  let currentPage = 6;
  let totalProcessed = 0;
  let shouldContinue = true;

  while (shouldContinue && currentPage <= 10) { // Limit to pages 6-10
    console.log(`\n📄 [Page ${currentPage}]`);

    const jobs = await collectJobsFromPage(currentPage);
    console.log(`   Found ${jobs.length} jobs`);

    if (jobs.length === 0) {
      console.log(`   No more jobs found. Stopping.`);
      break;
    }

    // Skip first job on page 6 (already applied)
    const jobsToProcess = currentPage === 6 ? jobs.slice(1) : jobs;

    for (let i = 0; i < jobsToProcess.length; i++) {
      const job = jobsToProcess[i];
      console.log(`\n   [${i + 1}/${jobsToProcess.length}]`);

      const result = await applyToJob(job);

      if (result.status === 'success') {
        results.success.push(result.job);
        results.appliedJobs.push(result.job.url);
      } else if (result.status === 'skipped') {
        results.skipped.push({ job: result.job, reason: result.reason });
      } else {
        results.failed.push({ job: result.job, reason: result.reason });
      }

      totalProcessed++;

      // Random delay (2-4 seconds)
      const delay = 2000 + Math.random() * 2000;
      console.log(`   ⏱️  Waiting ${(delay / 1000).toFixed(1)}s...`);
      await page.waitForTimeout(delay);
    }

    // Move to next page
    currentPage++;
  }

  // Final summary
  console.log(`\n${'='.repeat(70)}`);
  console.log(`📊 Final Summary`);
  console.log(`${'='.repeat(70)}`);
  console.log(`   Total Processed: ${totalProcessed}`);
  console.log(`   ✅ Successfully Applied: ${results.success.length}`);
  console.log(`   ⚠️  Skipped: ${results.skipped.length}`);
  console.log(`   ❌ Failed: ${results.failed.length}`);
  console.log(`${'='.repeat(70)}\n`);

  if (results.success.length > 0) {
    console.log(`\n✅ Successfully Applied (${results.success.length}):`);
    results.success.forEach((job, i) => {
      console.log(`   ${i + 1}. ${job.title}`);
    });
  }

  if (results.skipped.length > 0) {
    console.log(`\n⚠️  Skipped Jobs (${results.skipped.length}):`);
    results.skipped.slice(0, 10).forEach((item, i) => {
      console.log(`   ${i + 1}. ${item.job.title} - ${item.reason}`);
    });
    if (results.skipped.length > 10) {
      console.log(`   ... and ${results.skipped.length - 10} more`);
    }
  }

  return results;
}

// Export for use
module.exports = { autoApply104Jobs };
