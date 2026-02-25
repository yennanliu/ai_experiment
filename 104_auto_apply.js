// 104.com.tw Auto-Apply Job Application Script
// This script automates job applications using Playwright

const results = {
  success: [],
  skipped: [],
  failed: []
};

// Function to apply to a single job
async function applyToJob(page, job) {
  console.log(`\n🔍 Processing: ${job.title}`);
  console.log(`   URL: ${job.url}`);

  try {
    // Navigate to job detail page
    await page.goto(job.url, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);

    // Check if already applied by looking at page content
    const pageText = await page.evaluate(() => document.body.textContent);
    if (pageText.includes('已應徵') || pageText.includes('今日已應徵')) {
      console.log(`   ⚠️  SKIPPED: Already applied`);
      return { status: 'skipped', reason: 'Already applied', job };
    }

    // Check for "我要應徵" button
    const applyButton = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button, a'));
      return buttons.some(btn => btn.textContent.includes('我要應徵') || btn.textContent.includes('應徵'));
    });

    if (!applyButton) {
      console.log(`   ⚠️  SKIPPED: No apply button found`);
      return { status: 'skipped', reason: 'No apply button', job };
    }

    // Click apply button
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button, a'));
      const applyBtn = buttons.find(btn => btn.textContent.includes('我要應徵') || btn.textContent.includes('應徵'));
      if (applyBtn) applyBtn.click();
    });

    await page.waitForTimeout(2000);

    // Check if URL changed to apply form
    const currentUrl = page.url();
    if (!currentUrl.includes('apply=form')) {
      console.log(`   ⚠️  SKIPPED: Apply form not opened`);
      return { status: 'skipped', reason: 'Apply form not opened', job };
    }

    // Select cover letter "自訂推薦信1"
    await page.evaluate(() => {
      // Open dropdown
      const dropdown = Array.from(document.querySelectorAll('div')).find(el =>
        el.textContent.includes('系統預設') || el.textContent.includes('自訂推薦信')
      );

      if (dropdown) {
        const clickableElement = dropdown.querySelector('.multiselect__select') ||
                                 dropdown.querySelector('[class*="select"]') ||
                                 dropdown;
        clickableElement.click();
      }
    });

    await page.waitForTimeout(1500);

    // Select the cover letter option
    await page.evaluate(() => {
      const option = Array.from(document.querySelectorAll('span, div, li')).find(el =>
        el.textContent.trim() === '自訂推薦信1'
      );
      if (option) {
        option.click();
      }
    });

    await page.waitForTimeout(1000);

    // Submit application
    await page.evaluate(() => {
      const submitButton = Array.from(document.querySelectorAll('button')).find(el =>
        el.textContent.includes('確認送出')
      );
      if (submitButton) {
        submitButton.click();
      }
    });

    await page.waitForTimeout(3000);

    // Check if successful by looking at URL
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

// Main automation function
async function autoApplyJobs(page, startPage = 6) {
  let currentPage = startPage;
  let hasNextPage = true;
  let totalProcessed = 0;

  console.log(`\n${'='.repeat(60)}`);
  console.log(`🚀 Starting 104.com.tw Auto-Apply Automation`);
  console.log(`   Starting from page ${currentPage}`);
  console.log(`${'='.repeat(60)}\n`);

  while (hasNextPage) {
    console.log(`\n📄 [Page ${currentPage}]`);

    // Collect jobs from current page
    const jobData = await page.evaluate(() => {
      const links = [];
      const jobElements = document.querySelectorAll('a[href*="/job/"]');
      const seenUrls = new Set();

      jobElements.forEach((linkElement) => {
        const jobUrl = linkElement.href;
        if (seenUrls.has(jobUrl)) return;
        seenUrls.add(jobUrl);

        const container = linkElement.closest('article') || linkElement.parentElement.parentElement;
        const jobTitle = linkElement.textContent.trim();
        const containerText = container.textContent || '';

        const alreadyApplied = containerText.includes('今日已應徵') || containerText.includes('已應徵');
        const cantApply = containerText.includes('無法應徵') || containerText.includes('關閉職缺');

        if (jobUrl.includes('/job/') && !alreadyApplied && !cantApply) {
          links.push({
            url: jobUrl,
            title: jobTitle.substring(0, 100),
            status: 'pending'
          });
        }
      });

      return links;
    });

    console.log(`   Found ${jobData.length} jobs to process`);

    // Apply to each job
    for (let i = 0; i < jobData.length; i++) {
      const job = jobData[i];
      console.log(`\n   [${i + 1}/${jobData.length}]`);

      const result = await applyToJob(page, job);

      // Store result
      if (result.status === 'success') {
        results.success.push(result.job);
      } else if (result.status === 'skipped') {
        results.skipped.push({ job: result.job, reason: result.reason });
      } else {
        results.failed.push({ job: result.job, reason: result.reason });
      }

      totalProcessed++;

      // Random delay between applications (2-4 seconds)
      const delay = 2000 + Math.random() * 2000;
      console.log(`   ⏱️  Waiting ${(delay / 1000).toFixed(1)}s before next job...`);
      await page.waitForTimeout(delay);

      // Navigate back to search results
      await page.goto(`https://www.104.com.tw/jobs/search/?page=${currentPage}&keyword=++++%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&jobsource=joblist_search&order=15&remoteWork=1,2&area=6001001000,6001002000`);
      await page.waitForTimeout(2000);
    }

    // Check for next page
    hasNextPage = await page.evaluate(() => {
      const nextButton = Array.from(document.querySelectorAll('button')).find(btn =>
        btn.textContent.includes('下一頁') || btn.getAttribute('aria-label')?.includes('next')
      );
      return nextButton && !nextButton.disabled;
    });

    if (hasNextPage) {
      console.log(`\n   ➡️  Moving to page ${currentPage + 1}...`);
      currentPage++;

      // Navigate to next page
      await page.goto(`https://www.104.com.tw/jobs/search/?page=${currentPage}&keyword=++++%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&jobsource=joblist_search&order=15&remoteWork=1,2&area=6001001000,6001002000`);
      await page.waitForTimeout(3000);
    } else {
      console.log(`\n   🏁 No more pages to process`);
    }
  }

  // Print summary
  console.log(`\n${'='.repeat(60)}`);
  console.log(`📊 Automation Summary`);
  console.log(`${'='.repeat(60)}`);
  console.log(`   Total Processed: ${totalProcessed}`);
  console.log(`   ✅ Successful: ${results.success.length}`);
  console.log(`   ⚠️  Skipped: ${results.skipped.length}`);
  console.log(`   ❌ Failed: ${results.failed.length}`);
  console.log(`${'='.repeat(60)}\n`);

  // Print details
  if (results.success.length > 0) {
    console.log(`\n✅ Successfully Applied (${results.success.length}):`);
    results.success.forEach((job, i) => {
      console.log(`   ${i + 1}. ${job.title}`);
    });
  }

  if (results.skipped.length > 0) {
    console.log(`\n⚠️  Skipped Jobs (${results.skipped.length}):`);
    results.skipped.forEach((item, i) => {
      console.log(`   ${i + 1}. ${item.job.title}`);
      console.log(`      Reason: ${item.reason}`);
    });
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

// Export for use with Playwright
module.exports = { autoApplyJobs, applyToJob };
