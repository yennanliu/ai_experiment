/**
 * Apply to a SINGLE job on 104.com.tw
 *
 * This is a simplified version for testing and manual execution
 * Use this to apply to one job at a time through browser console
 *
 * Usage: Run each function step by step via Playwright MCP browser tools
 */

// Step 1: Click apply button for a specific job (by index, starting from 0)
async function clickApply(jobIndex = 0) {
  const buttons = document.querySelectorAll('.apply-button__button');

  if (jobIndex >= buttons.length) {
    throw new Error(`Only ${buttons.length} jobs available, requested index ${jobIndex}`);
  }

  console.log(`Clicking apply button for job ${jobIndex + 1}/${buttons.length}`);
  buttons[jobIndex].click();

  return {
    success: true,
    message: 'Apply button clicked - new tab should open'
  };
}

// Step 2: Select cover letter (run this in the NEW TAB)
async function selectCoverLetter(coverLetterName = '自訂推薦信1') {
  // Wait a bit for page to load
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Click dropdown to open
  const elements = Array.from(document.querySelectorAll('*'));
  const systemDefault = elements.find(el =>
    el.textContent === '系統預設' && el.tagName === 'SPAN'
  );

  if (!systemDefault || !systemDefault.parentElement) {
    throw new Error('Dropdown not found');
  }

  systemDefault.parentElement.click();
  console.log('Dropdown opened');

  // Wait for options to appear
  await new Promise(resolve => setTimeout(resolve, 500));

  // Select the option
  const options = document.querySelectorAll('.multiselect__option');
  let found = false;

  options.forEach(option => {
    if (option.textContent.trim() === coverLetterName) {
      option.click();
      found = true;
      console.log(`Selected: ${coverLetterName}`);
    }
  });

  if (!found) {
    throw new Error(`Cover letter "${coverLetterName}" not found`);
  }

  return {
    success: true,
    message: `Cover letter "${coverLetterName}" selected`
  };
}

// Step 3: Submit application (run this in the NEW TAB)
async function submitApplication() {
  const buttons = document.querySelectorAll('button');

  let found = false;
  buttons.forEach(btn => {
    if (btn.textContent.includes('確認送出')) {
      btn.click();
      found = true;
      console.log('Submit button clicked');
    }
  });

  if (!found) {
    throw new Error('Submit button not found');
  }

  // Wait for submission
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Check if success page
  const isSuccess = window.location.href.includes('/job/apply/done/');

  return {
    success: isSuccess,
    message: isSuccess ? 'Application submitted successfully!' : 'Submission may have failed',
    url: window.location.href
  };
}

// Helper: Get list of jobs on current page
function listJobs() {
  const containers = document.querySelectorAll('[class*="job-list-container"]');
  const jobs = [];

  containers.forEach((container, index) => {
    const titleLink = container.querySelector('a[href*="/job/"]');
    const title = titleLink ? titleLink.textContent.trim() : 'Unknown';

    const companyElement = container.querySelector('[class*="company"]');
    const company = companyElement ? companyElement.textContent.trim() : 'Unknown';

    const containerText = container.textContent;
    const alreadyApplied = containerText.includes('已應徵');

    jobs.push({
      index: index,
      title: title.substring(0, 60),
      company: company.substring(0, 40),
      alreadyApplied: alreadyApplied
    });
  });

  console.table(jobs);
  return jobs;
}

// Export functions for browser use
if (typeof window !== 'undefined') {
  window.apply104 = {
    clickApply,
    selectCoverLetter,
    submitApplication,
    listJobs
  };

  console.log('✅ 104 Automation Helper loaded!');
  console.log('Available functions:');
  console.log('  - apply104.listJobs()              // List all jobs on page');
  console.log('  - apply104.clickApply(0)           // Click apply for job 0');
  console.log('  - apply104.selectCoverLetter()     // Select cover letter (in new tab)');
  console.log('  - apply104.submitApplication()     // Submit (in new tab)');
}
