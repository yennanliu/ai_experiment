// Single job application with SLOW mode and detailed console output
// This will show you exactly what's happening at each step

async function applySingleJobSlow(page) {
  console.log('\n╔══════════════════════════════════════════════════════════╗');
  console.log('║  104.com.tw - SLOW MODE - Watch Each Step Happen        ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');
  
  // Step 1: Go to search page
  console.log('📍 STEP 1: Navigating to search page...');
  await page.goto('https://www.104.com.tw/jobs/search/?page=6&keyword=++++%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&jobsource=joblist_search&order=15&remoteWork=1,2&area=6001001000,6001002000');
  console.log('   ⏳ Waiting 5 seconds for page to fully load...');
  await page.waitForTimeout(5000);
  console.log('   ✅ Page loaded\n');
  
  // Step 2: Collect jobs
  console.log('📍 STEP 2: Finding available jobs...');
  const jobs = await page.evaluate(() => {
    const links = [];
    const jobElements = document.querySelectorAll('a[href*="/job/"]');
    const seenUrls = new Set();

    jobElements.forEach((linkElement) => {
      const jobUrl = linkElement.href;
      if (seenUrls.has(jobUrl)) return;
      seenUrls.add(jobUrl);

      const jobTitle = linkElement.textContent.trim();
      if (jobUrl.includes('/job/')) {
        links.push({ url: jobUrl, title: jobTitle.substring(0, 70) });
      }
    });

    return links;
  });
  
  console.log(`   Found ${jobs.length} jobs on this page`);
  
  if (jobs.length < 3) {
    console.log('   ❌ Not enough jobs found. Stopping.\n');
    return { error: 'Not enough jobs' };
  }
  
  // Pick job #3 (skip first 2 since already applied)
  const targetJob = jobs[2];
  console.log(`   🎯 Target: ${targetJob.title}`);
  console.log(`   🔗 URL: ${targetJob.url}\n`);
  
  // Step 3: Open job page
  console.log('📍 STEP 3: Opening job detail page...');
  await page.goto(targetJob.url);
  console.log('   ⏳ Waiting 3 seconds...');
  await page.waitForTimeout(3000);
  console.log(`   ✅ Opened: ${page.url()}\n`);
  
  // Step 4: Find and CLICK the apply button
  console.log('📍 STEP 4: Looking for "應徵" (Apply) button...');
  await page.waitForTimeout(1000);
  
  // Take screenshot before clicking
  await page.screenshot({ path: '/Users/jerryliu/ai_experiment/104/before_click.png' });
  console.log('   📸 Screenshot saved: before_click.png');
  
  const buttonClicked = await page.evaluate(() => {
    const allElements = Array.from(document.querySelectorAll('button, a, div'));
    const applyBtn = allElements.find(el => {
      const text = el.textContent || '';
      return (text.includes('我要應徵') || text.trim() === '應徵') &&
             !text.includes('已應徵') &&
             !text.includes('人應徵') &&
             el.offsetParent !== null;
    });

    if (applyBtn) {
      applyBtn.click();
      return true;
    }
    return false;
  });
  
  if (!buttonClicked) {
    console.log('   ❌ Apply button not found\n');
    return { error: 'No apply button' };
  }
  
  console.log('   ✅ Apply button CLICKED!');
  console.log('   ⏳ Waiting 3 seconds for form to open...');
  await page.waitForTimeout(3000);
  console.log(`   Current URL: ${page.url()}\n`);
  
  // Take screenshot after clicking
  await page.screenshot({ path: '/Users/jerryliu/ai_experiment/104/after_click.png' });
  console.log('   📸 Screenshot saved: after_click.png\n');
  
  if (!page.url().includes('apply=form')) {
    console.log('   ❌ Application form did not open\n');
    return { error: 'Form not opened', url: page.url() };
  }
  
  console.log('   ✅✅✅ SUCCESS! Application form opened!\n');
  
  // Step 5: Select cover letter
  console.log('📍 STEP 5: Opening cover letter dropdown...');
  await page.waitForTimeout(1000);
  
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
    console.log('   ❌ Dropdown not found\n');
    return { error: 'Dropdown not found' };
  }
  
  console.log('   ✅ Dropdown opened');
  console.log('   ⏳ Waiting 2 seconds...');
  await page.waitForTimeout(2000);
  
  // Step 6: Select "自訂推薦信1"
  console.log('\n📍 STEP 6: Selecting "自訂推薦信1"...');
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
    console.log('   ❌ Cover letter not found\n');
    return { error: 'Cover letter not found' };
  }
  
  console.log('   ✅ Cover letter selected');
  console.log('   ⏳ Waiting 2 seconds...');
  await page.waitForTimeout(2000);
  
  // Take screenshot before submit
  await page.screenshot({ path: '/Users/jerryliu/ai_experiment/104/before_submit.png' });
  console.log('   📸 Screenshot saved: before_submit.png\n');
  
  // Step 7: Click SUBMIT
  console.log('📍 STEP 7: Clicking "確認送出" (Submit)...');
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
    console.log('   ❌ Submit button not found\n');
    return { error: 'Submit button not found' };
  }
  
  console.log('   ✅ Submit button CLICKED!');
  console.log('   ⏳ Waiting 4 seconds for confirmation...');
  await page.waitForTimeout(4000);
  
  // Take screenshot after submit
  await page.screenshot({ path: '/Users/jerryliu/ai_experiment/104/after_submit.png' });
  console.log('   📸 Screenshot saved: after_submit.png\n');
  
  // Step 8: Verify success
  console.log('📍 STEP 8: Verifying success...');
  console.log(`   Final URL: ${page.url()}`);
  
  if (page.url().includes('/job/apply/done/')) {
    console.log('\n');
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║  ✅✅✅ SUCCESS! APPLICATION SUBMITTED! ✅✅✅           ║');
    console.log('╚══════════════════════════════════════════════════════════╝\n');
    
    return {
      status: 'SUCCESS',
      job: targetJob.title,
      url: page.url()
    };
  } else {
    console.log('   ❌ Did not reach success page\n');
    return { error: 'Not confirmed', url: page.url() };
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { applySingleJobSlow };
}
