const puppeteer = require('puppeteer');

/**
 * Utility to add a random delay between min and max seconds.
 */
const randomDelay = async (min = 1, max = 3) => {
    const delay = Math.floor(Math.random() * (max - min + 1) + min) * 1000;
    console.log(`Waiting for ${delay / 1000} seconds...`);
    return new Promise(resolve => setTimeout(resolve, delay));
};

(async () => {
    const browser = await puppeteer.launch({
        headless: false, // Set to false to see the actions or if required for bypass
        defaultViewport: null,
        args: ['--start-maximized']
    });

    const page = await browser.newPage();

    // Set a realistic User-Agent
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');

    try {
        console.log('Navigating to Tasker cases page...');
        await page.goto('https://www.tasker.com.tw/cases?selected_tags=1', {
            waitUntil: 'networkidle2'
        });

        // Wait for the case list to load
        // Selector for case items (assumed based on common patterns)
        const caseListSelector = '.case-item, .case-list-item, [class*="case-item"]';
        await page.waitForSelector(caseListSelector, { timeout: 10000 });

        console.log('Selecting the first available case...');
        // Find the first "我要提案" button
        // Common button text in Chinese: 我要提案
        const proposalButtonSelector = '::-p-text(我要提案)';
        const firstButton = await page.$(proposalButtonSelector);

        if (!firstButton) {
            throw new Error('Could not find the "我要提案" button.');
        }

        await randomDelay();
        console.log('Clicking "我要提案" button...');

        // Wait for target to be created (new tab)
        const newTargetPromise = new Promise(resolve => browser.once('targetcreated', target => resolve(target)));
        await firstButton.click();
        const newTarget = await newTargetPromise;

        console.log('Switching to the newly opened tab...');
        const newPage = await newTarget.page();
        await newPage.bringToFront();
        await newPage.waitForNetworkIdle();

        console.log('Processing proposal page...');

        // Price Injection Logic:
        // 1. Locate budget range.
        // Usually, the budget range is displayed on the page.
        // We'll search for patterns like "5,000 ~ 10,000" or similar.
        // This part might need specific selectors if they are available.
        // For now, let's assume we can extract them from elements.
        
        // Hypothetical selectors based on Tasker's typical layout
        // Minimum Price Input: often name="min_price" or similar
        // Maximum Price Input: often name="max_price" or similar
        
        // We'll try to find the budget text on the page.
        const budgetText = await newPage.evaluate(() => {
            const bodyText = document.body.innerText;
            const match = bodyText.match(/(\d{1,3}(,\d{3})*)\s*~\s*(\d{1,3}(,\d{3})*)/);
            if (match) {
                return {
                    min: match[1].replace(/,/g, ''),
                    max: match[3].replace(/,/g, '')
                };
            }
            return null;
        });

        if (budgetText) {
            console.log(`Budget found: ${budgetText.min} - ${budgetText.max}`);
            
            // Fill inputs. We use generic selectors here; refine if we know the exact ones.
            // Common inputs: input[name="amount"], input[id="price"], etc.
            // On Tasker, it might be more specific.
            const minInputSelector = 'input[name*="min"], input[placeholder*="最低"]';
            const maxInputSelector = 'input[name*="max"], input[placeholder*="最高"]';

            await newPage.waitForSelector(minInputSelector, { timeout: 5000 }).catch(() => console.log('Min input selector not found.'));
            await newPage.waitForSelector(maxInputSelector, { timeout: 5000 }).catch(() => console.log('Max input selector not found.'));

            await randomDelay();
            await newPage.type(minInputSelector, budgetText.min);
            await randomDelay();
            await newPage.type(maxInputSelector, budgetText.max);
            
            console.log('Prices injected.');
        } else {
            console.warn('Could not detect budget range automatically. Please verify selectors.');
        }

        // Submission
        console.log('Submitting proposal...');
        const submitButtonSelector = 'button::-p-text(送出提案), .btn-submit, [class*="submit"]';
        const submitButton = await newPage.$(submitButtonSelector);

        if (submitButton) {
            await randomDelay(2, 4);
            // await submitButton.click(); // Commented out for safety during development
            console.log('Submission button found. Click operation is commented out for safety.');
        } else {
            console.warn('Submit button not found.');
        }

    } catch (error) {
        console.error('An error occurred during automation:', error.message);
    } finally {
        // await browser.close();
        console.log('Browser remains open for inspection. Close it manually or uncomment browser.close().');
    }
})();
