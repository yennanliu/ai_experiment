const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

/**
 * Utility to add a random delay between min and max seconds.
 */
const randomDelay = async (min = 1, max = 3) => {
    const delay = Math.floor(Math.random() * (max - min + 1) + min) * 1000;
    console.log(`⏳ Waiting ${delay / 1000}s...`);
    return new Promise(resolve => setTimeout(resolve, delay));
};

/**
 * Log results to a file
 */
const logResult = (message) => {
    const timestamp = new Date().toISOString();
    const logPath = path.join(__dirname, 'automation_log.txt');
    const logEntry = `[${timestamp}] ${message}\n`;
    console.log(message);
    fs.appendFileSync(logPath, logEntry);
};

/**
 * Main automation function
 */
(async () => {
    const browser = await puppeteer.launch({
        headless: false,
        defaultViewport: null,
        args: ['--start-maximized']
    });

    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');

    try {
        logResult('🚀 Starting Tasker automation...');

        // Step 1: Navigate to cases page
        logResult('📍 Navigating to Tasker cases page...');
        await page.goto('https://www.tasker.com.tw/cases?selected_tags=1', {
            waitUntil: 'networkidle2'
        });

        // Step 2: Check if login is required
        const currentUrl = page.url();
        if (currentUrl.includes('/auth/login')) {
            const account = process.env.TASKER_ACCOUNT;
            const password = process.env.TASKER_PASSWORD;

            if (account && password) {
                logResult('🔐 Using credentials from environment variables...');

                // Fill in phone number
                await page.waitForSelector('input[type="text"], input[type="tel"]', { timeout: 5000 });
                const phoneInputs = await page.$$('input[type="text"], input[type="tel"]');

                if (phoneInputs.length > 0) {
                    await phoneInputs[0].focus();
                    await page.keyboard.type(account, { delay: 50 });
                    logResult('📱 Phone number entered');

                    // Look for continue/next button
                    await randomDelay(1, 2);
                    const buttons = await page.$$('button');
                    let nextBtn = null;

                    for (const btn of buttons) {
                        const text = await page.evaluate(el => el.textContent, btn);
                        if (text.includes('繼續') || text.includes('下一步') || text.includes('Next')) {
                            nextBtn = btn;
                            break;
                        }
                    }

                    if (nextBtn) {
                        await nextBtn.click();
                        logResult('✅ Clicked continue button');
                        await randomDelay(1, 2);

                        // Enter password
                        const pwdInputs = await page.$$('input[type="password"]');
                        if (pwdInputs.length > 0) {
                            await pwdInputs[0].focus();
                            await page.keyboard.type(password, { delay: 50 });
                            logResult('🔑 Password entered');

                            await randomDelay(1, 2);

                            // Click login button
                            const loginButtons = await page.$$('button');
                            for (const btn of loginButtons) {
                                const text = await page.evaluate(el => el.textContent, btn);
                                if (text.includes('登入') || text.includes('Login')) {
                                    await btn.click();
                                    logResult('✅ Clicked login button');
                                    break;
                                }
                            }

                            // Wait for navigation after login
                            try {
                                await page.waitForNavigation({ timeout: 30000, waitUntil: 'networkidle2' });
                                logResult('✅ Login successful, navigating...');
                            } catch (e) {
                                logResult('⚠️  Navigation timeout, checking current page...');
                                await randomDelay(2, 3);
                            }
                        }
                    }
                }
            } else {
                logResult('⚠️  Login required. Set TASKER_ACCOUNT and TASKER_PASSWORD env vars to auto-login.');
                logResult('   Example: TASKER_ACCOUNT=0963335868 TASKER_PASSWORD=*** node tasker_apply.js');
                console.log('\n⏸️  Please log in manually and then the script will continue...\n');

                // Wait for login to complete
                try {
                    await page.waitForNavigation({ timeout: 120000, waitUntil: 'networkidle2' });
                    logResult('✅ Manual login detected. Continuing...');
                } catch (e) {
                    logResult('⚠️  Login timeout');
                }
            }
        }

        // Step 3: Wait for case list to load
        logResult('⏳ Waiting for case list to load...');
        await page.waitForSelector('button', { timeout: 10000 });
        await randomDelay(1, 2);

        // Step 4: Collect all "我要提案" buttons
        const caseData = await page.evaluate(() => {
            const buttons = Array.from(document.querySelectorAll('button')).filter(btn =>
                btn.textContent.includes('我要提案')
            );

            return buttons.map((btn, index) => {
                const parent = btn.closest('div[class*="wrapper-list-case"], div[class*="case"]');
                const link = parent?.querySelector('a');
                const title = parent?.querySelector('h3, h4, [class*="title"]')?.textContent || 'Unknown';

                return {
                    index,
                    title: title.trim().substring(0, 80),
                    caseUrl: link?.href || null,
                    buttonElement: btn.outerHTML.substring(0, 100)
                };
            });
        });

        logResult(`📋 Found ${caseData.length} cases with "我要提案" button`);
        caseData.forEach((c, i) => {
            logResult(`   ${i + 1}. ${c.title}`);
        });

        // Step 5: Process first case
        if (caseData.length > 0) {
            logResult(`\n🎯 Processing first case: "${caseData[0].title}"`);
            await randomDelay(2, 3);

            // Click the first "我要提案" button
            const clickResult = await page.evaluate(() => {
                const buttons = Array.from(document.querySelectorAll('button')).filter(btn =>
                    btn.textContent.includes('我要提案')
                );

                if (buttons.length > 0) {
                    buttons[0].click();
                    return { success: true };
                }
                return { success: false, error: 'Button not found' };
            });

            if (clickResult.success) {
                logResult('✅ Clicked "我要提案" button');

                // Wait for navigation or form to appear
                await randomDelay(1, 2);

                // Check if we navigated to a proposal form
                const newUrl = page.url();
                logResult(`📄 Current URL: ${newUrl}`);

                // If we're on a proposal form page, process it
                if (newUrl.includes('/cases/') && !newUrl.includes('selected_tags')) {
                    logResult('📝 Proposal form detected');

                    // Check for budget field
                    const budgetInfo = await page.evaluate(() => {
                        // Look for budget inputs
                        const budgetInputs = Array.from(document.querySelectorAll('input')).filter(inp =>
                            inp.placeholder?.includes('預算') ||
                            inp.name?.includes('budget') ||
                            inp.id?.includes('budget')
                        );

                        return {
                            hasInputs: budgetInputs.length > 0,
                            inputs: budgetInputs.map(inp => ({
                                name: inp.name,
                                placeholder: inp.placeholder,
                                id: inp.id
                            }))
                        };
                    });

                    logResult(`💰 Budget inputs found: ${budgetInfo.hasInputs}`);

                    if (budgetInfo.inputs.length > 0) {
                        logResult('   Inputs detected:');
                        budgetInfo.inputs.forEach((inp, i) => {
                            logResult(`   ${i + 1}. name="${inp.name}", placeholder="${inp.placeholder}"`);
                        });
                    }

                    // Try to find submit button
                    const submitInfo = await page.evaluate(() => {
                        const buttons = Array.from(document.querySelectorAll('button'));
                        const submitBtn = buttons.find(btn =>
                            btn.textContent.includes('送出') ||
                            btn.textContent.includes('確認') ||
                            btn.textContent.includes('提案')
                        );

                        return {
                            found: !!submitBtn,
                            text: submitBtn?.textContent || null
                        };
                    });

                    logResult(`📤 Submit button: ${submitInfo.found ? submitInfo.text : 'Not found'}`);

                } else if (newUrl.includes('/auth/login')) {
                    logResult('⚠️  Redirected to login. User session may have expired.');
                }
            } else {
                logResult('❌ Failed to click button: ' + clickResult.error);
            }
        } else {
            logResult('⚠️  No cases found on the page');
        }

        logResult('\n✨ Automation completed!');

    } catch (error) {
        logResult(`❌ ERROR: ${error.message}`);
        logResult(`   Stack: ${error.stack}`);
    } finally {
        logResult('🔍 Browser remains open for inspection. Check the page and close manually.');
        // Uncomment to auto-close: await browser.close();
    }
})();
