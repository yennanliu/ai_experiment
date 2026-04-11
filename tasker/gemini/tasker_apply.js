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
 * Perform login on the login page
 */
const performLogin = async (page, account, password, randomDelay, logResult) => {
    try {
        // Wait for phone input
        await page.waitForSelector('input[type="text"], input[type="tel"]', { timeout: 5000 });
        const phoneInputs = await page.$$('input[type="text"], input[type="tel"]');

        if (phoneInputs.length > 0) {
            await phoneInputs[0].focus();
            await page.keyboard.type(account, { delay: 50 });
            logResult('📱 Phone number entered');

            await randomDelay(1, 2);

            // Look for continue button
            const buttons = await page.$$('button');
            for (const btn of buttons) {
                const text = await page.evaluate(el => el.textContent, btn);
                if (text.includes('繼續') || text.includes('下一步') || text.trim() === '繼續') {
                    await btn.click();
                    logResult('✅ Clicked continue button');
                    await randomDelay(1, 2);
                    break;
                }
            }

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
                    if (text.includes('登入') || text.trim() === '登入') {
                        await btn.click();
                        logResult('✅ Clicked login button');
                        break;
                    }
                }

                // Wait for navigation after login
                try {
                    await page.waitForNavigation({ timeout: 15000, waitUntil: 'networkidle2' });
                    logResult('✅ Login successful!');
                } catch (e) {
                    logResult('⚠️  Navigation timeout after login, checking page...');
                    await randomDelay(2, 3);
                }
            }
        }
    } catch (error) {
        logResult(`❌ Login failed: ${error.message}`);
    }
};

/**
 * Inspect proposal form for fields and buttons
 */
const inspectProposalForm = async (page, logResult) => {
    try {
        // Check for budget field
        const budgetInfo = await page.evaluate(() => {
            const inputs = Array.from(document.querySelectorAll('input'));
            const selects = Array.from(document.querySelectorAll('select'));
            const textareas = Array.from(document.querySelectorAll('textarea'));

            const budgetInputs = inputs.filter(inp =>
                inp.placeholder?.includes('預算') ||
                inp.name?.includes('budget') ||
                inp.id?.includes('budget') ||
                inp.placeholder?.includes('金額')
            );

            return {
                hasInputs: budgetInputs.length > 0,
                inputs: budgetInputs.map(inp => ({
                    name: inp.name,
                    placeholder: inp.placeholder,
                    id: inp.id,
                    type: inp.type,
                    value: inp.value
                })),
                allInputs: inputs.map(inp => ({
                    name: inp.name,
                    placeholder: inp.placeholder,
                    id: inp.id,
                    type: inp.type
                })).slice(0, 10)
            };
        });

        logResult(`💰 Budget inputs found: ${budgetInfo.hasInputs}`);

        if (budgetInfo.inputs.length > 0) {
            logResult('   Budget fields detected:');
            budgetInfo.inputs.forEach((inp, i) => {
                logResult(`   ${i + 1}. name="${inp.name}", placeholder="${inp.placeholder}", type="${inp.type}"`);
            });
        } else if (budgetInfo.allInputs.length > 0) {
            logResult('   Other input fields on form:');
            budgetInfo.allInputs.forEach((inp, i) => {
                logResult(`   ${i + 1}. name="${inp.name}", placeholder="${inp.placeholder}", type="${inp.type}"`);
            });
        }

        // Try to find proposal button
        const proposalBtnInfo = await page.evaluate(() => {
            const buttons = Array.from(document.querySelectorAll('button'));
            const proposalBtn = buttons.find(btn =>
                btn.textContent.trim() === '我要提案' ||
                btn.textContent.includes('我要提案')
            );

            return {
                found: !!proposalBtn,
                text: proposalBtn?.textContent?.trim() || null,
                allButtons: buttons.map(btn => btn.textContent.trim()).filter(t => t.length > 0)
            };
        });

        logResult(`📤 "我要提案" button: ${proposalBtnInfo.found ? 'Found' : 'Not found'}`);
        if (proposalBtnInfo.found) {
            logResult('⚠️  This page shows the case details. Click "我要提案" to open the actual proposal form.');
            logResult('   Available buttons on this page:');
            proposalBtnInfo.allButtons.slice(0, 10).forEach((text, i) => {
                logResult(`   ${i + 1}. ${text}`);
            });
        }
    } catch (error) {
        logResult(`❌ Form inspection failed: ${error.message}`);
    }
};

/**
 * Click proposal button and fill/submit the form
 */
const clickProposalAndFillForm = async (page, logResult, randomDelay) => {
    try {
        logResult('🔘 Attempting to click "我要提案" button...');

        // Click the proposal button
        await page.evaluate(() => {
            const buttons = Array.from(document.querySelectorAll('button'));
            const btn = buttons.find(b => b.textContent.trim() === '我要提案');
            if (btn) btn.click();
        });

        await randomDelay(2, 3);

        // Check for modal or dialog
        const hasModal = await page.evaluate(() => {
            const modals = document.querySelectorAll('[role="dialog"], .modal, [class*="modal"], [class*="dialog"]');
            return modals.length > 0;
        });

        logResult(`📋 Modal/Dialog detected: ${hasModal ? 'Yes' : 'No'}`);

        // Inspect all form elements on the page
        const formInfo = await page.evaluate(() => {
            const inputs = Array.from(document.querySelectorAll('input'));
            const textareas = Array.from(document.querySelectorAll('textarea'));
            const selects = Array.from(document.querySelectorAll('select'));
            const buttons = Array.from(document.querySelectorAll('button'));

            // Find price/budget related inputs
            const priceInputs = inputs.filter(inp =>
                inp.placeholder?.includes('金額') ||
                inp.placeholder?.includes('預算') ||
                inp.name?.includes('price') ||
                inp.name?.includes('budget') ||
                inp.id?.includes('price') ||
                inp.id?.includes('budget') ||
                inp.getAttribute('aria-label')?.includes('金額') ||
                inp.getAttribute('aria-label')?.includes('預算')
            );

            // Look for labels associated with inputs
            const inputsWithLabels = inputs.map(inp => {
                let label = '';
                if (inp.labels && inp.labels.length > 0) {
                    label = inp.labels[0].textContent;
                } else if (inp.name) {
                    const associatedLabel = document.querySelector(`label[for="${inp.name}"]`);
                    if (associatedLabel) {
                        label = associatedLabel.textContent;
                    }
                }
                return {
                    name: inp.name,
                    id: inp.id,
                    type: inp.type,
                    placeholder: inp.placeholder,
                    value: inp.value,
                    label: label,
                    visible: inp.offsetParent !== null,
                    ariaLabel: inp.getAttribute('aria-label'),
                    dataTestid: inp.getAttribute('data-testid')
                };
            });

            return {
                inputs: inputsWithLabels,
                priceInputs: priceInputs.map(inp => ({
                    name: inp.name,
                    placeholder: inp.placeholder,
                    id: inp.id,
                    type: inp.type
                })),
                textareas: textareas.map(ta => ({
                    name: ta.name,
                    placeholder: ta.placeholder,
                    id: ta.id,
                    visible: ta.offsetParent !== null
                })),
                selects: selects.map(sel => ({
                    name: sel.name,
                    id: sel.id,
                    options: Array.from(sel.options).map(opt => opt.text)
                })),
                buttons: buttons.map(btn => ({
                    text: btn.textContent.trim(),
                    visible: btn.offsetParent !== null
                }))
            };
        });

        logResult(`📋 Form structure:`);

        if (formInfo.inputs.length > 0) {
            const visibleCount = formInfo.inputs.filter(i => i.visible).length;
            logResult(`   Input fields (${formInfo.inputs.length} total, ${visibleCount} visible):`);

            // Show visible inputs
            const visibleInputs = formInfo.inputs.filter(i => i.visible);
            if (visibleInputs.length > 0) {
                visibleInputs.slice(0, 10).forEach((inp, i) => {
                    const label = inp.label || inp.placeholder || inp.ariaLabel || inp.name || `Field ${i + 1}`;
                    logResult(`   ${i + 1}. [${inp.type}] ${label}`);
                });
            }

            // Show first few non-visible inputs for debugging
            const hiddenInputs = formInfo.inputs.filter(i => !i.visible);
            if (hiddenInputs.length > 0 && visibleCount === 0) {
                logResult(`   ⚠️  (${hiddenInputs.length} hidden inputs - may be used by JavaScript):`);
                hiddenInputs.slice(0, 8).forEach((inp, i) => {
                    const label = inp.label || inp.placeholder || inp.name || `Hidden ${i + 1}`;
                    logResult(`   H${i + 1}. [${inp.type}] ${label} (name: "${inp.name}")`);
                });
            }
        }

        if (formInfo.priceInputs.length > 0) {
            logResult(`💰 Price/Budget inputs found:`);
            formInfo.priceInputs.forEach((inp, i) => {
                logResult(`   ${i + 1}. name="${inp.name}" placeholder="${inp.placeholder}"`);
            });
        }

        if (formInfo.textareas.length > 0) {
            logResult(`   Textareas (${formInfo.textareas.filter(t => t.visible).length} visible):`);
            formInfo.textareas.filter(t => t.visible).forEach((ta, i) => {
                logResult(`   ${i + 1}. name="${ta.name}" placeholder="${ta.placeholder}"`);
            });
        }

        if (formInfo.selects.length > 0) {
            logResult(`   Dropdowns (${formInfo.selects.length}):`);
            formInfo.selects.forEach((sel, i) => {
                logResult(`   ${i + 1}. ${sel.name || 'Select'}`);
            });
        }

        // Find submit button
        const submitBtn = formInfo.buttons.find(b =>
            b.text.includes('送出') ||
            b.text.includes('提案') ||
            b.text.includes('確認')
        );

        if (submitBtn) {
            logResult(`📤 Submit button found: "${submitBtn.text}"`);

            // Try to fill price inputs with a default quote
            if (formInfo.priceInputs.length > 0) {
                logResult('💵 Attempting to fill price fields with default quote...');

                const filled = await page.evaluate((priceInputCount) => {
                    const inputs = Array.from(document.querySelectorAll('input'));
                    const priceInputs = inputs.filter(inp =>
                        inp.placeholder?.includes('金額') ||
                        inp.placeholder?.includes('預算') ||
                        inp.name?.includes('price') ||
                        inp.name?.includes('budget')
                    );

                    // Fill first price input with a quote (e.g., 5000)
                    if (priceInputs.length > 0) {
                        priceInputs[0].value = '5000';
                        priceInputs[0].dispatchEvent(new Event('input', { bubbles: true }));
                        priceInputs[0].dispatchEvent(new Event('change', { bubbles: true }));
                        return { filled: true, count: 1 };
                    }
                    return { filled: false, count: 0 };
                }, formInfo.priceInputs.length);

                if (filled.filled) {
                    logResult(`✅ Filled ${filled.count} price field(s) with 5000`);

                    // Wait a moment for any handlers
                    await randomDelay(1, 2);

                    // Submit the form
                    logResult(`📤 Submitting proposal...`);
                    const submitResult = await page.evaluate(() => {
                        const buttons = Array.from(document.querySelectorAll('button'));
                        const btn = buttons.find(b =>
                            b.textContent.includes('送出') ||
                            b.textContent.includes('提案') ||
                            b.textContent.includes('確認')
                        );
                        if (btn) {
                            btn.click();
                            return { success: true, text: btn.textContent.trim() };
                        }
                        return { success: false };
                    });

                    if (submitResult.success) {
                        logResult(`✅ Clicked submit button: "${submitResult.text}"`);
                        await randomDelay(2, 3);

                        // Check for success message
                        const successMsg = await page.evaluate(() => {
                            const text = document.body.innerText;
                            if (text.includes('成功') || text.includes('提案完成') || text.includes('應徵成功')) {
                                return true;
                            }
                            return false;
                        });

                        if (successMsg) {
                            logResult(`🎉 Proposal submitted successfully!`);
                        } else {
                            logResult(`📄 Form submitted. Final URL: ${page.url()}`);
                        }
                    } else {
                        logResult(`❌ Could not find submit button`);
                    }
                } else {
                    logResult(`⚠️  Could not fill price fields`);
                }
            } else {
                logResult(`⚠️  No price inputs found. Manual inspection needed.`);
            }
        } else {
            logResult(`⚠️  No submit button found`);
        }
    } catch (error) {
        logResult(`❌ Failed to process proposal form: ${error.message}`);
    }
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
                    const btn = buttons[0];
                    console.log('Button found:', btn.className);
                    console.log('Button HTML:', btn.outerHTML.substring(0, 200));
                    buttons[0].click();
                    return { success: true, buttonClass: btn.className };
                }
                return { success: false, error: 'Button not found' };
            });

            if (clickResult.success) {
                logResult(`✅ Clicked "我要提案" button (class: ${clickResult.buttonClass})`);

                // Wait a bit to see if page is changing
                await randomDelay(2, 3);

                // Check the resulting URL
                const newUrl = page.url();
                logResult(`📄 Current URL after click: ${newUrl}`);

                // Try to wait for any navigation that might happen
                try {
                    await page.waitForNavigation({ timeout: 5000, waitUntil: 'networkidle2' });
                    logResult(`📄 Navigation detected, new URL: ${page.url()}`);
                } catch (e) {
                    logResult('⚠️  No navigation detected');
                }

                // Check the final URL
                const finalUrl = page.url();
                logResult(`📄 Final URL: ${finalUrl}`);

                // Handle login redirect
                if (finalUrl.includes('/auth/login')) {
                    logResult('🔐 Login required to access proposal form');
                    const account = process.env.TASKER_ACCOUNT;
                    const password = process.env.TASKER_PASSWORD;

                    if (account && password) {
                        logResult('🔓 Attempting auto-login...');
                        await performLogin(page, account, password, randomDelay, logResult);

                        // After login, navigate to case page directly
                        logResult('🔄 Navigating to case page after login...');
                        if (caseData[0].caseUrl) {
                            await page.goto(caseData[0].caseUrl, { waitUntil: 'networkidle2' });
                            logResult(`📄 Navigated to: ${page.url()}`);

                            await randomDelay(2, 3);

                            // Check for proposal form on case page
                            const casePageUrl = page.url();
                            if (casePageUrl.includes('/cases/') && !casePageUrl.includes('selected_tags')) {
                                logResult('📝 Case detail page loaded');
                                await inspectProposalForm(page, logResult);

                                // Click proposal button and fill/submit form
                                await randomDelay(1, 2);
                                await clickProposalAndFillForm(page, logResult, randomDelay);
                            } else {
                                logResult('⚠️  Case page not loaded as expected: ' + casePageUrl);
                            }
                        } else {
                            logResult('⚠️  No case URL available, navigating to search page');
                            await page.goto('https://www.tasker.com.tw/cases?selected_tags=1', { waitUntil: 'networkidle2' });
                            await randomDelay(2, 3);

                            // Click button again
                            await page.evaluate(() => {
                                const buttons = Array.from(document.querySelectorAll('button')).filter(btn =>
                                    btn.textContent.includes('我要提案')
                                );
                                if (buttons.length > 0) buttons[0].click();
                            });

                            await randomDelay(2, 3);
                            logResult(`📄 URL after click: ${page.url()}`);
                        }
                    }
                } else if (finalUrl.includes('/cases/') && !finalUrl.includes('selected_tags')) {
                    logResult('📝 Proposal form detected');
                    await inspectProposalForm(page, logResult);
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
        // Close browser after automation
        logResult('\n🔒 Closing browser...');
        await randomDelay(2, 3);
        await browser.close();
        logResult('✅ Browser closed. Automation finished.');
    }
})();
