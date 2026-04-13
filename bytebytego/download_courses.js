/**
 * ByteByteGo Course PDF Downloader
 * Run via: node download_courses.js
 * Requires: npm install playwright
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUTPUT_DIR = path.join(__dirname, 'pdfs');

const COURSES = [
  { name: 'How_to_Write_a_Good_Resume', startUrl: 'https://bytebytego.com/courses/tech-resume/p0-c2-introduction' },
  { name: 'Coding_Interview_Patterns', startUrl: 'https://bytebytego.com/courses/coding-interview-patterns/introduction' },
  { name: 'System_Design_Interview', startUrl: 'https://bytebytego.com/courses/system-design-interview/foreword' },
  { name: 'Object_Oriented_Design_Interview', startUrl: 'https://bytebytego.com/courses/ood-interview/introduction' },
  { name: 'Machine_Learning_System_Design', startUrl: 'https://bytebytego.com/courses/machine-learning-system-design-interview/introduction' },
  { name: 'Mobile_System_Design', startUrl: 'https://bytebytego.com/courses/mobile-system-design-interview/introduction' },
  { name: 'Generative_AI_System_Design', startUrl: 'https://bytebytego.com/courses/genai-system-design-interview/introduction' },
];

async function getChapterUrls(page) {
  return await page.evaluate(() => {
    const results = [];
    const seen = new Set();
    document.querySelectorAll('*').forEach(el => {
      const menuId = el.getAttribute('data-menu-id');
      if (menuId && menuId.includes('/courses/')) {
        // Extract the path from "rc-menu-uuid-XXXX-/courses/..."
        const match = menuId.match(/\/courses\/.+/);
        if (match && !seen.has(match[0])) {
          seen.add(match[0]);
          results.push({
            title: el.innerText.trim().split('\n')[0],
            url: 'https://bytebytego.com' + match[0]
          });
        }
      }
    });
    return results;
  });
}

async function downloadCourse(page, course, outputDir) {
  console.log(`\n=== Downloading: ${course.name} ===`);

  // Navigate to first chapter to load sidebar
  await page.goto(course.startUrl, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  const chapters = await getChapterUrls(page);
  console.log(`  Found ${chapters.length} chapters`);

  if (chapters.length === 0) {
    console.log(`  WARNING: No chapters found, saving current page only`);
    const pdfPath = path.join(outputDir, `${course.name}.pdf`);
    await page.pdf({ path: pdfPath, format: 'A4', printBackground: true, margin: { top: '15mm', bottom: '15mm', left: '10mm', right: '10mm' } });
    console.log(`  Saved: ${pdfPath}`);
    return;
  }

  const courseDir = path.join(outputDir, course.name);
  if (!fs.existsSync(courseDir)) fs.mkdirSync(courseDir, { recursive: true });

  for (let i = 0; i < chapters.length; i++) {
    const ch = chapters[i];
    const safeName = ch.title.replace(/[^a-zA-Z0-9\-_ ]/g, '_').trim().substring(0, 80);
    const pdfPath = path.join(courseDir, `${String(i + 1).padStart(2, '0')}_${safeName}.pdf`);

    if (fs.existsSync(pdfPath)) {
      console.log(`  [${i+1}/${chapters.length}] SKIP (exists): ${safeName}`);
      continue;
    }

    console.log(`  [${i+1}/${chapters.length}] Saving: ${safeName}`);
    try {
      await page.goto(ch.url, { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(1500);

      // Set wider viewport so content doesn't overflow on PDF export
      await page.setViewportSize({ width: 1400, height: 900 });

      // Hide sidebar for cleaner PDF
      await page.evaluate(() => {
        const sidebar = document.querySelector('.ant-layout-sider, [class*="sider"], [class*="Sider"]');
        if (sidebar) sidebar.style.display = 'none';
        const header = document.querySelector('header');
        if (header) header.style.display = 'none';

        // Center and constrain main content width to prevent right-side clipping
        const main = document.querySelector('main, [class*="content"], article');
        if (main) {
          main.style.maxWidth = '900px';
          main.style.margin = '0 auto';
        }
      });

      await page.pdf({
        path: pdfPath,
        format: 'A4',
        printBackground: true,
        scale: 0.85,
        margin: { top: '15mm', bottom: '15mm', left: '15mm', right: '15mm' }
      });

      await page.waitForTimeout(800);
    } catch (err) {
      console.log(`  [${i+1}/${chapters.length}] FAILED: ${err.message}`);
    }
  }

  console.log(`  Done! PDFs in: ${courseDir}`);
}

async function main() {
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  // Connect to existing Chrome session via remote debugging
  let browser;
  try {
    browser = await chromium.connectOverCDP('http://localhost:9222');
    console.log('Connected to existing Chrome session');
  } catch (e) {
    console.log('Could not connect to Chrome on port 9222, launching new browser...');
    browser = await chromium.launch({ headless: false });
  }

  const contexts = browser.contexts();
  const context = contexts[0] || await browser.newContext();
  const page = await context.newPage();

  try {
    // Verify logged in
    await page.goto('https://bytebytego.com/my-courses', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    const url = page.url();
    if (url.includes('sign-in') || url.includes('login')) {
      console.error('Not logged in! Please log in first.');
      process.exit(1);
    }
    console.log('Logged in. Starting downloads...\n');

    for (const course of COURSES) {
      await downloadCourse(page, course, OUTPUT_DIR);
    }

    console.log(`\n✓ All done! PDFs saved to: ${OUTPUT_DIR}`);
  } finally {
    await page.close();
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
