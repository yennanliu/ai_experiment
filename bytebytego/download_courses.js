#!/usr/bin/env node
/**
 * ByteByteGo Course PDF Downloader
 *
 * Usage:
 *   node download_courses.js
 *
 * Prerequisites:
 *   npm install playwright
 *   npx playwright install chromium
 *
 * Set env vars or edit the credentials below:
 *   BBG_EMAIL=your@email.com
 *   BBG_PASSWORD=yourpassword
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const EMAIL = process.env.BBG_EMAIL || '';
const PASSWORD = process.env.BBG_PASSWORD || '';
const OUTPUT_DIR = process.env.BBG_OUTPUT_DIR || path.join(__dirname, 'pdfs');

if (!EMAIL || !PASSWORD) {
  console.error('Error: Set BBG_EMAIL and BBG_PASSWORD environment variables.');
  process.exit(1);
}

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function login(page) {
  console.log('Navigating to login page...');
  await page.goto('https://bytebytego.com/sign-in', { waitUntil: 'networkidle' });

  await page.fill('input[type="email"]', EMAIL);
  await page.fill('input[type="password"]', PASSWORD);
  await page.click('button[type="submit"]');

  // Wait for redirect after login
  await page.waitForURL('**/my-courses**', { timeout: 15000 }).catch(() => {
    // May redirect elsewhere; check if we're logged in
  });
  await sleep(2000);
  console.log('Logged in.');
}

async function getCourseLinks(page) {
  console.log('Fetching course list from /my-courses...');
  await page.goto('https://bytebytego.com/my-courses', { waitUntil: 'networkidle' });
  await sleep(2000);

  const courses = await page.evaluate(() => {
    const links = [];
    // Course cards typically have links to /courses/<slug>
    document.querySelectorAll('a[href*="/courses/"]').forEach(el => {
      const href = el.href;
      const title = el.innerText.trim() || el.querySelector('h2,h3,h4,span')?.innerText.trim() || href;
      if (!links.find(c => c.url === href)) {
        links.push({ url: href, title: title.substring(0, 80) });
      }
    });
    return links;
  });

  console.log(`Found ${courses.length} course link(s).`);
  return courses;
}

async function getChapterLinks(page, courseUrl) {
  await page.goto(courseUrl, { waitUntil: 'networkidle' });
  await sleep(2000);

  // Collect all chapter/lesson links within the course
  const chapters = await page.evaluate(() => {
    const links = [];
    const seen = new Set();
    document.querySelectorAll('a[href*="/courses/"]').forEach(el => {
      const href = el.href;
      if (!seen.has(href) && href !== window.location.href) {
        seen.add(href);
        links.push({ url: href, title: el.innerText.trim().substring(0, 80) });
      }
    });
    return links;
  });

  return chapters;
}

async function savePageAsPdf(page, url, outputPath) {
  await page.goto(url, { waitUntil: 'networkidle' });
  await sleep(2000);

  // Expand any collapsed sections
  await page.evaluate(() => {
    document.querySelectorAll('[aria-expanded="false"]').forEach(el => el.click());
  });
  await sleep(1000);

  await page.pdf({
    path: outputPath,
    format: 'A4',
    printBackground: true,
    margin: { top: '20mm', bottom: '20mm', left: '15mm', right: '15mm' },
  });
}

function sanitizeFilename(name) {
  return name.replace(/[^a-zA-Z0-9\-_\u4e00-\u9fff ]/g, '_').trim().substring(0, 100);
}

async function main() {
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: false }); // headless:false so you can handle 2FA if needed
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    await login(page);

    const courses = await getCourseLinks(page);
    if (courses.length === 0) {
      console.error('No courses found. Are you logged in? Check the selectors.');
      return;
    }

    for (const course of courses) {
      const courseName = sanitizeFilename(course.title || 'course');
      const courseDir = path.join(OUTPUT_DIR, courseName);
      if (!fs.existsSync(courseDir)) fs.mkdirSync(courseDir, { recursive: true });

      console.log(`\n=== Course: ${courseName} ===`);
      console.log(`URL: ${course.url}`);

      // Option A: Save entire course overview as one PDF
      const overviewPdf = path.join(OUTPUT_DIR, `${courseName}.pdf`);
      console.log(`Saving course overview PDF: ${overviewPdf}`);
      await savePageAsPdf(page, course.url, overviewPdf);

      // Option B: Also save individual chapters
      const chapters = await getChapterLinks(page, course.url);
      console.log(`Found ${chapters.length} chapter(s) in course.`);

      for (let i = 0; i < chapters.length; i++) {
        const ch = chapters[i];
        const chName = sanitizeFilename(ch.title || `chapter_${i + 1}`);
        const chPdf = path.join(courseDir, `${String(i + 1).padStart(2, '0')}_${chName}.pdf`);

        if (fs.existsSync(chPdf)) {
          console.log(`  [SKIP] ${chName} (already exists)`);
          continue;
        }

        console.log(`  [${i + 1}/${chapters.length}] Saving: ${chName}`);
        try {
          await savePageAsPdf(page, ch.url, chPdf);
          await sleep(1500 + Math.random() * 1000);
        } catch (err) {
          console.error(`  [FAIL] ${chName}: ${err.message}`);
        }
      }
    }

    console.log('\nDone! PDFs saved to:', OUTPUT_DIR);
  } finally {
    await browser.close();
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
