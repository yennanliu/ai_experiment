const fs = require("node:fs/promises");
const path = require("node:path");
const puppeteer = require("puppeteer");

const DEFAULT_START_URL = "https://www.tasker.com.tw/cases?selected_tags=1";
const ARTIFACTS_DIR = path.join(__dirname, "..", "artifacts", "tasker");

function envFlag(name, defaultValue) {
  const v = process.env[name];
  if (v == null) return defaultValue;
  return v === "1" || v.toLowerCase() === "true" || v.toLowerCase() === "yes";
}

function envInt(name, fallback) {
  const v = process.env[name];
  if (v == null || v.trim() === "") return fallback;
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? n : fallback;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function sleepRandom(minMs, maxMs) {
  const span = Math.max(0, maxMs - minMs);
  const ms = minMs + Math.floor(Math.random() * (span + 1));
  await sleep(ms);
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function screenshot(page, name) {
  await ensureDir(ARTIFACTS_DIR);
  const file = path.join(
    ARTIFACTS_DIR,
    `${new Date().toISOString().replaceAll(":", "-")}-${name}.png`
  );
  await page.screenshot({ path: file, fullPage: true });
  return file;
}

async function clickByText(
  page,
  text,
  { tag = "*", timeoutMs = 10_000, match = "contains" } = {}
) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const didClick = await page.evaluate(
      ({ text, tag, match }) => {
        const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
        const wanted = norm(text);
        const candidates = Array.from(document.querySelectorAll(tag));
        const el = candidates.find((e) => {
          const got = norm(e.textContent);
          if (!got) return false;
          return match === "exact" ? got === wanted : got.includes(wanted);
        });
        if (!el) return false;

        const clickable =
          el.closest("button, a, [role='button'], input[type='button'], input[type='submit']") ||
          el;
        clickable.scrollIntoView({ block: "center", inline: "center" });
        clickable.click();
        return true;
      },
      { text, tag, match }
    );
    if (didClick) return true;
    await sleep(250);
  }
  return false;
}

async function getFirstCaseUrl(page) {
  const url = await page.evaluate(() => {
    const links = Array.from(document.querySelectorAll("a[href]"));
    const candidates = links
      .map((a) => a.getAttribute("href"))
      .filter(Boolean)
      .map((href) => {
        try {
          return new URL(href, location.href).toString();
        } catch {
          return null;
        }
      })
      .filter(Boolean);

    const isBad = (u) =>
      u.includes("/case/post_instruction") ||
      u.includes("/case/post") ||
      u.includes("post_instruction") ||
      u.includes("post_case") ||
      u.includes("/cases/top");

    const isCaseDetail = (u) => /\/cases\/\d+/.test(u) || /\/case\/\d+/.test(u);
    const preferred = candidates.find((u) => !isBad(u) && isCaseDetail(u));
    if (preferred) return preferred;

    // Extra heuristic: sometimes there are query-based case links; try to find an href containing digits.
    const maybe = candidates.find((u) => !isBad(u) && /\d{4,}/.test(u) && u.includes("/case"));
    return maybe || null;
  });
  return url || null;
}

function parseBudgetFromText(text) {
  const t = (text || "").replace(/\s+/g, " ");
  const patterns = [
    /([0-9][0-9,]*)\s*[-~～]\s*([0-9][0-9,]*)\s*元?/,
    /\$\s*([0-9][0-9,]*)\s*[-~～]\s*\$?\s*([0-9][0-9,]*)/,
  ];

  for (const re of patterns) {
    const m = t.match(re);
    if (!m) continue;
    const low = Number(m[1].replaceAll(",", ""));
    const high = Number(m[2].replaceAll(",", ""));
    if (Number.isFinite(low) && Number.isFinite(high) && low > 0 && high >= low) {
      return { low, high, raw: m[0] };
    }
  }
  return null;
}

async function extractBudgetRange(page) {
  const text = await page.evaluate(() => document.body.innerText || "");
  return parseBudgetFromText(text);
}

async function fillMinMaxPrice(page, { low, high }) {
  const targets = await page.evaluate(() => {
    const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

    const scoreInput = (input) => {
      const attrs = [
        input.getAttribute("name"),
        input.getAttribute("id"),
        input.getAttribute("placeholder"),
        input.getAttribute("aria-label"),
      ]
        .filter(Boolean)
        .map((s) => s.toLowerCase());
      const joined = attrs.join(" ");
      return joined;
    };

    const inputs = Array.from(document.querySelectorAll("input"));
    const priceInputs = inputs
      .filter((i) => {
        const type = (i.getAttribute("type") || "").toLowerCase();
        if (type && !["text", "number", "tel"].includes(type)) return false;
        if (i.disabled || i.readOnly) return false;
        const s = scoreInput(i);
        return s.includes("price") || s.includes("金額") || s.includes("費用") || s.includes("報價");
      })
      .slice(0, 10);

    // Try to identify "min" and "max" by nearby labels / placeholders.
    const classify = (input) => {
      const s = scoreInput(input);
      const wrapperText = norm(
        (input.closest("label, .form-group, .form-item, .input-group, .row, .col") || input)
          .textContent
      );

      const blob = `${s} ${wrapperText}`.toLowerCase();
      const isMin = blob.includes("min") || blob.includes("最低") || blob.includes("下限") || blob.includes("起");
      const isMax = blob.includes("max") || blob.includes("最高") || blob.includes("上限") || blob.includes("迄");
      return { isMin, isMax };
    };

    let min = null;
    let max = null;
    for (const i of priceInputs) {
      const c = classify(i);
      if (!min && c.isMin && !c.isMax) min = i;
      if (!max && c.isMax && !c.isMin) max = i;
    }

    // Fallback: if we found 2+, just take first as min and second as max.
    if ((!min || !max) && priceInputs.length >= 2) {
      min = min || priceInputs[0];
      max = max || priceInputs[1];
    }

    const toSelector = (el) => {
      if (!el) return null;
      if (el.id) return `#${CSS.escape(el.id)}`;
      const name = el.getAttribute("name");
      if (name) return `input[name="${CSS.escape(name)}"]`;
      return null;
    };

    return {
      minSelector: toSelector(min),
      maxSelector: toSelector(max),
    };
  });

  const { minSelector, maxSelector } = targets || {};
  if (!minSelector || !maxSelector) {
    return { ok: false, reason: "Could not locate min/max price inputs" };
  }

  const minValue = low;
  const maxValue = high;

  await page.waitForSelector(minSelector, { timeout: 10_000 });
  await page.waitForSelector(maxSelector, { timeout: 10_000 });

  // Clear + type with a tiny delay for realism.
  await page.click(minSelector, { clickCount: 3 });
  await page.keyboard.type(String(minValue), { delay: 25 });
  await sleepRandom(300, 900);
  await page.click(maxSelector, { clickCount: 3 });
  await page.keyboard.type(String(maxValue), { delay: 25 });

  return { ok: true, minSelector, maxSelector, minValue, maxValue };
}

async function waitForNewTab(browser, openerPage, timeoutMs = 15_000) {
  const openerTarget = openerPage.target();
  const target = await browser.waitForTarget(
    (t) => t.opener() === openerTarget && t.type() === "page",
    { timeout: timeoutMs }
  );
  const page = await target.page();
  if (!page) throw new Error("New tab opened, but could not resolve Page");
  return page;
}

async function main() {
  const headless = envFlag("HEADLESS", false);
  const slowMo = envInt("SLOW_MO_MS", 0);
  const startUrl = process.env.START_URL || DEFAULT_START_URL;

  let browser = null;
  let page = null;
  try {
    console.log(`[tasker] Launching browser (headless=${headless})`);
    browser = await puppeteer.launch({
      headless: headless ? "new" : false,
      slowMo,
      defaultViewport: null,
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    page = await browser.newPage();
    await page.setUserAgent(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    );
    page.setDefaultTimeout(20_000);

    console.log("[tasker] Opening cases list");
    await page.goto(startUrl, { waitUntil: "domcontentloaded" });
    await sleepRandom(1000, 2500);

    // Wait for some links to render; Tasker is SPA-like.
    await page.waitForFunction(() => document.querySelectorAll("a[href]").length > 10, {
      timeout: 20_000,
    });

    const firstCaseUrl = await getFirstCaseUrl(page);
    if (!firstCaseUrl) throw new Error("Could not find a case URL on the listing page");
    console.log(`[tasker] Opening first case: ${firstCaseUrl}`);

    await page.goto(firstCaseUrl, { waitUntil: "domcontentloaded" });
    await sleepRandom(1000, 2500);

    console.log('[tasker] Clicking "我要提案"');
    const clicked = await clickByText(page, "我要提案", { timeoutMs: 12_000 });
    if (!clicked) throw new Error('Could not find/click "我要提案"');

    const proposalPage = await waitForNewTab(browser, page, 20_000);
    await proposalPage.bringToFront();
    proposalPage.setDefaultTimeout(20_000);
    await sleepRandom(1200, 2600);

    console.log("[tasker] Extracting budget range");
    const budget = await extractBudgetRange(proposalPage);
    if (!budget) {
      throw new Error("Could not parse budget range from proposal page text");
    }
    console.log(`[tasker] Budget: ${budget.raw} (low=${budget.low}, high=${budget.high})`);

    console.log("[tasker] Filling min/max price");
    const fillResult = await fillMinMaxPrice(proposalPage, budget);
    if (!fillResult.ok) throw new Error(fillResult.reason);
    console.log(
      `[tasker] Filled min=${fillResult.minValue}, max=${fillResult.maxValue} (${fillResult.minSelector}, ${fillResult.maxSelector})`
    );

    await sleepRandom(1000, 2500);
    console.log('[tasker] Clicking "送出提案"');
    const submitted = await clickByText(proposalPage, "送出提案", { timeoutMs: 12_000 });
    if (!submitted) throw new Error('Could not find/click "送出提案"');

    await sleepRandom(1500, 3000);
    console.log("[tasker] Done (submitted click attempted).");
  } catch (err) {
    console.log(`[tasker] FAILED: ${err && err.message ? err.message : String(err)}`);
    try {
      if (page) {
        const file = await screenshot(page, "error");
        console.log(`[tasker] Screenshot saved: ${file}`);
      }
    } catch {
      // ignore
    }
    throw err;
  } finally {
    if (browser) {
      if (!envFlag("KEEP_OPEN", false)) {
        await browser.close();
      } else {
        console.log("[tasker] KEEP_OPEN=1 set; leaving browser open.");
      }
    }
  }
}

main().catch(() => {
  process.exitCode = 1;
});

