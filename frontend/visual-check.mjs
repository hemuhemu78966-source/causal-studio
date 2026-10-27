import { chromium, expect } from "@playwright/test";
import fs from "node:fs";
fs.mkdirSync("artifacts/ui-v3", { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
    reducedMotion: "reduce",
  });
  await page.goto("http://127.0.0.1:8000");
  await page
    .getByRole("heading", { name: "Understand the system.", exact: false })
    .waitFor({ timeout: 60000 });
  const results = [];
  const pages = [
    "Overview",
    "Causal graph",
    "Interventions",
    "Action planner",
    "Model lab",
    "Dataset",
  ];
  for (const name of pages) {
    await page
      .getByRole("navigation")
      .getByRole("button", { name, exact: true })
      .click();
    const audit = await page.evaluate(() => {
      const parse = (s) => (s.match(/[\d.]+/g) || []).map(Number);
      const lum = (c) =>
        c
          .slice(0, 3)
          .map((v) => {
            v /= 255;
            return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
          })
          .reduce((s, v, i) => s + v * [0.2126, 0.7152, 0.0722][i], 0);
      const over = (a, b) => {
        const alpha = a[3] ?? 1;
        return a.slice(0, 3).map((v, i) => v * alpha + b[i] * (1 - alpha));
      };
      const failures = [];
      let count = 0,
        min = 99;
      for (const el of document.querySelectorAll(
        "main *,nav button,.topbar *",
      )) {
        if (
          el instanceof SVGElement ||
          !el.getClientRects().length ||
          !Array.from(el.childNodes).some(
            (n) => n.nodeType === 3 && n.textContent.trim(),
          ) ||
          el.closest("button:disabled")
        )
          continue;
        const style = getComputedStyle(el);
        if (style.visibility !== "visible" || Number(style.opacity) < 0.95)
          continue;
        const layers = [];
        let current = el;
        while (current) {
          layers.push(parse(getComputedStyle(current).backgroundColor));
          current = current.parentElement;
        }
        let bg = [11, 17, 28];
        for (const layer of layers.reverse())
          if (layer.length >= 3) bg = over(layer, bg);
        const color = parse(style.color);
        const fg = over(color, bg);
        const l1 = lum(fg),
          l2 = lum(bg);
        const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
        count++;
        min = Math.min(min, ratio);
        const size = parseFloat(style.fontSize);
        const large =
          size >= 24 || (size >= 18.66 && Number(style.fontWeight) >= 700);
        const threshold = large ? 3 : 4.5;
        if (ratio < threshold - 0.01)
          failures.push({
            text: el.textContent.trim().slice(0, 60),
            ratio: +ratio.toFixed(2),
            fg: style.color,
            bg: bg.map(Math.round),
            size,
          });
      }
      return {
        textElements: count,
        minimumContrast: +min.toFixed(2),
        failures,
      };
    });
    results.push({ page: name, ...audit });
    console.log(name, JSON.stringify(audit));
  }
  const layouts = [];
  for (const width of [1440, 1060, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 950 });
    for (const name of ['Overview','Causal graph','Interventions','Action planner','Model lab','Dataset']) {
      if (width <= 650) await page.getByRole('button', { name: 'Open navigation' }).click();
      await page.getByRole('navigation').getByRole('button', { name, exact: true }).click();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
      layouts.push({width,page:name,overflow});
      if (overflow) throw Error('Layout overflow at '+width+' on '+name);
      if (width === 390 && name === 'Overview') await page.screenshot({path:'artifacts/ui-v3/mobile-overview.png',fullPage:true});
    }
  }
  fs.writeFileSync(
    "artifacts/ui-v3/contrast-and-layout.json",
    JSON.stringify({ results, layouts }, null, 2),
  );
  if (results.some((r) => r.failures.length))
    throw Error("Text contrast below thresholds");
  console.log("ALL TEXT CONTRAST AND FIVE VIEWPORT CHECKS PASSED");
} finally {
  await browser.close();
}
