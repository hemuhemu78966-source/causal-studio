import { chromium, expect } from "@playwright/test";
import fs from "node:fs";
fs.mkdirSync("artifacts/ui-v3", { recursive: true });
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  reducedMotion: "reduce",
  acceptDownloads: true,
});
const page = await context.newPage();
const failures = [],
  checks = [];
page.on("pageerror", (e) => failures.push(e.message));
async function check(name, fn) {
  await fn();
  checks.push(name);
  console.log("PASS", name);
}
const nav = (name) =>
  page.getByRole("navigation").getByRole("button", { name, exact: true });
await page.goto("http://127.0.0.1:8000");
await page
  .getByRole("heading", { name: "Understand the system.", exact: false })
  .waitFor({ timeout: 90000 });
await check("Desktop overview loads real model", async () => {
  await expect(page.locator(".data-ribbon strong").first()).toContainText(
    "2,000",
  );
  await expect(page.getByText("77.0%", { exact: true }).first()).toBeVisible();
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  await page.screenshot({
    path: "artifacts/ui-v3/desktop-overview.png",
    fullPage: true,
  });
});
await check("Graph interactions and known graph", async () => {
  await nav("Causal graph").click();
  await page.getByRole("button", { name: "Zoom in" }).click();
  await page.getByRole("button", { name: "Reset graph" }).click();
  await page
    .getByRole("button", { name: "Known benchmark graph", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Synthetic ground truth" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Learned graph", exact: true })
    .click();
});
await check("Keyboard graph inspection", async () => {
  await nav("Causal graph").click();
  await page
    .getByRole("button", { name: "Inspect Savings", exact: true })
    .focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".node-inspector")).toContainText("Savings");
  await page.getByRole("button", { name: "Clear node selection" }).click();
  await expect(page.locator(".node-inspector")).not.toBeVisible();
});
await check("Intervention computes a scenario", async () => {
  await nav("Interventions").click();
  await page
    .getByRole("button", { name: "Run simulation", exact: true })
    .click();
  await expect(page.getByText("MODEL-ESTIMATED EFFECT")).toBeVisible({
    timeout: 30000,
  });
  await expect(page.getByText("+55.8", { exact: false })).toBeVisible();
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  await page.screenshot({
    path: "artifacts/ui-v3/desktop-intervention.png",
    fullPage: true,
  });
});
await check("Intervention desired outcome is configurable", async () => {
  await page
    .getByRole("combobox", { name: "Desired outcome", exact: true })
    .selectOption("0");
  await page
    .getByRole("button", { name: "Run simulation", exact: true })
    .click();
  await expect(page.getByText("-55.8", { exact: false })).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByRole("combobox", { name: "Desired outcome", exact: true })
    .selectOption("1");
  await page
    .getByRole("button", { name: "Run simulation", exact: true })
    .click();
  await expect(page.getByText("+55.8", { exact: false })).toBeVisible({
    timeout: 30000,
  });
});
await check("Scenario export contains real probabilities", async () => {
  const d = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export scenario" }).click();
  const download = await d;
  const text = fs.readFileSync(await download.path(), "utf8");
  if (!JSON.parse(text).interventional) throw Error("Missing probabilities");
});
await check("Plan search and immutable feature", async () => {
  await nav("Action planner").click();
  await page
    .getByRole("combobox", { name: "Savings", exact: true })
    .selectOption("0");
  await page
    .getByRole("combobox", { name: "Credit Score", exact: true })
    .selectOption("0");
  await page.getByRole("button", { name: "Find action plans" }).click();
  await expect(page.locator(".plans")).toBeVisible({ timeout: 30000 });
  await expect(page.getByText("LOWEST COST", { exact: true })).toBeVisible();
  const d = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export action plans" }).click();
  const data = JSON.parse(fs.readFileSync(await (await d).path(), "utf8"));
  if (data.plans.some((p) => "Age" in p.interventions))
    throw Error("Changed immutable Age");
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  await page.screenshot({
    path: "artifacts/ui-v3/desktop-planner.png",
    fullPage: true,
  });
});
await check("Model evaluation and confusion matrix", async () => {
  await nav("Model lab").click();
  await expect(page.getByText("Random forest", { exact: true })).toBeVisible();
  await expect(page.locator(".matrix")).toBeVisible();
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  await page.screenshot({
    path: "artifacts/ui-v3/desktop-evaluation.png",
    fullPage: true,
  });
});
await check("Dataset search understands readable labels", async () => {
  await nav("Dataset").click();
  await page.getByRole("textbox", { name: "Search records" }).fill("Approved");
  await expect(page.locator(".records tbody tr").first()).toBeVisible();
  if ((await page.locator(".records tbody tr").count()) >= 100)
    throw Error("Search did not filter");
});
await check("CSV download", async () => {
  const d = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download CSV" }).click();
  const text = fs.readFileSync(await (await d).path(), "utf8");
  if (text.split("\n").length < 2000) throw Error("CSV incomplete");
});
await check("No guide or evaluator content in the product", async () => {
  for (const name of [
    "Overview",
    "Causal graph",
    "Interventions",
    "Action planner",
    "Model lab",
    "Dataset",
  ]) {
    await nav(name).click();
    const text = await page.locator("body").innerText();
    if (/evaluator|demo guide|academic edition|viva/i.test(text))
      throw Error("Academic guidance remains on " + name);
  }
  if ((await page.getByRole("navigation").getByRole("button").count()) !== 6)
    throw Error("Unexpected navigation items");
});
await check("Settings keyboard dismissal", async () => {
  await page.getByRole("button", { name: "Configure model" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
});
await check("Custom CSV trains through the UI", async () => {
  await page.getByRole("button", { name: "Configure model" }).click();
  await page
    .getByRole("combobox", { name: "Dataset", exact: true })
    .selectOption("custom");
  let csv = "Effort,Support,Result\n";
  for (let i = 0; i < 200; i++)
    csv += `${i % 2 ? "high" : "low"},${i % 3 ? "yes" : "no"},${i % 2 ? "pass" : "fail"}\n`;
  await page
    .getByLabel("Upload CSV")
    .setInputFiles({
      name: "student-demo.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(csv),
    });
  await expect(
    page.getByLabel("Target outcome", { exact: true }),
  ).toBeVisible();
  await page
    .getByLabel("Upload CSV")
    .setInputFiles({
      name: "invalid.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("OnlyColumn\nvalue\n"),
    });
  await expect(
    page.getByRole("button", { name: "Build workspace" }),
  ).toBeDisabled();
  await page
    .getByLabel("Upload CSV")
    .setInputFiles({
      name: "student-demo.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(csv),
    });
  await expect(
    page.getByRole("button", { name: "Build workspace" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Build workspace" }).click();
  await expect(page.getByRole("dialog")).not.toBeVisible({ timeout: 60000 });
  await expect(page.locator(".workspace-bar")).toContainText("Custom dataset");
  await expect(
    page.getByText("User-provided CSV", { exact: true }).first(),
  ).toBeVisible();
});
await check("API rejects invalid requests", async () => {
  const r = await page.request.post("http://127.0.0.1:8000/api/train", {
    data: { samples: 1 },
  });
  if (r.status() !== 422) throw Error("Expected 422");
});
await check("Report download", async () => {
  const d = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export report", exact: true })
    .click();
  const report = JSON.parse(fs.readFileSync(await (await d).path(), "utf8"));
  if (report.source !== "User-provided CSV") throw Error("Wrong report");
});
await check("Mobile navigation and no horizontal overflow", async () => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open navigation" }).click();
  await nav("Interventions").click();
  await expect(
    page.getByRole("heading", {
      name: "The intervention simulator.",
      exact: false,
    }),
  ).toBeVisible();
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw Error("Mobile page overflow");
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  await page.screenshot({
    path: "artifacts/ui-v3/mobile-intervention.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Open navigation" }).click();
  await nav("Overview").click();
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  await page.screenshot({
    path: "artifacts/ui-v3/mobile-overview.png",
    fullPage: true,
  });
});
if (failures.length) throw Error(failures.join("\n"));
fs.writeFileSync(
  "artifacts/ui-v3/browser-results.json",
  JSON.stringify({ checks, consoleErrors: failures }, null, 2),
);
console.log("ALL", checks.length, "BROWSER CHECKS PASSED");
await browser.close();
