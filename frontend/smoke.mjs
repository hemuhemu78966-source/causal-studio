import { chromium } from '@playwright/test';
import fs from 'node:fs';
fs.mkdirSync("artifacts/ui-v3", { recursive: true });
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1080},deviceScaleFactor:1});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto('http://127.0.0.1:8000');
await page.getByRole('heading',{name:'Understand the system.' ,exact:false}).waitFor({timeout:90000});
await page.screenshot({path:'artifacts/ui-v3/desktop-overview.png',fullPage:true});
console.log(JSON.stringify({title:await page.title(),errors,overflow:await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)}));
await browser.close();

