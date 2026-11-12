// End-to-end checks against isolated local QA services. No mocked API responses.
// FEDMED_QA_COORDINATOR / FEDMED_QA_INSTITUTION point at smoke-test instances.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';

const coordinator = process.env.FEDMED_QA_COORDINATOR || 'http://127.0.0.1:49687';
const institution = process.env.FEDMED_QA_INSTITUTION || 'http://127.0.0.1:49691';
const output = 'test-results/browser';
await mkdir(output, {recursive:true});
const browser = await chromium.launch({channel:'chrome',headless:true});
const context = await browser.newContext({viewport:{width:1440,height:1000}});
const page = await context.newPage();
const errors = [], external = [];
page.on('pageerror', e => errors.push(e.message));
page.on('request', r => { if (!r.url().startsWith('http://127.0.0.1:') && !r.url().startsWith('data:')) external.push(r.url()); });
const snap = name => page.screenshot({path:`${output}/${name}.png`,fullPage:true});
async function login(url) {
  await page.goto(url);
  await page.getByLabel('Username',{exact:true}).waitFor();
  {
    await page.getByLabel('Username',{exact:true}).fill('operator');
    await page.getByLabel('Password',{exact:true}).fill('test-only-password');
    await page.getByRole('button',{name:'Sign in to workspace'}).click();
  }
  await page.getByRole('heading',{name:'Device readiness',exact:true}).waitFor();
}
try {
  await page.goto(coordinator);
  await page.getByLabel('Username',{exact:true}).waitFor();
  await snap('01-login');
  await login(coordinator);
  await snap('02-coordinator-overview');
  await page.getByRole('link',{name:'Institutions',exact:true}).click();
  await page.getByRole('button',{name:'Pair an institution'}).waitFor();
  await snap('03-institutions');
  await page.getByRole('button',{name:'Pair an institution'}).click();
  await page.getByRole('dialog').waitFor();
  assert.ok((await page.getByLabel('Pairing token').inputValue()).length > 20);
  await page.getByRole('button',{name:'Done',exact:true}).click();
  await page.getByRole('link',{name:'Training',exact:true}).click();
  await page.getByRole('button',{name:'New session',exact:true}).click();
  await snap('04-session-configuration');
  await page.getByRole('button',{name:'Close dialog'}).click();
  await snap('05-training');
  await page.getByRole('link',{name:'Experiments',exact:true}).click();
  await page.getByRole('button',{name:'Import result folders'}).click();
  await page.locator('tbody tr').first().waitFor({timeout:30000});
  await snap('06-experiments');
  await page.getByRole('link',{name:'Models',exact:true}).click();
  await page.getByRole('button',{name:'Import checkpoint'}).click();
  await page.getByLabel('Existing project checkpoint').waitFor();
  assert.ok(await page.getByLabel('Existing project checkpoint').locator('option').count() > 1);
  await snap('07-checkpoint-import');
  const candidates = await page.request.get(coordinator+'/api/v1/models/candidates').then(r => r.json()).then(r => r.data);
  const candidate = candidates.find(c => c.dataset === 'bloodmnist' && c.model === 'baseline_bn');
  assert.ok(candidate, 'A saved BloodMNIST checkpoint is available');
  await page.getByLabel('Existing project checkpoint').selectOption(String(candidates.indexOf(candidate)));
  const imported = page.waitForResponse(r => r.url().endsWith('/models/import') && r.request().method()==='POST');
  await page.getByRole('button',{name:'Verify & register checkpoint'}).click();
  const importJob = await (await imported).json().then(r => r.data);
  for(let attempts=0; attempts<60; attempts++) {
    const job = await page.request.get(coordinator+'/api/v1/jobs/'+importJob.id).then(r => r.json()).then(r => r.data);
    assert.notEqual(job.status, 'failed', job.error);
    if(job.status==='completed') break;
    if(attempts===59) throw new Error('Checkpoint validation timed out');
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  await page.reload();
  await page.getByRole('heading',{name:'Verified models',exact:true}).waitFor();
  await login(institution);
  await page.getByRole('link',{name:'Dataset',exact:true}).click();
  await page.getByRole('heading',{name:'Training class distribution',exact:true}).waitFor();
  await snap('08-institution-dataset');
  await page.getByRole('link',{name:'Training',exact:true}).click();
  await page.getByRole('heading',{name:'This institution’s accuracy',exact:true}).waitFor();
  await page.locator('.recharts-line-dot').first().waitFor();
  await page.getByRole('link',{name:'Predict',exact:true}).click();
  const modelSelect = page.getByLabel('Verified model');
  await modelSelect.waitFor();
  const bloodOption = await modelSelect.locator('option').filter({hasText:'BloodMNIST'}).first().getAttribute('value');
  await modelSelect.selectOption(bloodOption);
  await page.locator('.prediction-samples button').first().click();
  await page.getByRole('button',{name:'Classify image'}).click({timeout:60000});
  await page.locator('.prediction-result').waitFor({timeout:60000});
  assert.equal(await page.locator('.score-list > div').count(),3);
  await snap('09-local-prediction');
  await page.setViewportSize({width:390,height:844});
  await snap('10-mobile-prediction');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Mobile layout must not overflow horizontally');
  assert.deepEqual(errors,[], 'No browser runtime errors');
  assert.deepEqual(external,[], 'All browser assets and API calls remain local');
  await writeFile(`${output}/report.json`,JSON.stringify({status:'passed',errors,external,screenshots:10,checks:['login','pairing','configuration','live results','history import','checkpoint import','refresh persistence','dataset','real prediction','responsive layout','offline assets']},null,2));
  console.log('Browser workflow passed; screenshots in '+output);
} catch (error) {
  await snap('failure');
  console.error(await page.locator('body').innerText());
  throw error;
} finally { await browser.close(); }
