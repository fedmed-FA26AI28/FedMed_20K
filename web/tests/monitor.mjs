// Browser verification against an isolated portal serving actual smoke-run data.
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
const url=process.env.FEDMED_QA_MONITOR || 'http://127.0.0.1:48705';
const method=process.env.FEDMED_QA_METHOD || 'fedavg';
const output='test-results/monitor';
await mkdir(output,{recursive:true});
const browser=await chromium.launch({channel:'chrome',headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];
page.on('pageerror',e=>errors.push(e.message));
try {
  await page.goto(url);
  await page.getByLabel('Username',{exact:true}).fill('operator');
  await page.getByLabel('Password',{exact:true}).fill('test-only-password');
  await page.getByRole('button',{name:'Sign in to workspace'}).click();
  await page.getByRole('link',{name:'Training',exact:true}).click();
  await page.getByText(`LOCAL integration ${method}`,{exact:true}).first().click();
  const monitor=page.getByRole('region',{name:'Experiment monitor'});
  await monitor.getByRole('heading',{name:'Experiment monitor',exact:true}).waitFor();
  await monitor.getByRole('heading',{name:'Accuracy across rounds',exact:true}).waitFor();
  await monitor.scrollIntoViewIfNeeded();
  await page.screenshot({path:`${output}/${method}-learning.png`,fullPage:true});
  assert.ok(await monitor.locator('.recharts-line').count()>0);
  if(method==='scr') await monitor.getByRole('heading',{name:'SCR weighted risk',exact:true}).waitFor();
  const compare=monitor.getByLabel('Compare run');
  if(await compare.locator('option').count()>1) {
    await compare.selectOption({index:1});
    await monitor.getByRole('heading',{name:'Test accuracy comparison',exact:true}).waitFor();
    assert.ok((await monitor.locator('p[role="status"]').innerText()).includes('Recorded') || (await monitor.locator('p[role="status"]').innerText()).includes('Conditions'));
    await compare.selectOption('');
  }
  await monitor.getByRole('tab',{name:'Client diagnostics',exact:true}).click();
  assert.equal(await monitor.locator('tbody').first().locator('tr').count(),3);
  await monitor.getByLabel('Monitor source').selectOption('0');
  await page.waitForFunction(()=>document.querySelector('#monitor-panel tbody')?.querySelectorAll('tr').length===1);
  await monitor.getByLabel('Monitor source').selectOption('all');
  await monitor.getByRole('tab',{name:'Class performance',exact:true}).click();
  await monitor.getByRole('heading',{name:'Confusion matrix',exact:true}).waitFor();
  assert.equal(await monitor.locator('.confusion-table tbody tr').count(),8);
  await page.screenshot({path:`${output}/${method}-classes.png`,fullPage:true});
  await monitor.getByRole('tab',{name:'System & timing',exact:true}).click();
  await monitor.getByRole('heading',{name:'Training process memory',exact:true}).waitFor();
  await monitor.getByRole('tab',{name:'Live logs',exact:true}).click();
  await monitor.getByLabel('Search training logs').fill('Epoch');
  await page.waitForFunction(()=>document.querySelector('.monitor-log')?.textContent.includes('Epoch ['));
  await page.screenshot({path:`${output}/${method}-logs.png`,fullPage:true});
  const csvHref=await monitor.getByRole('link',{name:'CSV',exact:true}).getAttribute('href');
  const csv=await page.request.get(url+csvHref);
  assert.equal(csv.status(),200);
  assert.ok((await csv.text()).includes('client_round'));
  const jsonHref=await monitor.getByRole('link',{name:'JSON',exact:true}).getAttribute('href');
  const exported=await page.request.get(url+jsonHref).then(r=>r.json());
  assert.ok(exported.logs.length && exported.evaluations.length);
  await monitor.getByRole('button',{name:'Pause',exact:true}).click();
  await monitor.getByRole('button',{name:'Resume',exact:true}).waitFor();
  await page.setViewportSize({width:390,height:844});
  await monitor.getByRole('tab',{name:'Class performance',exact:true}).click();
  await page.screenshot({path:`${output}/${method}-mobile.png`,fullPage:true});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'No page-level horizontal overflow');
  await page.reload();
  await page.getByRole('heading',{name:'Training sessions',exact:true}).waitFor();
  assert.deepEqual(errors,[]);
  await writeFile(`${output}/${method}-report.json`,JSON.stringify({method,source:url,errors,csv_export:true,json_export:true,source_filter:true,mobile:true,real_recorded_data:true},null,2));
  console.log('Monitor browser checks passed for '+method);
} finally {
  await browser.close();
}
