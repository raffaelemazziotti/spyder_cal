const path = require('node:path');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const { PythonWorker } = require('../worker.cjs');
(async()=>{
const root = path.resolve(__dirname,'../..');
const executable = process.env.SPYDER_BACKEND || path.join(root,'desktop/build/spyder-cal-backend',process.platform==='win32'?'spyder-cal-backend.exe':'spyder-cal-backend');
const worker = new PythonWorker(executable, path.dirname(executable), {frozen:true});
const stages = [];
worker.on('message',m=>{if(m.event==='phase')stages.push(m.phase);if(['phase','error','fatal','offline'].includes(m.event)) console.log(JSON.stringify(m));});
try {
 console.log(JSON.stringify(await worker.ready));
 const session=JSON.parse(await fs.readFile(path.join(root,'desktop/.cache/fixture.json'),'utf8'));
 assert.equal((await worker.request('validate',{session})).event,'validated');
 const result=await worker.request('export',{session,path:path.join(root,'desktop/.cache/frozen-export.json')});
 assert.equal(result.paths.length,4);
 await assert.rejects(worker.request('probe',{libusb_path:'definitely-not-a-real-usb-library'}), /libusb|USB/i);
 const display = await worker.request('test_display',{screen:0});
 assert.ok(['display_tested','cancelled'].includes(display.event));
 assert.ok(stages.includes('display_test_visible'),'Real fullscreen window must be shown; Escape may dismiss it');
 console.log('Frozen runtime: JSON, USB error recovery, four exports and real PsychoPy window passed.');
}finally{await worker.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
