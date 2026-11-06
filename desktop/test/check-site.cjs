// Static GitHub Pages check; no desktop/sensor operations or network access.
const { app, BrowserWindow } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
app.commandLine.appendSwitch('disable-gpu');
app.on('window-all-closed', () => {});
const watchdog = setTimeout(() => app.exit(1), 60000);
app.whenReady().then(async () => {
  try {
    for(const page of ['index.html','installation.html','api.html']) {
    for(const [width,height] of [[1200,900],[390,844],[320,740]]) {
      const window = new BrowserWindow({width,height,useContentSize:true,show:false,webPreferences:{offscreen:true,sandbox:true,contextIsolation:true,nodeIntegration:false}});
      let painted;
      window.webContents.on('paint', (_event, _dirty, image) => { painted = image; });
      await window.loadFile(path.join(root,'docs',page));
      await window.webContents.executeJavaScript('document.fonts.ready.then(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))))');
      const state = await window.webContents.executeJavaScript(`({
        title:document.title, width:innerWidth, overflow:document.documentElement.scrollWidth>innerWidth,
        images:[...document.images].every(image=>image.complete && image.naturalWidth>0),
        downloads:[...document.querySelectorAll('.download-button')].map(a=>a.href),
        links:[...document.querySelectorAll('a')].map(a=>a.getAttribute('href')),
        scripts:document.scripts.length
      })`);
      assert.equal(state.overflow,false,`No horizontal overflow at ${width}px`);
      assert.equal(state.width,width,'Renderer uses the requested viewport');
      assert.equal(state.images,true); assert.equal(state.scripts,0);
      assert.equal(state.downloads.length,page==='index.html'?2:0);
      assert.ok(state.downloads.every(href=>href.startsWith('https://github.com/raffaelemazziotti/spyder_cal/releases/latest/download/')));
      for(const link of state.links.filter(href=>!href.startsWith('https:'))) {
        const [file,fragment]=link.split('#');
        const target=path.join(root,'docs',file==='./'?'index.html':file||page);
        assert.ok(fs.existsSync(target),link);
        if(fragment) assert.ok(fs.readFileSync(target,'utf8').includes(`id="${fragment}"`),`Missing anchor ${link}`);
      }
      painted=undefined;
      window.webContents.invalidate();
      await new Promise((resolve,reject)=>{
        const end=Date.now()+5000;
        const wait=()=>{ if(painted && painted.getSize().width===width && painted.getSize().height===height && painted.toBitmap().some((v,i)=>i%4!==3 && v<200))return resolve(); if(Date.now()>end)return reject(new Error(`Rendered frame missing: ${page} at ${width}px`)); setTimeout(wait,20); };wait();
      });
      const pixels=painted.toBitmap();
      assert.ok(pixels.some((value,index)=>index%4!==3 && value<200),'Screenshot contains rendered content');
      if(width!==320) fs.writeFileSync(path.join(root,`desktop/.cache/site-${page.replace('.html','')}-${width}.png`),painted.toPNG());
      window.destroy();
    }
    }
    console.log('Site passed: homepage and both guides at 1200/390/320px, local links and anchors, images, download URLs, no JavaScript.');
    clearTimeout(watchdog);app.quit();
  } catch(error){console.error(error);clearTimeout(watchdog);app.exit(1);}
});
