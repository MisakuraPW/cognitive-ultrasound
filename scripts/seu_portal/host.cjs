'use strict';
const fs=require('node:fs');
const net=require('node:net');
const path=require('node:path');
const os=require('node:os');
const crypto=require('node:crypto');
const {stateDir,stateFile,endpoint}=require('./client.cjs');
const {validated}=require('./commands.cjs');
const portal='https://sc.seu.edu.cn/webssh-connect';
function playwright() {
  const candidates=['playwright-core',path.join(os.homedir(),'.cache','codex-runtimes','codex-primary-runtime','dependencies','node','node_modules','playwright-core')];
  for(const name of candidates) {
    try { return require(require.resolve(name)); }
    catch(error) { if(error.code!=='MODULE_NOT_FOUND') throw error; }
  }
  throw new Error('Install playwright-core or use the bundled Codex Node runtime.');
}
function browserPath() {
  const places=[
    [process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe'],
    [process.env.ProgramFiles,'Google/Chrome/Application/chrome.exe'],
    [process.env.LOCALAPPDATA,'Google/Chrome/Application/chrome.exe'],
  ];
  for(const [root,name] of places) if(root && fs.existsSync(path.join(root,name))) return path.join(root,name);
  throw new Error('Local Microsoft Edge or Google Chrome is required.');
}
function safePage(page) {
  try { const url=new URL(page.url()); return {host:url.hostname, page: url.origin==='https://sc.seu.edu.cn' && url.pathname==='/webssh-connect'?'webssh':undefined}; }
  catch { return {host:'starting'}; }
}

async function run() {
  fs.mkdirSync(stateDir,{recursive:true,mode:0o700});
  const token=crypto.randomBytes(32).toString('hex');
  let browser, context, busy=false, ready=false, stopping=false;
  const server=net.createServer(socket => {
    socket.setEncoding('utf8');
    socket.setTimeout(185000,()=>socket.destroy());
    let input='';
    socket.on('error',()=>{});
    socket.on('data',async function onData(part) {
      input+=part;
      if(input.length>4096) return socket.destroy();
      if(!input.includes('\n')) return;
      socket.removeListener('data',onData);
      let ownsBusy=false;
      try {
        const payload=JSON.parse(input);
        if(typeof payload.token!=='string' || payload.token.length!==token.length ||
            !crypto.timingSafeEqual(Buffer.from(token),Buffer.from(payload.token,'utf8'))) {
          throw new Error('Invalid local session identity.');
        }
        const action=validated(payload.request);
        if(action.op==='session') {
          const pages=(context?.pages()||[]).map(safePage);
          return socket.end(JSON.stringify({ok:true,protocol:2,browser:ready?'open':'starting',pages}));
        }
        if(action.op==='close') {
          if(busy) throw new Error('Another operation is in progress.');
          socket.end(JSON.stringify({ok:true,browser:'closing'}));
          void stop(); return;
        }
        if(!ready) throw new Error('The dedicated browser is starting.');
        if(busy) return socket.end(JSON.stringify({ok:false,code:'BUSY',error:'Another operation is active; the new request has not started.'}));
        ownsBusy=busy=true;
        let result;
        if(action.op==='reconnect') {
          const old=context.pages().filter(p=>schoolPortalTab(p));
          const fresh=await context.newPage();
          await fresh.goto(portal,{waitUntil:'commit',timeout:25000});
          result=await require('./terminal.cjs').execute(context,{op:'probe'},fresh);
          if(result.ok) for(const p of old) await p.close().catch(()=>{});
        } else {
          const time=new Date().toISOString();
          try { result=await require('./terminal.cjs').execute(context, action); }
          finally {
            if(action.op==='submit') {
              const id=/Submitted batch job\s+(\d+)/i.exec(result?.output||'')?.[1];
              const receipt={time,task:action.task,partition:action.partition||null,jobId:id||null,
                outcome:id?'accepted':(result?'check-output':'unknown-check-history')};
              fs.appendFileSync(path.join(stateDir,'submitted.jsonl'),JSON.stringify(receipt)+'\n',{mode:0o600});
            }
          }
        }
        socket.end(JSON.stringify(result));
      } catch(error) {
        socket.end(JSON.stringify({ok:false,error:error.message}));
      } finally { if(ownsBusy) busy=false; }
    });
  });
  function removeState() {
    try {if(JSON.parse(fs.readFileSync(stateFile,'utf8')).token===token)fs.unlinkSync(stateFile);}catch{ /* already closed */ }
  }
  async function stop() {
    if(stopping)return;
    stopping=true;
    server.close(); removeState();
    try {await browser?.close();} finally {process.exit(0);}
  }
  function schoolPortalTab(page) {
    try { return new URL(page.url()).origin===new URL(portal).origin && new URL(page.url()).pathname==='/webssh-connect'; }
    catch {return false;}
  }
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen({path:endpoint,readableAll:false,writableAll:false},resolve);});
  try {
    browser=await playwright().chromium.launch({executablePath:browserPath(),headless:false,
      args:['--disable-background-timer-throttling','--disable-renderer-backgrounding','--window-size=1180,850']});
    browser.on('disconnected',()=>{void stop();});
    context=await browser.newContext({viewport:{width:1160,height:810},acceptDownloads:true});
    const page=await context.newPage();
    await page.goto(portal,{waitUntil:'commit',timeout:25000});
    ready=true;
    const tmp=stateFile+'.'+process.pid;
    fs.writeFileSync(tmp,JSON.stringify({token,pid:process.pid}),{mode:0o600});
    fs.renameSync(tmp,stateFile);
  } catch(error) {
    server.close(); removeState(); await browser?.close(); throw error;
  }
}
run().catch(error=>{console.error(error.message);process.exitCode=1;});
