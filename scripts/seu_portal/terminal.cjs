'use strict';
const crypto=require('node:crypto');
const portal='https://sc.seu.edu.cn/webssh-connect';
const origin=new URL(portal).origin;
const inputSelector='textarea.xterm-helper-textarea, .xterm textarea';
const pause=ms=>new Promise(r=>setTimeout(r,ms));

function schoolPage(page) {
  try {return new URL(page.url()).origin===origin;} catch {return false;}
}
async function findTerminal(context, selected) {
  const pages=selected?[selected]:context.pages();
  for(const page of pages.filter(schoolPage).reverse()) {
    for(const frame of page.frames()) {
      if(frame.url()!=='about:blank' && !schoolPage(frame))continue;
      const input=frame.locator(inputSelector).first();
      if(await input.count()) return {page,frame,input};
    }
  }
  return null;
}
async function getTerminal(context, selected) {
  let terminal=await findTerminal(context,selected);
  if(terminal)return terminal;
  let page=selected||context.pages().filter(schoolPage).at(-1);
  if(!page)throw new Error('LOGIN_REQUIRED: complete the school sign-in in the dedicated browser.');
  if(await page.locator('input[type=password]:visible').count())
    throw new Error('LOGIN_REQUIRED: complete sign-in in the dedicated browser.');
  const here=new URL(page.url());
  if(!selected && here.pathname!==new URL(portal).pathname) {
    page=await context.newPage();
    await page.goto(portal,{waitUntil:'commit',timeout:16000});
  }
  for(let i=0;i<15;i++) {
    terminal=await findTerminal(context,selected);
    if(terminal)return terminal;
    await pause(320);
  }
  const locations=context.pages().filter(schoolPage).map(p=>({
    path:new URL(p.url()).pathname,
    frames:p.frames().map(f=>{try {const u=new URL(f.url());return u.origin==='null'?u.protocol:u.origin;}catch{return 'unavailable';}}),
  }));
  const labels=schoolPage(page)?await page.getByRole('button').allTextContents():[];
  throw new Error('NO_TERMINAL: open 智算平台 → 远程终端 in the dedicated browser. '+
    JSON.stringify({locations,buttons:labels.map(x=>x.trim()).filter(Boolean).slice(0,12)}));
}

async function send(terminal,line) {
  if(line.includes('\n')||line.includes('\r'))throw new Error('Expected one shell input line.');
  await terminal.input.focus({timeout:3000});
  await terminal.input.evaluate((el,value)=>{
    const data=new DataTransfer();
    data.setData('text/plain',value);
    el.dispatchEvent(new ClipboardEvent('paste',{clipboardData:data,bubbles:true,cancelable:true}));
  },line);
  await terminal.input.press('Enter',{timeout:3000});
}

// Canvas-based xterm versions implement ordinary selection/copy themselves.
// The copy event uses an in-memory DataTransfer, never the OS clipboard.
async function readScreen(terminal) {
  const region=terminal.input.locator('xpath=ancestor::*[contains(concat(" ",normalize-space(@class)," ")," xterm ")][1]');
  const textRows=region.locator('.xterm-rows > div,.xterm-accessibility-tree > div');
  if(await textRows.count()) return (await textRows.allTextContents()).join('\n');
  const screen=region.locator('.xterm-screen').first();
  const bounds=await screen.boundingBox({timeout:2000});
  if(!bounds||bounds.width<30||bounds.height<25)throw new Error('The dedicated WebSSH terminal has no viewport.');
  const m=terminal.page.mouse;
  await m.move(bounds.x+1,bounds.y+2); await m.down();
  try {
    await m.move(bounds.x+bounds.width-3,bounds.y+bounds.height-3,{steps:5});
  } finally {await m.up();}
  const captured=await terminal.input.evaluate(el=>{
    const data=new DataTransfer();
    el.dispatchEvent(new ClipboardEvent('copy',{clipboardData:data,bubbles:true,cancelable:true}));
    return data.getData('text/plain');
  });
  // A neutral click inside this page clears the selection before more output.
  await m.click(bounds.x+bounds.width-3,bounds.y+bounds.height-3);
  return captured;
}
const chunk=480;
function packet(marker,offset) {
  return "printf '\\n%s%s:%s:%s:%s:%s:%s!\\n' '"+marker.slice(0,5)+"' '"+marker.slice(5)+
    "' \"$seu_portal_rc\" \"${#seu_portal_data}\" \"$seu_portal_cut\" "+offset+
    ' "${seu_portal_data:'+offset+':'+chunk+'}"';
}
async function receive(terminal,marker,offset,timeout) {
  const pattern=new RegExp(marker+':(\\d+):(\\d+):([01]):'+offset+':([A-Za-z0-9+/=]*)!');
  const end=Date.now()+timeout;
  do {
    await pause(350);
    const screen=await readScreen(terminal);
    const found=pattern.exec(screen.replace(/\s/g,''));
    if(found)return {rc:Number(found[1]),len:Number(found[2]),cut:found[3]==='1',data:found[4]};
  }while(Date.now()<end);
  throw new Error('WebSSH did not return a complete response. Inspect the dedicated terminal and the job queue before retrying a submission.');
}
async function execute(context,action,selected) {
  // All shell input is constructed locally from known operations and validated arguments.
  const {shellCommand}=require('./workload.cjs');
  const command=shellCommand(action);
  const terminal=await getTerminal(context,selected);
  const marker='SEU'+crypto.randomBytes(8).toString('hex');
  const start='seu_portal_out=$( ( '+command+' ) 2>&1); seu_portal_rc=$?; '+
    'seu_portal_cut=0; (( ${#seu_portal_out} <= 24000 )) || seu_portal_cut=1; '+
    'seu_portal_data=$(printf %s "${seu_portal_out:0:24000}" | base64 -w0); unset seu_portal_out; ';
  await send(terminal,start+packet(marker,0));
  const first=await receive(terminal,marker,0,action.op==='sync'?53000:23000);
  if(first.len>140000)throw new Error('Invalid WebSSH response size.');
  let data=first.data;
  if(data.length!==Math.min(chunk,first.len))throw new Error('Incomplete first response block.');
  while(data.length<first.len) {
    const offset=data.length;
    await send(terminal,packet(marker,offset));
    const next=await receive(terminal,marker,offset,6000);
    if(next.len!==first.len || next.rc!==first.rc || next.data.length!==Math.min(chunk,first.len-offset))
      throw new Error('WebSSH response blocks changed during retrieval.');
    data+=next.data;
  }
  await send(terminal,'unset seu_portal_data seu_portal_rc seu_portal_cut');
  const output=Buffer.from(data,'base64').toString('utf8')+(first.cut?'\n[output capped at 24000 characters]':'');
  return {ok:first.rc===0,exitCode:first.rc,output};
}
module.exports={execute,packet};
