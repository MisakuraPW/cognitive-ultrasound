// Local control of one dedicated SEU browser. No TCP listener or remote agent.
'use strict';
const crypto = require('node:crypto');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');

const stateDir = path.join(process.env.LOCALAPPDATA || os.tmpdir(), 'SEU-CASL', 'WebSSH');
const stateFile = path.join(stateDir, 'session.json');
const endpoint = '\\\\.\\pipe\\seu-casl-' + crypto.createHash('sha256').update(os.userInfo().username + ':' + stateDir).digest('hex').slice(0,24);
const { parseAction, usage } = require('./commands.cjs');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

function readState() {
  try { return JSON.parse(fs.readFileSync(stateFile, 'utf8')); }
  catch { return null; }
}
function call(request, timeout = 180000) {
  const state = readState();
  if (!state) return Promise.reject(new Error('NO_SESSION'));
  return new Promise((resolve, reject) => {
    const socket = net.createConnection(endpoint);
    const timer = setTimeout(() => socket.destroy(new Error('Local request timed out.')), timeout);
    socket.setEncoding('utf8');
    let reply = '';
    socket.on('connect', () => socket.write(JSON.stringify({token:state.token, request}) + '\n'));
    socket.on('data', chunk => {
      reply += chunk;
      if (reply.length > 600000) socket.destroy(new Error('Oversized local response.'));
    });
    socket.on('error', reject);
    socket.on('close', () => {
      clearTimeout(timer);
      try { resolve(JSON.parse(reply)); }
      catch { reject(new Error('NO_SESSION')); }
    });
  });
}

async function main(argv) {
  const action = parseAction(argv);
  if (action.op === 'help') {
    console.log('node scripts/seu_portal/client.cjs ' + usage);
    return;
  }
  if (action.op === 'login') {
    try {
      const current = await call({op:'session'}, 2000);
      if (current.ok) return current;
      throw new Error(current.error || 'Existing session could not be verified.');
    } catch (error) {
      if (!['ENOENT','ECONNREFUSED','NO_SESSION'].includes(error.code || error.message)) throw error;
    }
    fs.mkdirSync(stateDir,{recursive:true,mode:0o700});
    const log = fs.openSync(path.join(stateDir,'browser-start.log'),'a',0o600);
    const child = spawn(process.execPath,[path.join(__dirname,'host.cjs')],{
      detached:true, windowsHide:true, stdio:['ignore',log,log],
    });
    child.unref();
    fs.closeSync(log);
    const deadline = Date.now() + 35000;
    while (Date.now() < deadline) {
      await pause(350);
      try { return await call({op:'session'}, 3000); }
      catch (error) {
        if (!['ENOENT','ECONNREFUSED','NO_SESSION'].includes(error.code || error.message)) throw error;
      }
    }
    throw new Error('The dedicated browser did not start. Check ' + path.join(stateDir,'browser-start.log'));
  }
  return runAction(action, call);
}

async function callWhenIdle(action, invoke) {
  const waitUntil=Date.now()+15000;
  let answer;
  do {
    answer=await invoke(action);
    if(answer.code!=='BUSY') return answer;
    await pause(500);
  } while (Date.now()<waitUntil);
  return answer;
}

async function runAction(action, invoke) {
  const result = await callWhenIdle(action, invoke);
  const readOnly = ['probe', 'queue', 'status', 'logs', 'history', 'gpu-report'].includes(action.op);
  const disconnected = /WebSSH did not return|no viewport|response blocks changed|Incomplete first response block/i;
  if (result.ok || !readOnly || !disconnected.test(result.error || '')) return result;
  const live = await callWhenIdle({op:'reconnect'}, invoke);
  if (!live.ok || action.op === 'probe') return live;
  return callWhenIdle(action, invoke);
}

module.exports = {call, main, runAction, stateDir, stateFile, endpoint};
if (require.main === module) {
  main(process.argv.slice(2)).then(result => {
    if (!result) return;
    if (result.output) process.stdout.write(result.output.endsWith('\n') ? result.output : result.output+'\n');
    else console.log(JSON.stringify(result, null, 2));
    if (!result.ok) process.exitCode=1;
  }).catch(error => {
    const unavailable=['ENOENT','ECONNREFUSED','NO_SESSION'].includes(error.code||error.message);
    console.error(unavailable?'The dedicated WebSSH session is closed or not visible under this Windows account. Run the login command.':error.message);
    process.exitCode=1;
  });
}
