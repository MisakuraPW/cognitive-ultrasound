'use strict';

const simple = new Set(['help', 'login', 'session', 'probe', 'queue', 'sync', 'reconnect', 'close']);
const usage = 'login | session | probe | queue | history [DAYS] | status JOB | logs JOB [LINES] | gpu-report JOB | sync | submit TASK [PARTITION] | reconnect | close';

function numberArg(value, defaultValue, maximum, name) {
  if (value === undefined) return defaultValue;
  if (typeof value !== 'string' || !/^[1-9]\d*$/.test(value) || Number(value) > maximum) {
    throw new Error(name + ' must be between 1 and ' + maximum + '.');
  }
  return Number(value);
}

function parseAction(argv) {
  const [op='help', ...args] = argv;
  if (simple.has(op) && args.length === 0) return {op};
  if (op === 'history' && args.length <= 1) {
    return {op, days: numberArg(args[0],14,90,'Days')};
  }
  if (['status','logs','gpu-report'].includes(op)) {
    const [id, limit] = args;
    if (typeof id !== 'string' || !/^[1-9]\d{0,17}$/.test(id)) throw new Error('A numeric Slurm job ID is required.');
    if (op !== 'logs' && args.length === 1) return {op,id};
    if (op === 'logs' && args.length <= 2) return {op,id,lines:numberArg(limit,50,200,'Lines')};
  }
  if (op === 'submit' && [1,2].includes(args.length)) {
    const [task,partition] = args;
    // The remote, version-controlled launcher decides which tasks actually exist.
    if (typeof task !== 'string' || !/^[a-z][a-z0-9-]{0,47}$/.test(task)) throw new Error('Invalid task name.');
    if (partition !== undefined && (typeof partition !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,47}$/.test(partition))) {
      throw new Error('Invalid partition.');
    }
    return {op,task,partition};
  }
  throw new Error('Invalid command or arguments. Usage: ' + usage);
}

function validated(request) {
  if (!request || typeof request !== 'object' || typeof request.op !== 'string') throw new Error('Invalid request.');
  const fields = {
    history: ['days'], status: ['id'], logs: ['id','lines'], 'gpu-report': ['id'], submit: ['task','partition'],
  }[request.op] || [];
  if (Object.keys(request).some(key => key!=='op' && !fields.includes(key))) throw new Error('Unexpected request field.');
  const args = fields.filter(key => request[key] !== undefined).map(key => String(request[key]));
  return parseAction([request.op,...args]);
}

module.exports = {parseAction,validated,usage};
