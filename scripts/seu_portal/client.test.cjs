'use strict';
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {parseAction, validated} = require('./commands.cjs');
const {shellCommand} = require('./workload.cjs');
const {runAction} = require('./client.cjs');

const bash = process.env.GIT_BASH_EXE || (process.platform === 'win32'
  ? ['G:/Git/bin/bash.exe', 'C:/Program Files/Git/bin/bash.exe'].find(p => fs.existsSync(p))
  : 'bash');

test('the RPC and CLI accept the same argument bounds', () => {
  for (const args of [['status', '123456'], ['logs','17','200'], ['history'], ['history','90'],
    ['submit','gpu-smoke','gpu-L20'], ['gpu-report','17'], ['session'], ['reconnect']]) {
    const expected = parseAction(args);
    assert.deepEqual(validated(JSON.parse(JSON.stringify(expected))), expected);
  }
  for (const args of [['status','17;id'], ['logs','12','0'], ['logs','12','201'], ['history','91'],
    ['submit','foo\nid'], ['submit','cpu-check','--wrap=x'], ['sync','HEAD'], ['exec','whoami']]) {
    assert.throws(() => parseAction(args));
  }
  assert.throws(() => validated({op:'probe',command:'unexpected'}));
});

test('generated commands are valid Bash, including the WebSSH capture wrapper', {skip: !bash}, () => {
  for (const args of [['probe'], ['queue'], ['history','7'], ['status','17'], ['logs','17','1'],
    ['gpu-report','17'], ['sync'], ['submit','cpu-check']]) {
    const command = shellCommand(parseAction(args));
    const result = spawnSync(bash, ['-n'], {input:'out=$( ( '+command+' ) 2>&1); rc=$?\n',encoding:'utf8'});
    assert.equal(result.status, 0, result.stderr || String(result.error));
  }
});

function runFixture(action, setup) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'seu-portal-test-'));
  try {
    const script = [
      'export PATH=/usr/bin:/bin:$PATH',
      'export CASL_SEU_ROOT="$PWD/casl"',
      'mkdir -p "$CASL_SEU_ROOT/jobs/acceptance"',
      setup,
      shellCommand(parseAction(action)),
      'exit $?',
    ].join('\n');
    return spawnSync(bash, ['--noprofile','--norc'], {input:script,cwd:dir,encoding:'utf8'});
  } finally { fs.rmSync(dir,{recursive:true,force:true}); }
}

test('a pending job is a successful query; historical failures preserve the exit code', {skip: !bash}, () => {
  const pending = runFixture(['status','11'], 'scontrol() { echo "JobId=11 UserId=$(id -un)(1000) JobState=PENDING ExitCode=0:0 Reason=Resources"; }');
  assert.equal(pending.status,0,pending.stderr);
  assert.match(pending.stdout,/JobState=PENDING/);
  const historical = runFixture(['status','12'], 'scontrol() { return 1; }; sacct() { echo "12|casl-gpu-smoke|gpu-L20|FAILED|127:0|00:00:00|||"; }');
  assert.equal(historical.status,0,historical.stderr);
  assert.match(historical.stdout,/FAILED\|127:0/);
  const absent = runFixture(['status','13'], 'scontrol() { return 1; }; sacct() { :; }');
  assert.notEqual(absent.status,0);
});

test('read log by job ID after accounting expiry, returning exactly the requested tail', {skip: !bash}, () => {
  const result = runFixture(['logs','12','2'], [
    'scontrol() { return 1; }',
    'printf "old\\nmiddle\\nlast\\n" > "$CASL_SEU_ROOT/jobs/acceptance/casl-gpu-smoke-12.log"',
  ].join('\n'));
  assert.equal(result.status,0,result.stderr);
  assert.equal(result.stdout.trim(),'middle\nlast');
});

test('a live job cannot turn its stdout path into an arbitrary file read', {skip: !bash}, () => {
  const result = runFixture(['logs','17'], [
    'printf "outside" > "$PWD/other"',
    'scontrol() { echo "JobId=17 UserId=$(id -un)(1000) StdOut=$PWD/other"; }',
  ].join('\n'));
  assert.notEqual(result.status,0);
  assert.match(result.stderr,/Log outside owned CASL job output/);
});

test('a read can reconnect once; an uncertain submission is never sent again', async () => {
  for (const op of ['status', 'submit', 'sync']) {
    const calls = [];
    const result = await runAction({op}, async action => {
      calls.push(action.op);
      return calls.length === 1 ? {ok:false,error:'WebSSH did not return a complete response.'} : {ok:true};
    });
    assert.deepEqual(calls, op === 'status' ? ['status','reconnect','status'] : [op]);
    assert.equal(result.ok, op === 'status');
  }
});

test('GPU job reaches its checks when the compute node has no Git', {skip: !bash}, () => {
  const working = fs.mkdtempSync(path.join(os.tmpdir(), 'seu-no-git-'));
  try {
    const sharedTools = path.join(working,'shared-tools');
    fs.mkdirSync(sharedTools);
    fs.writeFileSync(path.join(sharedTools,'git'),'#!/bin/bash\necho FIXTURE_SHARED_GIT\n',{mode:0o755});
    const mock = [
      'set -e',
      'export PATH=/usr/bin:/bin:$PATH',
      'export CASL_SEU_ROOT="$PWD/work" CASL_SEU_NVME_ROOT="$PWD/nvme" SLURM_JOB_ID=123',
      'mkdir -p env/etc/profile.d',
      ': > env/etc/profile.d/conda.sh',
      'mkdir_cmd=$(command -v mkdir)',
      'mkdir() { "$mkdir_cmd" "$@"; }',
      'hostname() { echo fixture-node; }',
      'module() { :; }',
      'conda() { if [[ $1 == info ]]; then printf "%s/env\\n" "$PWD"; fi; }',
      'nvidia-smi() { printf "FIXTURE_NVIDIA_SMI LD=%s\\nXLA=%s\\nPATH=%s\\n" "$LD_LIBRARY_PATH" "$XLA_FLAGS" "$PATH"; }',
      'python() { printf "FIXTURE_PYTHON %s\\n" "$*"; }',
      'PATH=/fixture-without-git; hash -r',
      'source "$SEU_FIXTURE_JOB" gpu-smoke "$PWD" abcdef /fixture/cudnn /fixture/toolkit "$PWD/shared-tools/git"',
    ].join('\n');
    const job = path.resolve(__dirname, '..', 'seu_casl_job.sh').replaceAll('\\','/');
    const result = spawnSync(bash, ['--noprofile','--norc'], {
      input:mock,cwd:working,encoding:'utf8',env:{...process.env,SEU_FIXTURE_JOB:job},
    });
    assert.equal(result.status,0,result.stderr);
    assert.match(result.stdout,/submitted_repo_rev=abcdef/);
    assert.match(result.stdout,/FIXTURE_SHARED_GIT/);
    assert.match(result.stdout,/FIXTURE_NVIDIA_SMI LD=\/fixture\/cudnn:\/fixture\/toolkit\/lib64/);
    assert.match(result.stdout,/--xla_gpu_cuda_data_dir=\/fixture\/toolkit/);
    assert.match(result.stdout,/PATH=\/fixture\/toolkit\/bin:/);
    assert.match(result.stdout,/FIXTURE_PYTHON scripts\/check_gpu.py/);
    assert.match(result.stdout,/FIXTURE_PYTHON -m cognitive_ultrasound evaluate/);
    assert.match(result.stdout,/FIXTURE_PYTHON scripts\/seu_casl_check.py gpu/);
  } finally { fs.rmSync(working, {recursive:true,force:true}); }
});

test('the login-node launcher passes the submission revisions to the batch worker', {skip: !bash}, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'seu-submit-fixture-'));
  try {
    const shims = path.join(dir,'shims');
    fs.mkdirSync(shims);
    fs.writeFileSync(path.join(shims,'sbatch_wrapper'), '#!/bin/bash\nprintf "%s\\n" "$@"\n', {mode:0o755});
    const envBin = path.join(dir,'env','bin');
    fs.mkdirSync(envBin,{recursive:true});
    fs.writeFileSync(path.join(envBin,'python'),'#!/bin/bash\necho /fixture/cudnn\n',{mode:0o755});
    for (const item of ['bin/nvcc','bin/ptxas','bin/nvlink','nvvm/libdevice/libdevice.10.bc','lib64/libcudart.so.12']) {
      const file = path.join(dir,'cuda',item);
      fs.mkdirSync(path.dirname(file),{recursive:true});
      fs.writeFileSync(file,item.startsWith('bin/') ? '#!/bin/bash\nexit 0\n' : 'fixture',{mode:0o755});
    }
    fs.writeFileSync(path.join(shims,'git'),'#!/bin/bash\necho git-fixture\n',{mode:0o755});
    const launcher = path.resolve(__dirname,'..','run_seu_casl.sh').replaceAll('\\','/');
    for (const task of ['gpu-smoke','cpu-check','bf-core-after-42','bf-core-after-43','accel-core-test-after-42']) {
      const script = [
        'export PATH="$PWD/shims:/usr/bin:/bin:$PATH" CASL_SEU_ROOT="$PWD/casl" CASL_SEU_ENV="$PWD/env"',
        'export CASL_SEU_CUDA_ROOT="$PWD/cuda"',
        'export CASL_SEU_GIT_BIN="$PWD/shims/git"',
        'git() { case "$2" in */vendor/casl/zea) echo zea-rev;; */vendor/casl) echo casl-rev;; *) echo repo-rev;; esac; }',
        'sinfo() { echo gpu-L20; }',
        'mkdir() { :; }',
        'id() { echo fixture-user; }',
        'sacct() { case "$SEU_TEST_TASK" in bf-core-after-42) echo "42|casl-bf-env|PENDING";; bf-core-after-43) echo "43|casl-bf-env|FAILED";; accel-core-test-after-42) echo "42|casl-bf-core|RUNNING";; esac; }',
        'compgen() { [[ $2 == */seqdiff-core-*/vivit/complete.json ]]; }',
        'source "$SEU_TEST_SUBMIT" submit "$SEU_TEST_TASK"',
      ].join('\n');
      const result = spawnSync(bash,['--noprofile','--norc'], {cwd:dir,input:script,encoding:'utf8',
        env:{...process.env,SEU_TEST_TASK:task,SEU_TEST_SUBMIT:launcher}});
      if (task.endsWith('-43')) {
        assert.equal(result.status,2,result.stderr);
        assert.match(result.stderr,/No active or successful casl-bf-env job/);
        continue;
      }
      assert.equal(result.status,0,result.stderr);
      const lines = result.stdout.trim().split(/\r?\n/);
      const batchTask = task.replace(/-after-42$/, '');
      assert.ok(lines.includes(batchTask));
      if (task.endsWith('-42')) {
        assert.ok(lines.includes('--dependency=afterok:42'));
        assert.ok(lines.includes('--kill-on-invalid-dep=yes'));
        assert.ok(lines.includes('--job-name=casl-' + batchTask));
      }
      if (task !== 'cpu-check') {
        assert.ok(lines.includes('--gres=gpu:1'));
        assert.deepEqual(lines.slice(-4,-2),['repo-rev','/fixture/cudnn']);
        assert.match(lines.at(-2),/\/cuda$/);
        assert.match(lines.at(-1),/\/shims\/git$/);
      } else {
        assert.deepEqual(lines.slice(-3),['repo-rev','casl-rev','zea-rev']);
      }
    }
  } finally { fs.rmSync(dir,{recursive:true,force:true}); }
});
