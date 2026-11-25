'use strict';
const {validated}=require('./commands.cjs');

const repo = 'cd -- "$HOME/casl/cognitive-ultrasound"';
const root = 'root="${CASL_SEU_ROOT:-$HOME/casl}"';
const jobFormat = '--format=JobIDRaw,JobName%48,Partition,State%24,ExitCode,Elapsed,Start,End';

function shellCommand(request) {
  const action=validated(request);
  switch(action.op) {
    case 'probe':
      return [
        'printf "node=%s user=%s\\n" "$(hostname -s)" "$(id -un)";',
        'command -v squeue && { command -v sbatch_wrapper || test -x /usr/bin/sbatch_wrapper; } &&',
        repo+' && git rev-parse --short HEAD && test -r scripts/run_seu_casl.sh && printf "SEU_PORTAL_READY\\n"',
      ].join(' ');
    case 'queue':
      // This site's interactive squeue wrapper supplies its own column layout.
      return 'set -o pipefail; squeue -u "$(id -un)" | awk \'NR<=101{print}\'';
    case 'history':
      return [
        root+';',
        'start=$(date -d "'+action.days+' days ago" +%F); printf "SLURM_ACCOUNTING\\n";',
        'sacct -X -u "$(id -un)" -S "$start" -P '+jobFormat+' | tail -n 80;',
        'printf "\\nRECENT_CASL_LOGS (newest first)\\n";',
        'find "$root/jobs" -maxdepth 3 -type f -user "$(id -un)" -mtime -'+action.days+
          ' -name "casl-*.log" -printf "%T@ %f\\n" | sort -rn | awk \'NR<=60 {print $2}\'',
      ].join(' ');
    case 'status':
      return [
        'set -o pipefail; me=$(id -un);',
        'record=$(scontrol show job -o '+action.id+' 2>/dev/null) || record=;',
        'if [[ $record == *"UserId=$me("* ]]; then',
        'printf "%s\\n" "$record" | tr " " "\\n" | awk \'/^(JobId|JobName|JobState|Reason|ExitCode|RunTime|Partition|StdOut|StdErr)=/{print}\';',
        'else rows=$(sacct -X -u "$me" -nP -j '+action.id+' '+jobFormat+') || exit $?;',
        'if [[ -z $rows ]]; then printf "No live or accounting record for this job under %s.\\n" "$me" >&2; exit 1; fi;',
        'printf "%s\\n%s\\n" "JobID|JobName|Partition|State|ExitCode|Elapsed|Start|End" "$rows"; fi',
      ].join(' ');
    case 'sync':
      return repo+' && export GIT_TERMINAL_PROMPT=0 && timeout 45 git pull --ff-only origin main && git rev-parse --short HEAD';
    case 'gpu-report':
      return [
        root+'; base=$(realpath -e -- "$root/jobs/acceptance") || exit 1;',
        'file="$base/gpu-'+action.id+'/gpu.json";',
        'if [[ ! -f $file ]]; then printf "No saved gpu.json for job '+action.id+'. Read its Slurm log for earlier failures.\\n" >&2; exit 1; fi;',
        'actual=$(realpath -e -- "$file") || exit 1;',
        '[[ $actual == "$file" && -O $actual ]] || exit 1; head -c 16000 -- "$actual"',
      ].join(' ');
    case 'submit':
      return repo+' && bash scripts/run_seu_casl.sh submit '+action.task+(action.partition?' '+action.partition:'');
    case 'logs':
      return [
        root+'; me=$(id -un); record=$(scontrol show job -o '+action.id+' 2>/dev/null) || record=;',
        'file=; if [[ $record == *"UserId=$me("* ]]; then',
        'file=$(printf "%s\\n" "$record" | sed -n \'s/.* StdOut=\\([^ ]*\\).*/\\1/p\'); fi;',
        'if [[ -z $file || ! -f $file ]]; then',
        'mapfile -t files < <(find "$root/jobs" -maxdepth 3 -user "$me" -type f -name "casl-*-'+action.id+'.log");',
        'if (( ${#files[@]} != 1 )); then printf "No unique existing CASL log for job '+action.id+'. It may still be pending.\\n" >&2; exit 1; fi;',
        'file="${files[0]}"; fi;',
        'base=$(realpath -e -- "$root/jobs") && target=$(realpath -e -- "$file") || exit 1;',
        'if [[ $target != "$base"/* || ! -O $target ]]; then printf "Log outside owned CASL job output.\\n" >&2; exit 1; fi;',
        'tail -n '+action.lines+' -- "$target"',
      ].join(' ');
    default: throw new Error('Unsupported portal operation.');
  }
}

module.exports={shellCommand};
