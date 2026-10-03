"""Retry the unchanged frozen protocol with native GPU access; preserve attempt 1."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=R/'governance/PRE_RUN_LOCK.json';frozen=json.loads(lock.read_text())
assert all(sha(R/n)==h for n,h in frozen['files_sha256'].items())
status=R/'ATTEMPT_2_STATUS.json'
if status.exists():raise SystemExit('Refusing to overwrite previous attempt')
event={'policy_sha256':sha(R/'governance/POLICY_V1.md'),'freeze_lock_sha256':sha(lock),'runner_sha256':sha(Path(__file__)),'started_at_utc':now(),'status':'running','reason':'Attempt 1 failed before inference because the sandbox did not expose a Metal GPU. Same policy, data, code, settings and seeds.','run_output':'runs/policy_first_gpu'}
status.write_text(json.dumps(event,indent=2))
ret=subprocess.run([sys.executable,'-u',str(R/'project/run_llmbox.py'),'--model',str(R.parent/'.models/phi-4-mini-instruct-4bit'),'--out',str(R/event['run_output'])],cwd=R/'project')
changed=[n for n,h in frozen['files_sha256'].items() if sha(R/n)!=h]
event.update(ended_at_utc=now(),exit_code=ret.returncode,changed_frozen_files=changed,status='complete' if ret.returncode==0 and not changed else 'failed')
status.write_text(json.dumps(event,indent=2)+'\n');print(json.dumps(event,indent=2),flush=True)
raise SystemExit(0 if event['status']=='complete' else 1)
