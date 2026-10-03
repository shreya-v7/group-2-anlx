"""Freeze policy and inputs before launching uncached model inference."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=R/'governance/PRE_RUN_LOCK.json'
if lock.exists():raise SystemExit('Existing freeze lock: refusing to overwrite chronology')
paths=[R/'governance/POLICY_V1.md',Path(__file__)]
paths += [p for folder in ['project','llmbox/src'] for p in (R/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.json'}]
frozen={str(p.relative_to(R)):sha(p) for p in sorted(paths)}
record={'frozen_at_utc':now(),'purpose':'New prospective policy-first rerun after historical experiments; not a claim to erase prior chronology','files_sha256':frozen,'run_output':'runs/policy_first','status':'frozen_before_subprocess'}
lock.write_text(json.dumps(record,indent=2)+'\n')
model=R.parent/'.models/phi-4-mini-instruct-4bit'
event={'policy_sha256':sha(R/'governance/POLICY_V1.md'),'freeze_lock_sha256':sha(lock),'launched_at_utc':now(),'status':'running'}
(R/'RUN_STATUS.json').write_text(json.dumps(event,indent=2))
print('Policy frozen:',record['frozen_at_utc'],flush=True)
print('Policy SHA256:',event['policy_sha256'],flush=True)
result=subprocess.run([sys.executable,'-u',str(R/'project/run_llmbox.py'),'--model',str(model),'--out',str(R/'runs/policy_first')],cwd=R/'project')
changed=[name for name,h in frozen.items() if sha(R/name)!=h]
event.update(ended_at_utc=now(),exit_code=result.returncode,changed_frozen_files=changed,status='complete' if result.returncode==0 and not changed else 'failed')
(R/'RUN_STATUS.json').write_text(json.dumps(event,indent=2)+'\n')
print(json.dumps(event,indent=2),flush=True)
raise SystemExit(0 if event['status']=='complete' else 1)
