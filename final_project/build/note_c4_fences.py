"""Oct 3: clarify Pharma C4 vs Tech C4 (verified from runs/policy_first_gpu/C1 and C4 raw outputs)."""
import json
from pathlib import Path
f = Path(__file__).with_name('report_content.json')
p = json.loads(f.read_text())
b = p[4]['blocks'][4]
old = 'C4 withholds all 50 answers, leaving no useful coverage.'
add = (' All 50 withholdings were schema failures: C4 received the same fenced model outputs as C1 (50/50 '
       'identical). Removing the complete fence before validation, as the Tech parser does, 31 of 50 would have '
       'passed every C4 check; this is a post-hoc diagnostic, not a delivered result.')
if add not in b['text']:
    assert b['text'].startswith(old)
    b['text'] = b['text'].replace(old, old + add, 1)
b = p[5]['blocks'][5]
old = 'Pharma C4 withheld every answer that failed verification and delivered nothing.'
new = ('Pharma C4 withheld every answer that failed verification and delivered nothing, but its failures were '
       'formatting, not content: with complete fences removed, 31 of 50 Pharma answers would have passed its '
       'stricter checks (all five source facts and verbatim quotes). Most of the 0 versus 39 gap is therefore '
       'the parser; the remaining difference is repair versus withholding, measured on different scores.')
if new not in b['text']:
    assert old in b['text']
    b['text'] = b['text'].replace(old, new, 1)
f.write_text(json.dumps(p, ensure_ascii=False, indent=1))
print('report_content.json updated')
