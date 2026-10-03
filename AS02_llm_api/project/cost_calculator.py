"""Time-only break-even calculation from explicit user inputs; not money estimates."""
import argparse
import json
import math
from pathlib import Path

FIELDS=['manual_seconds_per_posting','review_seconds_per_posting','correction_seconds_per_incorrect_posting','usable_without_correction_fraction','human_setup_seconds','blocked_wait_fraction']


def calculate(inputs, metrics):
    for key in FIELDS:
        if type(inputs.get(key)) not in (int,float) or not math.isfinite(inputs[key]) or inputs[key]<0:
            raise ValueError('A nonnegative finite user-supplied number is required for '+key)
    for key in ['usable_without_correction_fraction','blocked_wait_fraction']:
        if inputs[key]>1:raise ValueError(key+' must be between 0 and 1')
    api_human_seconds=(inputs['review_seconds_per_posting']+
        (1-inputs['usable_without_correction_fraction'])*inputs['correction_seconds_per_incorrect_posting']+
        inputs['blocked_wait_fraction']*metrics['mean_latency_seconds'])
    saved=inputs['manual_seconds_per_posting']-api_human_seconds
    return {'api_human_seconds_per_posting':api_human_seconds,'human_seconds_saved_per_posting':saved,
            'time_break_even_postings':math.ceil(inputs['human_setup_seconds']/saved) if saved>0 else None,
            'break_even_status':'finite under supplied assumptions' if saved>0 else 'no time saving under supplied assumptions',
            'measured_model_seconds_per_request':metrics['mean_latency_seconds'],
            'limitations':['Time-only estimate; electricity, hardware, money and organizational error consequences not priced.',
                           'Usability must come from human review, not schema-validity rate.',
                           'Assumes sequential waiting, review and correction; adjust blocked_wait_fraction for work done while waiting.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--metrics',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=calculate(json.loads(Path(a.inputs).read_text()),json.loads(Path(a.metrics).read_text()))
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
