"""Prepare and run private coursework experiments; never invent human scores."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

from career_api import CareerAPI, REQUEST, reference, norm

ROOT = Path(__file__).resolve().parent
REGIMES = {
    'B1': ('generate', 'development', 0.2, 0.8, 256),
    'B3': ('generate', 'development', 0.7, 0.95, 384),
    'C1': ('structured', 'evaluation', 0.2, 0.8, 256),
    'C2': ('tool', 'evaluation', 0.2, 0.8, 256),
    'C3': ('guardrail', 'evaluation', 0.2, 0.8, 256),
    'C4': ('verified', 'evaluation', 0.2, 0.8, 256),
}


def load_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    corpus = load_rows(args.corpus)
    if len({r['doc_id'] for r in corpus}) != len(corpus):
        raise ValueError('Duplicate corpus IDs')
    previous = {r['doc_id'] for r in load_rows(args.previous_labels)}
    if not previous <= {r['doc_id'] for r in corpus}:
        raise ValueError('Previous label IDs missing from corpus')
    # Previously inspected AS01 references stay in development, never evaluation.
    known = [r for r in corpus if r['doc_id'] in previous]
    unseen = [r for r in corpus if r['doc_id'] not in previous]
    rng = random.Random(820)
    rng.shuffle(known)
    rng.shuffle(unseen)
    dev_count = len(corpus) // 2
    if len(known) > dev_count or min(dev_count, len(corpus)-dev_count) < 50:
        raise ValueError('Need at least 50 disjoint records per split')
    dev = known + unseen[:dev_count-len(known)]
    evaluation = unseen[dev_count-len(known):]
    rng.shuffle(dev)
    # Fail on exact text duplicates across splits. Near duplicates remain a limitation.
    def text_hash(r):
        return hashlib.sha256(' '.join(r['raw_text'].split()).encode()).hexdigest()
    if {text_hash(r) for r in dev} & {text_hash(r) for r in evaluation}:
        raise ValueError('Exact duplicate descriptions cross splits')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out/'development.jsonl', dev)
    write_rows(out/'evaluation.jsonl', evaluation)
    write_rows(out/'source_references.jsonl', [reference(r) for r in corpus])
    manifest = {'seed': 820, 'corpus_sha256': sha(args.corpus),
                'previous_labels_sha256': sha(args.previous_labels),
                'development_ids': [r['doc_id'] for r in dev],
                'evaluation_ids': [r['doc_id'] for r in evaluation],
                'experiment_development_ids': [r['doc_id'] for r in dev[:50]],
                'experiment_evaluation_ids': [r['doc_id'] for r in evaluation[:50]],
                'previously_labeled_ids': sorted(previous),
                'split_hashes': {name: sha(out/name) for name in ['development.jsonl', 'evaluation.jsonl']},
                'reference_origin': 'original source table and corpus provenance, not human labels',
                'limitation': 'Shared employers and near-duplicate roles may span splits; not employer-held-out generalization.'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Prepared', len(dev), 'development and', len(evaluation), 'evaluation records.')


class VMBackend:
    def __init__(self, llmbox, model_path):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path.insert(0, str(Path(llmbox).resolve()))
        from omegaconf import OmegaConf
        from src.generation import GenerationManager
        import torch
        torch.set_num_threads(2)
        self.torch = torch
        self.OmegaConf = OmegaConf
        self.manager = GenerationManager()
        self.cfg = OmegaConf.create({'model': {
            'name': 'phi4_instruct', 'source': 'local', 'local_path': str(Path(model_path).resolve()),
            'model_id': 'microsoft/Phi-4-mini-instruct', 'device': 'cpu', 'dtype': 'bfloat16',
            'trust_remote_code': False, 'chat_template_kwargs': {}}})
        self.model, self.tokenizer, self.device = self.manager._load_model_and_tokenizer(self.cfg)
        self.source_sha256 = sha(Path(llmbox)/'src/generation.py')

    def __call__(self, messages, settings):
        self.torch.manual_seed(settings['seed'])
        self.cfg.generation = settings
        t = time.perf_counter()
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                   return_tensors='pt', add_generation_prompt=True).to(self.device)
        input_count = inputs['input_ids'].shape[-1]
        limit = getattr(self.model.config, 'max_position_embeddings', None)
        if limit and input_count + settings['max_new_tokens'] > limit:
            raise ValueError('Context length would be exceeded; input not silently truncated')
        kwargs = self.manager._generation_kwargs(self.cfg, self.tokenizer)
        if not settings['do_sample']:
            for key in ['temperature', 'top_p', 'top_k']:
                kwargs.pop(key, None)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, **kwargs)
        new_ids = ids[0, input_count:]
        return {'text': self.tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                'input_tokens': int(input_count), 'output_tokens': len(new_ids),
                'output_cap_reached': len(new_ids) >= settings['max_new_tokens'],
                'model_seconds': time.perf_counter()-t}


class MLXBackend:
    """Optional Mac runner. Results must be labeled 4-bit MLX, not VM results."""
    def __init__(self, llmbox, model_path):
        os.environ['HF_HUB_OFFLINE'] = '1'
        from mlx_lm import load
        self.model, self.tokenizer = load(model_path)
        self.source_sha256 = None

    def __call__(self, messages, settings):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        mx.random.seed(settings['seed'])
        start = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=settings['temperature'] if settings['do_sample'] else 0,
                               top_p=settings['top_p'], top_k=settings['top_k'])
        text, last = '', None
        for item in stream_generate(self.model, self.tokenizer, prompt=prompt, sampler=sampler,
                                    max_tokens=settings['max_new_tokens']):
            text += item.text
            last = item
        if last is None:
            raise RuntimeError('No model output')
        return {'text': text.strip(), 'input_tokens': last.prompt_tokens,
                'output_tokens': last.generation_tokens,
                'output_cap_reached': last.generation_tokens >= settings['max_new_tokens'],
                'model_seconds': time.perf_counter()-start}


def score(rows, corpus):
    by_id = {r['doc_id']: r for r in corpus}
    keys = ['doc_id', 'source_url', 'job_title', 'employer', 'location']
    per_field = {k: 0 for k in keys}
    for result in rows:
        gold = reference(by_id[result['doc_id']])
        for k in keys:
            if not result['blocked'] and result['facts'] and norm(result['facts'][k]) == norm(gold[k]):
                per_field[k] += 1
    return {'count': len(rows), 'schema_valid_count': sum(r['schema_valid'] for r in rows),
            'blocked_count': sum(r['blocked'] for r in rows),
            'delivered_field_accuracy': {k: v/len(rows) for k, v in per_field.items()},
            'delivered_all_source_fields_correct_rate': sum(
                bool(r['facts']) and not r['blocked'] and all(norm(r['facts'][k]) == norm(reference(by_id[r['doc_id']])[k]) for k in keys)
                for r in rows)/len(rows),
            'successful_tool_calls': sum(any(e.get('ok') for e in r['tool_events']) for r in rows),
            'output_cap_reached_calls': sum(c['output_cap_reached'] for r in rows for c in r['calls']),
            'mean_latency_seconds': statistics.mean(r['latency_seconds'] for r in rows),
            'mean_input_tokens': statistics.mean(r['input_tokens'] for r in rows),
            'mean_output_tokens': statistics.mean(r['output_tokens'] for r in rows),
            'qualification_semantic_accuracy': None, 'work_arrangement_accuracy': None,
            'human_review_status': 'Pending; source string checks are not human judgments'}


def run(args):
    data = Path(args.data)
    split_manifest = json.loads((data/'manifest.json').read_text())
    for name, expected_hash in split_manifest['split_hashes'].items():
        if sha(data/name) != expected_hash:
            raise ValueError('Split changed after preparation: ' + name)
    dev = load_rows(data/'development.jsonl')
    evaluation = load_rows(data/'evaluation.jsonl')
    if {r['doc_id'] for r in dev} & {r['doc_id'] for r in evaluation}:
        raise ValueError('Overlapping splits')
    mode, split, temp, top_p, cap = REGIMES[args.regime]
    dataset = dev if split == 'development' or args.limit == 1 else evaluation
    selected = dataset[:args.limit]
    if len(selected) != args.limit:
        raise ValueError('Insufficient inputs')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    settings = {'temperature': temp, 'top_p': top_p, 'max_new_tokens': cap,
                'do_sample': True, 'top_k': 0, 'repetition_penalty': 1.0, 'seed': 820}
    manifest = {'status': 'running', 'regime': args.regime, 'mode': mode, 'settings': settings,
                'backend': args.backend, 'model_path': args.model, 'split_manifest_sha256': sha(data/'manifest.json'),
                'input_ids': [r['doc_id'] for r in selected], 'is_smoke_test': args.limit != 50,
                'effective_split': 'development' if args.limit == 1 else split,
                'code_hashes': {p.name: sha(p) for p in ROOT.glob('*.py')},
                'packages': {p: importlib.metadata.version(p) for p in ['transformers', 'pydantic']}}
    output_manifest = out/'manifest.json'
    output_manifest.write_text(json.dumps(manifest, indent=2))
    results = []
    try:
        start = time.perf_counter()
        backend = (VMBackend if args.backend == 'vm' else MLXBackend)(args.llmbox, args.model)
        manifest['model_load_seconds'] = time.perf_counter()-start
        manifest['llmbox_generation_sha256'] = backend.source_sha256
        api = CareerAPI(backend, dev+evaluation)
        with (out/'responses.jsonl').open('x') as f:
            for i, record in enumerate(selected):
                case_settings = {**settings, 'seed': 820+i}
                result = api.answer(record['doc_id'], mode, case_settings)
                result['settings'] = case_settings
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
                f.flush()
                results.append(result)
                print(f'{args.regime}: {i+1}/{len(selected)} saved; {result["latency_seconds"]:.1f}s', flush=True)
        (out/'metrics.json').write_text(json.dumps(score(results, dataset), indent=2))
        review = [{'doc_id': r['doc_id'], 'response': r['response'], 'source': source,
                   'work_arrangement_correct': None, 'qualifications_correct': None,
                   'usable_without_correction': None, 'review_seconds': None,
                   'correction_seconds': None, 'notes': ''}
                  for r, source in zip(results, selected)]
        write_rows(out/'human_review_PENDING.jsonl', review)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        manifest['completed_count'] = len(results)
        output_manifest.write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--corpus', required=True)
    p.add_argument('--previous-labels', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser('run')
    p.add_argument('--data', required=True)
    p.add_argument('--regime', choices=REGIMES, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, choices=[1, 50], default=50)
    p.add_argument('--backend', choices=['vm', 'mlx'], default='vm')
    p.add_argument('--llmbox', default='../llmbox')
    p.add_argument('--model', default='/opt/95820-models/microsoft/Phi-4-mini-instruct')
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
