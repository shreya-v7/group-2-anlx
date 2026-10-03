"""Actual LLMBox Modes with a documented MLX generation backend extension.

Upstream mode/prompt/dispatch implementations remain unmodified. No cached model
outputs are used. This is local 4-bit inference, not the full-precision VM model.
"""
import contextlib
import io
import json
from pathlib import Path
import re
import sys
import time

LLMBOX = Path(__file__).resolve().parents[1] / 'llmbox'
sys.path.insert(0, str(LLMBOX))
from omegaconf import OmegaConf
from src.generation import GenerationManager
from src.modes import Modes
from src.tools import TOOL_REGISTRY
from career_api import SYSTEM, REQUEST, REFUSAL, JobFacts, source_view, risk_flags, parse_facts, factual_errors
from experiments import MLXBackend


class MLXGenerationManager(GenerationManager):
    def __init__(self, model_path=None, backend=None):
        super().__init__()
        self.backend = backend if backend is not None else MLXBackend(None, model_path)
        self.calls = []
        self.tool_events = []

    def _load_model_and_tokenizer(self, cfg):
        # Reuse one loaded model across requests, without conversational state.
        return getattr(self.backend, 'model', None), getattr(self.backend, 'tokenizer', None), 'mlx'

    def _generate_once(self, model, tokenizer, device, messages, cfg, **template_kwargs):
        if template_kwargs:
            raise ValueError('This Phi backend requires tools embedded by LLMBox, not generic template kwargs')
        if messages and messages[-1]['role'] == 'tool':
            # Custom two-stage workflow: give the extraction schema only after
            # lookup, so the routing turn need not satisfy two competing schemas.
            messages.append({'role':'user','content':SYSTEM+'\nRequired JSON Schema:\n'+Path(cfg.structured_output.schema_path).read_text()})
        settings = OmegaConf.to_container(cfg.generation, resolve=True)
        settings['seed'] = cfg.seed
        answer = self.backend(messages, settings)
        self.calls.append({'messages': json.loads(json.dumps(messages)), 'settings': settings, **answer})
        return answer['text']

    def parse_tool_calls(self, text):
        # Use the upstream parser only for an explicit leading marker, then
        # enforce a one-tool, zero-argument whitelist before dispatch. Upstream
        # tolerates trailing model prose; it remains logged, not trusted data.
        # Also accept a single complete fenced JSON response.
        raw = text.strip()
        fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', raw, re.S)
        if fenced:
            raw = fenced.group(1).strip()
        if raw.startswith('functools'):
            calls = super().parse_tool_calls(raw)
        else:
            try:
                calls = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        call = calls[0]
        if not isinstance(call, dict) or set(call) != {'name', 'arguments'}:
            return None
        if call['name'] != 'lookup_selected_posting' or call['arguments'] != {}:
            return None
        return calls

    def _dispatch_tool_calls_and_continue(self, *args, **kwargs):
        # Execute the actual upstream dispatcher, with a per-request registry.
        answer, events = super()._dispatch_tool_calls_and_continue(*args, **kwargs)
        self.tool_events.extend({**event, 'ok': not ('error' in event['result'])} for event in events)
        return answer, events


class LLMBoxCareerAPI:
    def __init__(self, records, model_path=None, backend=None):
        self.records = {r['doc_id']: r for r in records}
        if len(self.records) != len(records):
            raise ValueError('Duplicate corpus IDs')
        self.generator = MLXGenerationManager(model_path, backend)
        self.modes = Modes()
        self.modes.generator = self.generator
        self.schema_path = Path(__file__).with_name('job_facts.schema.json')
        if not self.schema_path.exists():
            raise ValueError('Run prepare_schema before starting experiments')

    def answer(self, doc_id, mode, settings, request=REQUEST, record_override=None):
        if mode not in {'generate','structured','tool','guardrail','verified'}:
            raise ValueError('Unknown mode')
        start = time.perf_counter()
        record = record_override if record_override is not None else self.records[doc_id]
        if record['doc_id'] != doc_id:
            raise ValueError('Record override must retain the requested ID')
        view = source_view(record)
        user = request + '\nSOURCE RECORD:\n' + json.dumps(view, ensure_ascii=False)
        self.generator.calls = []
        self.generator.tool_events = []
        reasons = risk_flags(user) if mode == 'guardrail' else []
        blocked = bool(reasons)
        cfg = OmegaConf.create({
            'prompt': user, 'prompt_file': None, 'system_prompt': SYSTEM,
            'seed': settings['seed'],
            'generation': {k:v for k,v in settings.items() if k != 'seed'},
            'model': {'name':'Phi-4-mini-instruct-4bit-MLX',
                      'supports_structured_output':True, 'supports_tool_calling':True,
                      'tool_calling_format':'functools_prompt'},
            'structured_output': {'enabled':True,'schema_path':str(self.schema_path),'strict':False},
            'tool_calling': {'enabled':True,'tool_choice':'auto','tools':[
                {'name':'lookup_selected_posting','description':'Read the currently selected public posting. Takes zero arguments.',
                 'parameters':{'type':'object','properties':{},'additionalProperties':False}}]},
        })
        raw = ''
        entrypoint = None
        if not blocked:
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                if mode == 'generate':
                    entrypoint = 'src.modes.Modes.run_generate'
                    self.modes.run_generate(cfg)
                elif mode == 'tool':
                    entrypoint = 'src.modes.Modes.run_tool_calling'
                    cfg.prompt = 'Read the selected posting using lookup_selected_posting with no arguments, then extract its facts.'
                    cfg.system_prompt = (
                        'You are a tool router. The lookup_selected_posting function is available and will be executed by the application. '
                        'Call it to retrieve the selected posting before answering. It takes ZERO arguments: arguments must be {}. '
                        'Do not add postingId, post_id, doc_id or any other argument. '
                        'Do not use Markdown fences, invent facts, or explain the call. '
                        'After the tool result you will receive the extraction instructions.')
                    saved = dict(TOOL_REGISTRY)
                    TOOL_REGISTRY.clear()
                    TOOL_REGISTRY['lookup_selected_posting'] = lambda: view
                    try:
                        # Upstream prints tool results to stderr. Capture them to
                        # avoid duplicating corpus text in terminal progress output.
                        with contextlib.redirect_stderr(io.StringIO()):
                            self.modes.run_tool_calling(cfg)
                    finally:
                        TOOL_REGISTRY.clear(); TOOL_REGISTRY.update(saved)
                else:
                    entrypoint = 'src.modes.Modes.run_structured_output'
                    self.modes.run_structured_output(cfg)
            raw = printed.getvalue().strip()
            if mode == 'guardrail':
                reasons += ['output:'+flag for flag in risk_flags(raw)]
                blocked = bool(reasons)
        facts, schema_error = parse_facts(raw) if raw else (None, 'No model output')
        errors = factual_errors(facts, record) if facts else []
        if mode == 'verified' and (not facts or errors):
            blocked = True
            reasons = ['invalid_schema'] if not facts else errors
        calls = self.generator.calls
        return {'doc_id':doc_id,'mode':mode,'request':request,
            'response':REFUSAL if blocked else raw,'raw_output':raw,'blocked':blocked,
            'reasons':reasons,'schema_valid':facts is not None,
            'facts':facts.model_dump() if facts else None,'schema_error':schema_error,
            'source_errors':errors,'calls':calls,'tool_events':list(self.generator.tool_events),
            'latency_seconds':time.perf_counter()-start,
            'input_tokens':sum(c['input_tokens'] for c in calls),
            'output_tokens':sum(c['output_tokens'] for c in calls),
            'llmbox_entrypoint':entrypoint,
            'configuration':OmegaConf.to_container(cfg,resolve=True)}


def prepare_schema():
    path = Path(__file__).with_name('job_facts.schema.json')
    text = json.dumps(JobFacts.model_json_schema(),ensure_ascii=False,indent=2)
    if path.exists() and path.read_text() != text:
        raise ValueError('Schema changed; preserve the existing protocol first')
    path.write_text(text)

if __name__ == '__main__':
    prepare_schema()
