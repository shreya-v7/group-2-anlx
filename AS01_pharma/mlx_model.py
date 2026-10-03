"""Optional Apple Silicon Phi backend. Load a local MLX conversion, never download.
Model used: mlx-community/Phi-4-mini-instruct-4bit, revision recorded in manifest.
"""
import json
import time
from pathlib import Path
from local_model import PhiResponse
from schema import extract_json, validate_against_schema

class MLXPhiClient:
    provider='local:phi-4-mini-instruct:mlx'
    def __init__(self,model_path,*,device='auto',max_new_tokens=4096):
        if not model_path or not (Path(model_path)/'config.json').exists():
            raise ValueError('MLX requires an existing local --model-path.')
        from mlx_lm import load
        self.llm,self.tokenizer=load(model_path)
        self.max_new_tokens=max_new_tokens
        self.config=json.loads((Path(model_path)/'config.json').read_text())
        self.model_path=model_path
        print(f'Loaded local MLX Phi from {model_path}',flush=True)
    def structured(self,prompt,schema,*,system=None,**kwargs):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        instruction='Respond with a single JSON object matching this schema. No preamble, no explanation, no markdown fences.\n'+json.dumps(schema,indent=2)
        messages=[{'role':'system','content':f'{system}\n\n{instruction}' if system else instruction},{'role':'user','content':prompt}]
        formatted=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        token_count=len(self.tokenizer.encode(formatted))
        if token_count+self.max_new_tokens>self.config.get('max_position_embeddings',131072):
            raise ValueError('Full document plus output exceeds model context capacity.')
        start=time.perf_counter()
        text=generate(self.llm,self.tokenizer,prompt=formatted,max_tokens=self.max_new_tokens,sampler=make_sampler(temp=0),verbose=False)
        elapsed=time.perf_counter()-start
        try: parsed=extract_json(text)
        except ValueError as exc:return PhiResponse(text,None,f'unparseable output: {exc}',elapsed)
        if not isinstance(parsed,dict):return PhiResponse(text,None,'expected object',elapsed)
        return PhiResponse(text,parsed,validate_against_schema(parsed,schema),elapsed)
