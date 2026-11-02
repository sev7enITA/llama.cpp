"""Matched native KV-prefix reuse probe. Developer experiment, not product runtime."""
import argparse,hashlib,json,os,random,statistics,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'host'))
from nodelift.native import Native,library_path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True);args=p.parse_args()
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
rng=random.Random(20261102);model=Path(args.model).resolve();engine=Native();rows=[]
try:
 info=engine.call('load',path=str(model),n_ctx=2048,threads=8,gpu_layers=999)
 text=('Local AI systems can reuse earlier context when the exact token prefix and model configuration remain unchanged. ' * 80)
 base=engine.call('tokenize',text=text,chat=False)['tokens']
 old=engine.call('tokenize',text='Earlier response: the result was recorded.',chat=False)['tokens']
 new=engine.call('tokenize',text='New question: explain the engineering trade-offs briefly.',chat=False)['tokens']
 assert len(base)>=1024 and old and new
 # Tokenization may add the same BOS token to both suffixes. Report the
 # complete shared prefix, not only the size of the synthetic base.
 shared_suffix=0
 for a,b in zip(old,new):
  if a!=b:break
  shared_suffix+=1
 for size in (256,1024):
  for rep in range(6):
   order=['cold','reuse'];rng.shuffle(order)
   for arm in order:
    prefix=base[:size]+new;engine.call('reset');engine.call('prefill',prefix=base[:size]+old)
    if arm=='cold':engine.call('reset')
    start=time.perf_counter();prefill=engine.call('prefill',prefix=prefix);done=time.perf_counter()
    answer=engine.call('draft',prefix=prefix,count=16);end=time.perf_counter()
    rows.append({'size':size,'actual_shared_prefix_tokens':size+shared_suffix,'repetition':rep,'arm':arm,'prefill_wall_ms':(done-start)*1000,'prefill_native_ms':prefill['ms'],'prefill_plus_16_wall_ms':(end-start)*1000,'tokens':answer['tokens'],'eos':answer['eos']})
    print(size,rep,arm,round((done-start)*1000,2),flush=True)
 pairs=[]
 for size in (256,1024):
  for rep in range(6):
   a=next(x for x in rows if x['size']==size and x['repetition']==rep and x['arm']=='cold');b=next(x for x in rows if x['size']==size and x['repetition']==rep and x['arm']=='reuse')
   pairs.append({'size':size,'repetition':rep,'prefill_speedup':a['prefill_wall_ms']/b['prefill_wall_ms'],'prefill_plus_16_speedup':a['prefill_plus_16_wall_ms']/b['prefill_plus_16_wall_ms'],'tokens_equal':a['tokens']==b['tokens'],'eos_equal':a['eos']==b['eos']})
 summary={str(size):{'median_prefill_speedup':statistics.median(x['prefill_speedup'] for x in pairs if x['size']==size),'median_prefill_plus_16_speedup':statistics.median(x['prefill_plus_16_speedup'] for x in pairs if x['size']==size),'matched_pairs':sum(x['tokens_equal'] and x['eos_equal'] for x in pairs if x['size']==size)} for size in (256,1024)}
 record={'model_file':Path(args.model).name,'model_sha256':sha(model),'library_sha256':sha(library_path()),'runtime':info,'seed':20261102,'threads_requested':8,'gpu_layers_requested':999,'plan':'PLAN.md','scope':'native microbenchmark, no model load or live controller','rows':rows,'pairs':pairs,'summary':summary}
 Path(args.out).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(summary),flush=True)
finally:engine.close()
