"""Real-weight correctness checks. Never substitutes synthetic speed results."""
import argparse
import json
import hashlib
import random
import threading
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'host'))
from nodelift.native import Native, library_path

parser=argparse.ArgumentParser()
parser.add_argument('--model',default=str(ROOT/'.models/qwen2.5-0.5b-instruct-q4_k_m.gguf'))
parser.add_argument('--gpu-layers',type=int,default=999)
parser.add_argument('--strict',action='store_true',help='Verify tokens one at a time using the serial correctness path')
parser.add_argument('--output',default=str(ROOT/'artifacts/native-correctness.json'))
args=parser.parse_args()
checks=[]
with Native() as reference, Native() as verifier:
    info=reference.call('load',path=args.model,n_ctx=512,threads=8,gpu_layers=args.gpu_layers)
    other=verifier.call('load',path=args.model,n_ctx=512,threads=8,gpu_layers=args.gpu_layers)
    assert info['tokenizer_fingerprint']==other['tokenizer_fingerprint']
    for prompt in ['Say hello briefly.','Return exactly the integers from 1 to 12, separated by commas.','Write a Python function adding two numbers.','Rispondi con una frase: perché il cielo appare blu?']:
        prefix=reference.call('tokenize',text=prompt,chat=True)['tokens']
        reference.call('reset');expected=reference.call('draft',prefix=prefix,count=64)['tokens']
        for k in (1,4,16):
            verifier.call('reset');actual=[];accepted=0
            while len(actual)<len(expected):
                # Perfect proposals from the separately computed target oracle.
                proposals=expected[len(actual):len(actual)+k]
                result=verifier.call('verify',strict=args.strict,prefix=prefix+actual,proposal=proposals,remaining=len(expected)-len(actual))
                actual+=result['tokens'];accepted+=result['accepted']
                if result['eos']:break
            assert actual==expected, f'greedy batch drift or verification error at k={k}: {prompt}'
            checks.append({'case':'perfect proposals','prompt':prompt,'k':k,'tokens':len(actual),'passed':True})
        verifier.call('reset');actual=[]
        while len(actual)<min(len(expected),24):
            wrong=(expected[len(actual)]+1)%info['vocab_size']
            result=verifier.call('verify',strict=args.strict,prefix=prefix+actual,proposal=[wrong]*4,remaining=min(len(expected),24)-len(actual))
            assert result['accepted']==0 and len(result['tokens'])==1
            actual+=result['tokens']
            if result['eos']:break
        assert actual==expected[:len(actual)]
        checks.append({'case':'all rejected','prompt':prompt,'tokens':len(actual),'passed':True})
        # Randomized partial-acceptance paths, including stale KV after correction.
        rng=random.Random(20261031);verifier.call('reset');actual=[]
        while len(actual)<len(expected):
            k=rng.randint(1,16);remaining=expected[len(actual):len(actual)+k]
            proposals=remaining[:]
            if proposals:
                reject=rng.randrange(len(proposals))
                proposals[reject]=(proposals[reject]+1)%info['vocab_size']
            result=verifier.call('verify',strict=args.strict,prefix=prefix+actual,proposal=proposals,remaining=len(expected)-len(actual))
            actual+=result['tokens']
            assert actual==expected[:len(actual)], 'partial acceptance/rollback drift'
            if result['eos']:break
        assert actual==expected
        checks.append({'case':'randomized partial acceptance','prompt':prompt,'tokens':len(actual),'passed':True})
        # Change the committed suffix, then restore it; stale KV must be discarded.
        verifier.call('draft',prefix=prefix+[(expected[0]+7)%info['vocab_size']],count=4)
        result=verifier.call('draft',prefix=prefix,count=min(8,len(expected)))
        assert result['tokens']==expected[:len(result['tokens'])]
        checks.append({'case':'prefix rollback','prompt':prompt,'passed':True})
    prefix=reference.call('tokenize',text='Hello',chat=True)['tokens']
    for bad in (-1,info['vocab_size']):
        try:verifier.call('verify',strict=args.strict,prefix=prefix,proposal=[bad],remaining=4)
        except RuntimeError:checks.append({'case':'invalid token rejected','token':bad,'passed':True})
        else:raise AssertionError('invalid token accepted')
    try:verifier.call('verify',strict=args.strict,prefix=[prefix[0]]*510,proposal=[prefix[0]]*16,remaining=16)
    except RuntimeError:checks.append({'case':'context overflow rejected','passed':True})
    else:raise AssertionError('context overflow accepted')
    # The final context slot may hold an accepted proposal, but not a bonus.
    # Search a few deterministic contexts so the test covers actual acceptance,
    # even when batched logits differ from the one-token path for one context.
    edge=None
    for text in ('count the words','hello world','test','one two three','cache'):
        pattern=verifier.call('tokenize',text=text,chat=False)['tokens']
        edge_prefix=(pattern*((info['n_ctx']+len(pattern)-2)//len(pattern)))[:info['n_ctx']-1]
        verifier.call('reset')
        last=verifier.call('draft',prefix=edge_prefix,count=1)['tokens']
        edge=verifier.call('verify',strict=args.strict,prefix=edge_prefix,proposal=last,remaining=4)
        assert len(edge['tokens'])==1, 'verification emitted beyond context'
        if edge['accepted']==1:break
    assert edge['accepted']==1, 'no accepted boundary proposal was exercised'
    checks.append({'case':'accepted boundary proposal excludes bonus','passed':True})
    try:verifier.call('verify',strict=args.strict,prefix=edge_prefix+last,proposal=[],remaining=1)
    except RuntimeError:checks.append({'case':'full context rejected','passed':True})
    else:raise AssertionError('full context accepted')
    # Both explicit cancellation and teardown alone must stop/retain active calls safely.
    for explicit_cancel in (True,False):
        raced=Native();raced.call('load',path=args.model,n_ctx=1024,threads=2,gpu_layers=args.gpu_layers)
        race_prefix=raced.call('tokenize',text='Write a detailed long implementation of a binary tree in Python.',chat=True)['tokens']
        outcome={}
        def active_call():
            try:outcome['value']=raced.call('draft',prefix=race_prefix,count=512)
            except RuntimeError as error:outcome['error']=str(error)
        thread=threading.Thread(target=active_call);thread.start();time.sleep(.05)
        if explicit_cancel:raced.cancel()
        raced.close();thread.join(15)
        assert not thread.is_alive() and ('value' in outcome or 'error' in outcome)
        checks.append({'case':'cancel/close during native call' if explicit_cancel else 'close alone during native call',
                       'passed':True,'outcome':outcome.get('error','completed before close')})
    output={'native_library_sha256':hashlib.sha256(library_path().read_bytes()).hexdigest(),
            'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'model':Path(args.model).name,'gpu_layers_requested':args.gpu_layers,'strict':args.strict,'info':info,'checks':checks,'time':time.time()}
    Path(args.output).write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(f'{len(checks)} real-model correctness checks passed; {args.output}')
