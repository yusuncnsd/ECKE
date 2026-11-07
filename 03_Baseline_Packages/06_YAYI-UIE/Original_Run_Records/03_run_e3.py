from __future__ import annotations
import argparse, ast, hashlib, json, os, re, tempfile, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'05_Frozen_Data_Indices/data'; RESULTS=ROOT/'results'; MODEL=ROOT/'model/YAYI-UIE'
INPUT=DATA/'independent_test_300_input_only.jsonl'; CACHE=RESULTS/'E3_raw_cache.jsonl'; FAIL=RESULTS/'E3_failures.json'; RUN=RESULTS/'E3_run_manifest.json'
FINAL=RESULTS/'E3_YAYI-UIE_300_predictions.jsonl'; PILOT=RESULTS/'E3_YAYI-UIE_pilot10.jsonl'
TYPES=('中药','方药','疾病','症状','病机','病位','治法')
SCHEMAS=(('中药','治疗','疾病'),('方药','治疗','疾病'),('治法','主治','疾病'),('病机','导致','疾病'),('病机','涉及','病位'),('症状','提示','病机'),('疾病','表现','症状'))

def read_jsonl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=p.name,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def append(obj):
    RESULTS.mkdir(parents=True,exist_ok=True)
    with CACHE.open('a',encoding='utf-8',newline='\n') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
def extract_json(raw):
    raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip(),flags=re.I|re.S)
    starts=[i for i,c in enumerate(raw) if c in '[{']
    for start in starts:
        for end in range(len(raw),start,-1):
            fragment=raw[start:end]
            try: return json.loads(fragment)
            except Exception:
                try:
                    value=ast.literal_eval(fragment)
                    if isinstance(value,(dict,list)): return value
                except Exception: pass
    return None
def ner_prompt(text):
    labels='/'.join(TYPES)
    return (f'文本：{text}\n〖实体抽取〗抽取文本中可能存在且与消渴专题直接相关的实体，'
            f'并以json{{{labels}：[实体]}}格式输出。实体必须逐字存在于原文；'
            '不得把仅有的临床表现改写为疾病；不得因方剂治疗消渴而展开组成中药。')
def relation_prompt(text):
    opts='；'.join(f'{h}-{r}-{t}' for h,r,t in SCHEMAS)
    rels=','.join(dict.fromkeys(r for _,r,_ in SCHEMAS))
    return (f'文本：{text}\n〖关系抽取〗已知关系列表是[{rels}]。根据关系列表抽取当前文本中明确成立且与消渴专题直接相关的关系三元组，'
            "按照json[{'relation':'', 'head':'', 'tail':''}, ]的格式输出。"
            f'合法的头尾实体类型组合仅限：{opts}。关系必须由原文直接支持，不得输出方剂组成关系。')
def wrap(prompt): return '<reserved_13>'+prompt+'<reserved_14>'
def generate(model,tok,prompt):
    inputs=tok(wrap(prompt),return_tensors='pt',truncation=True,max_length=3072).to(model.device)
    with torch.inference_mode(): out=model.generate(**inputs,do_sample=False,temperature=None,top_p=None,max_new_tokens=768,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
def parse_entities(raw,text):
    obj=extract_json(raw); out=set()
    if isinstance(obj,dict):
        for typ,items in obj.items():
            typ=str(typ).strip()
            if typ not in TYPES: continue
            if isinstance(items,str): items=[items]
            if isinstance(items,list):
                for item in items:
                    s=item.get('text') if isinstance(item,dict) else item
                    if isinstance(s,str):
                        s=s.strip().strip('“”\"\'')
                    if isinstance(s,str) and s and s in text: out.add((s,typ))
    return out,isinstance(obj,dict)
def parse_relations(raw,text):
    obj=extract_json(raw); out=set()
    if isinstance(obj,dict): obj=obj.get('relations',obj.get('关系',[]))
    if isinstance(obj,list):
        for x in obj:
            if isinstance(x,dict):
                h=x.get('head') or x.get('头实体'); r=x.get('relation') or x.get('关系'); t=x.get('tail') or x.get('尾实体')
                if all(isinstance(v,str) for v in (h,r,t)): h,r,t=(v.strip().strip('“”\"\'') for v in (h,r,t))
                if all(isinstance(v,str) for v in (h,r,t)) and h in text and t in text and r in {z[1] for z in SCHEMAS}: out.add((h,r,t))
            elif isinstance(x,list) and len(x)>=3 and all(isinstance(v,str) for v in x[:3]):
                h,r,t=x[:3]
                if h in text and t in text:
                    if r in {z[1] for z in SCHEMAS}: out.add((h,r,t))
    return out,obj is not None
def normalize(row,raw_e,raw_r):
    entities,ok_e=parse_entities(raw_e,row['text']); triples,ok_r=parse_relations(raw_r,row['text'])
    by_surface={}
    for s,t in entities: by_surface.setdefault(s,set()).add(t)
    relations=set()
    for h,r,t in triples:
        if h not in by_surface or t not in by_surface: continue
        if any((ht,r,tt) in SCHEMAS for ht in by_surface[h] for tt in by_surface[t]): relations.add((h,r,t))
    return {'ann_id':row['ann_id'],'record_id':row['record_id'],'text':row['text'],'entities':[list(x) for x in sorted(entities)],'relations':[list(x) for x in sorted(relations)],'method':'E3_YAYI-UIE','parse_status':{'entity_json':ok_e,'relation_json':ok_r},'raw_entity':raw_e,'raw_relation':raw_r}
def main(mode):
    rows=read_jsonl(INPUT); rows=rows[:10] if mode=='pilot10' else rows
    binding={'version':'E3_YAYI_UIE_formal_v3_official_task_markers_20260826','model_repository':'wenge-research/yayi-uie','test_sha256':sha(INPUT),'schema_sha256':sha(DATA/'xiaoke_schema_v5.json'),'setting':'strict_zero_shot_official_task_markers','temperature':0.0,'max_input_tokens':3072,'max_new_tokens':768,'mode':mode}
    mode_cache=RESULTS/f'E3_{mode}_raw_cache.jsonl'; global CACHE; CACHE=mode_cache
    mode_manifest=RESULTS/f'E3_{mode}_run_manifest.json'
    if mode_manifest.exists() and json.loads(mode_manifest.read_text(encoding='utf-8'))!=binding: raise RuntimeError('E3缓存指纹不一致，禁止混用')
    atomic_json(mode_manifest,binding); existing={x['record_id']:x for x in read_jsonl(CACHE)} if CACHE.exists() else {}; pending=[x for x in rows if x['record_id'] not in existing]
    print(f'[E3] 已恢复{len(rows)-len(pending)}条，待处理{len(pending)}条。')
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,use_fast=False); tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,quantization_config=quant,device_map='auto',torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).eval()
    start=time.time(); failures=[]
    for i,row in enumerate(pending,1):
        try:
            result=normalize(row,generate(model,tok,ner_prompt(row['text'])),generate(model,tok,relation_prompt(row['text']))); append(result); existing[row['record_id']]=result
        except Exception as exc: failures.append({'record_id':row['record_id'],'error':repr(exc)})
        elapsed=time.time()-start; remain=elapsed/i*(len(pending)-i)
        print(f'[E3] {i}/{len(pending)} ({100*i/max(1,len(pending)):.1f}%) 本批成功={i-len(failures)} 本批失败={len(failures)} 待处理={len(pending)-i} 已用={elapsed:.1f}s 预计剩余={remain:.1f}s',flush=True)
    atomic_json(RESULTS/f'E3_{mode}_failures.json',failures); missing=[x['record_id'] for x in rows if x['record_id'] not in existing]
    if missing: raise RuntimeError(f'E3未全部完成：缺少{len(missing)}条；重新运行仅处理失败项')
    output=[]
    for row in rows:
        x=dict(existing[row['record_id']]); x.pop('raw_entity',None); x.pop('raw_relation',None); output.append(x)
    target=PILOT if mode=='pilot10' else FINAL; target.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in output)+'\n',encoding='utf-8'); print('E3完成：',target)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['pilot10','full300']); main(ap.parse_args().mode)
