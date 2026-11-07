import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'05_Frozen_Data_Indices/independent_test_300_input_only.jsonl'; S=json.loads((ROOT/'05_Frozen_Data_Indices/xiaoke_schema_v5.json').read_text(encoding='utf-8'))
TYPES=set(S['entity_types']); LEGAL={(x['head_type'],x['relation'],x['tail_type']) for x in S['relation_schemas']}
def rows(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def main(mode):
    pilot=mode=='pilot10'; p=ROOT/('results/E3_YAYI-UIE_pilot10.jsonl' if pilot else 'results/E3_YAYI-UIE_300_predictions.jsonl')
    src=rows(DATA)[:10] if pilot else rows(DATA); pred=rows(p)
    expected=10 if pilot else 300
    if len(pred)!=expected or [x['record_id'] for x in pred]!=[x['record_id'] for x in src]: raise RuntimeError('数量、ID或顺序错误')
    parse_warnings=[]
    multi_type=[]; nonempty_entities=0
    for r in pred:
        sm={}
        for s,t in r['entities']:
            if t not in TYPES or s not in r['text']: raise RuntimeError(f"非法实体:{r['record_id']}:{s}|{t}")
            sm.setdefault(s,set()).add(t)
        if sm: nonempty_entities+=1
        multi_type.extend((r['record_id'],s,sorted(ts)) for s,ts in sm.items() if len(ts)>2)
        for h,rel,t in r['relations']:
            if h not in sm or t not in sm or not any((ht,rel,tt) in LEGAL for ht in sm[h] for tt in sm[t]): raise RuntimeError(f"非法关系:{r['record_id']}:{h}|{rel}|{t}")
        if not all(r.get('parse_status',{}).values()): parse_warnings.append(r['record_id'])
    if multi_type: raise RuntimeError(f'发现同一表面形式被复制到3类及以上，拒绝进入正式运行：{multi_type[:5]}')
    if pilot and nonempty_entities==0: raise RuntimeError('10条实体输出全部为空，拒绝进入正式运行')
    report={'status':'passed','mode':mode,'records':expected,'nonempty_entity_records':nonempty_entities,'parse_warning_records':parse_warnings,'prediction_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'test_input_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest()}
    out=ROOT/('results/E3_pilot10_output_audit.json' if pilot else 'results/E3_cloud_output_audit.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'E3 {mode}结构审计通过；实体非空{nonempty_entities}/{expected}，解析警告{len(parse_warnings)}条。')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',nargs='?',choices=['pilot10','full300'],default='full300'); main(ap.parse_args().mode)
