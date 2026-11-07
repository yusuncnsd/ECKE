"""离线结构检查；合成案例非提示示例，不读Gold。"""
import ast,importlib.util,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('runner',ROOT/'02_run_adelie_optimized.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
data,schema,cfg=m.preflight()
assert cfg['do_sample'] is False and cfg['temperature']==0 and cfg['repetition_penalty']==1.1
assert len(m.prompts(data[0]['text'],schema))==1
row={'ann_id':'synthetic','record_id':'synthetic','text':'甲方治乙病。'}
e={t:[] for t in schema['entity_types']};e['方药']=['甲方'];e['疾病']=['乙病']
raw=json.dumps({'entities':e,'relations':[{'head':'甲方','relation':'治疗','tail':'乙病'}]},ensure_ascii=False)
p=m.normalize_joint(row,raw,schema);m.audit([p],[row],schema)
assert p['relations']==[['甲方','治疗','乙病']] and p['method']=='ADELIE-DPO-7B-Optimized'
for raw in ('{"entities":', '{"entities":{},"relations":[]}', '{"entities":{},"relations":[],"extra":1}'):
    try:m.normalize_joint(row,raw,schema)
    except ValueError:pass
    else:raise AssertionError('不完整/空对象被接受')
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'truncated.json'
    try:m.capture_generation(path,{'raw_response':'{"entities":','reached_budget_without_eos':True})
    except ValueError:pass
    else:raise AssertionError('截断被接受')
    assert path.exists()
assert 'ADELIE_v3_pilot_records.zip' not in (ROOT/'02_run_adelie_optimized.py').read_text(encoding='utf-8')
print('通过：300条输入/Schema指纹、联合解析、合法关系、截断拒绝与先落盘、独立缓存；未读Gold、未使用GPU。')
