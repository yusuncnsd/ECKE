"""ADELIE-DPO-7B：零样本NER、端到端RE；不读取Gold或开发案例。"""

from __future__ import annotations

import argparse, ast, hashlib, json, os, tempfile, time, zipfile

from pathlib import Path



ROOT = Path(__file__).resolve().parent

DATA = ROOT / '05_Frozen_Data_Indices/data'

INPUT = DATA / 'independent_test_300_input_only.jsonl'

SCHEMA = DATA / 'xiaoke_schema_v5.json'

CONFIG = ROOT / 'config_optimized.json'

MODEL = ROOT / 'model/ADELIE-DPO'

VERSION = 'ADELIE_adapted_joint_greedy_20261006'



def sha(path):

    h = hashlib.sha256()

    with Path(path).open('rb') as f:

        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):

            h.update(b)

    return h.hexdigest()



def rows(path):

    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]



def atomic_text(path, text):

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')

    try:

        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:

            f.write(text); f.flush(); os.fsync(f.fileno())

        for attempt in range(6):

            try:

                os.replace(tmp, path); break

            except PermissionError:

                if attempt == 5: raise

                time.sleep(min(8, 2 ** attempt))

    finally:

        if os.path.exists(tmp): os.unlink(tmp)



def save(path, obj):

    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2))



def preflight():

    manifest = json.loads((ROOT / 'asset_manifest.json').read_text(encoding='utf-8'))

    if sha(INPUT) != manifest['test_input_sha256'] or sha(SCHEMA) != manifest['schema_sha256']:

        raise RuntimeError('Inputs/Schema与冻结指纹不一致')

    data = rows(INPUT)

    if len(data) != 300 or len({x['record_id'] for x in data}) != 300 or len({x['ann_id'] for x in data}) != 300:

        raise RuntimeError('测试输入必须为300个唯一ID')

    if any(set(x) != {'ann_id', 'record_id', 'text'} or not isinstance(x['text'], str) or not x['text'] for x in data):

        raise RuntimeError('输入含答案/额外字段或无效原文')

    schema = json.loads(SCHEMA.read_text(encoding='utf-8'))

    if len(schema['entity_types']) != 7 or len(schema['relation_schemas']) != 7:

        raise RuntimeError('不是已冻结7类实体、7种关系模式')

    return data, schema, json.loads(CONFIG.read_text(encoding='utf-8'))



def prompts(text, schema):
    patterns='; '.join(x['head_type']+'|'+x['relation']+'|'+x['tail_type'] for x in schema['relation_schemas'])
    shape={'entities':{t:[] for t in schema['entity_types']},'relations':[]}
    prompt=('Extract only xiaoke (消渴) knowledge explicitly supported by this text. '
            'Return ONE JSON object, no explanation: '+json.dumps(shape,ensure_ascii=False)+'. '
            'Entity values must be exact text spans, not category names. Disease requires a written disease name. '
            'Do not include unrelated medicine or formula ingredients. '
            '内消/胃消/渴疾/渴症/大渴/中焦渴/下焦渴/谷消 are 症状. '
            'Each relation is {"head":"exact entity span","relation":"allowed relation label","tail":"exact entity span"}. '
            'Both endpoints must appear in your entities with these allowed types: '+patterns+'. '
            'A compatible pair alone is not evidence of a relation. Use [] when absent. Text:\n'+text)
    return (prompt,)


def wrap(prompt):

    # 对应官方create_prompt_with_tulu_chat_format([user], add_bos=False)；tokenizer添加BOS一次。

    return '<|user|>\n' + prompt + '\n<|assistant|>\n'



def parse(raw, expected):

    if not isinstance(raw, str): raise ValueError('响应不是字符串')

    raw = raw.strip()

    if raw.startswith('[Answer]:'):

        raw = raw[len('[Answer]:'):].strip()

    if raw.startswith('```'):

        lines = raw.splitlines()

        if len(lines) >= 3 and lines[-1].strip() == '```': raw = '\n'.join(lines[1:-1])

    try:

        obj = json.loads(raw)

    except json.JSONDecodeError:

        try: obj = ast.literal_eval(raw)

        except (ValueError, SyntaxError) as exc: raise ValueError('响应不是完整JSON/安全字面量') from exc

    # 完整实体记录列表仅做结构转换，不推断或改写预测值。

    if expected is dict and isinstance(obj, list):

        if obj and all(isinstance(x, dict) and set(x) == {'名称', '类型'} for x in obj):

            converted = {}

            for item in obj:

                if not isinstance(item['名称'], str) or not isinstance(item['类型'], list) or not item['类型'] or any(not isinstance(t, str) for t in item['类型']):

                    raise ValueError('实体记录格式错误')

                for typ in item['类型']: converted.setdefault(typ, []).append(item['名称'])

            obj = converted

        elif len(obj) == 1 and isinstance(obj[0], dict): obj = obj[0]

    if not isinstance(obj, expected): raise ValueError('响应顶层结构错误')

    return obj



def capture_generation(path, details):

    # 在解析及截断判断之前落盘；失败响应绝不进入成功缓存。

    save(path, details)

    if details['reached_budget_without_eos']:

        raise ValueError('输出达到长度上限，禁止视为完整预测；原始响应已保存')

    return details['raw_response']



def normalize(row, raw_e, raw_r, schema):

    obj_e, obj_r = parse(raw_e, dict), parse(raw_r, list)

    types = set(schema['entity_types'])

    legal = {(x['head_type'], x['relation'], x['tail_type']) for x in schema['relation_schemas']}

    if any(k not in types or not isinstance(v, list) or any(not isinstance(s, str) for s in v) for k,v in obj_e.items()):

        raise ValueError('实体字典类别或列表结构错误')

    if not obj_e: raise ValueError('空对象不能代替各类别空列表')

    entities = {(s.strip(), typ) for typ, items in obj_e.items() for s in items if s.strip() and s.strip() in row['text']}

    by_surface = {}

    for s, typ in entities: by_surface.setdefault(s, set()).add(typ)

    relations = set(); discarded = len([s for v in obj_e.values() for s in v if not s.strip() or s.strip() not in row['text']])

    for item in obj_r:

        if not isinstance(item, dict) or set(item) != {'head', 'relation', 'tail'} or not all(isinstance(v, str) for v in item.values()):

            raise ValueError('关系项结构错误')

        h, r, t = (item[k].strip() for k in ('head', 'relation', 'tail'))

        if any((ht, r, tt) in legal for ht in by_surface.get(h, ()) for tt in by_surface.get(t, ())):

            relations.add((h,r,t))

        else: discarded += 1

    return {**row, 'entities': [list(x) for x in sorted(entities)], 'relations': [list(x) for x in sorted(relations)],

            'method': 'ADELIE-DPO-7B-Optimized', 'parse_status': {'entity_json': True, 'relation_json': True},

            'structural_discard_count': discarded, 'raw_entity': raw_e, 'raw_relation': raw_r}



def normalize_joint(row, raw, schema):
    obj = parse(raw,dict)
    if set(obj) != {'entities','relations'}: raise ValueError('联合响应必须仅包含entities/relations')
    result = normalize(row,json.dumps(obj['entities'],ensure_ascii=False),json.dumps(obj['relations'],ensure_ascii=False),schema)
    result['raw_joint']=raw
    return result


def read_cache(path, expected):

    if not path.exists(): return {}

    existing = {}

    chunks = path.read_bytes().splitlines(keepends=True); good = b''

    for i, chunk in enumerate(chunks):

        try: item = json.loads(chunk)

        except (ValueError, UnicodeDecodeError):

            if i != len(chunks)-1 or chunk.endswith(b'\n'): raise RuntimeError('缓存中间行损坏')

            atomic_text(path, good.decode('utf-8')); print('恢复中断留下的未提交末行', flush=True); break

        rid = item['record_id']

        if rid in existing or rid not in expected or item['text'] != expected[rid]['text']:

            raise RuntimeError('缓存重复ID或原文不一致')

        existing[rid] = item; good += chunk

    # 有效JSON但末尾未换行时，下次追加前补换行。

    if good and not good.endswith(b'\n'): atomic_text(path, good.decode('utf-8') + '\n')

    return existing



def audit(pred, source, schema):

    if [x['record_id'] for x in pred] != [x['record_id'] for x in source]: raise RuntimeError('输出ID顺序不一致')

    legal = {(x['head_type'], x['relation'], x['tail_type']) for x in schema['relation_schemas']}

    for x, src in zip(pred, source):

        if any(x[k] != src[k] for k in ('ann_id', 'text')): raise RuntimeError('输出原文/ann_id不一致')

        sm = {}

        for s,t in x['entities']:

            if not s or s not in x['text'] or t not in schema['entity_types']: raise RuntimeError('非法实体')

            sm.setdefault(s, set()).add(t)

        if any(len(v)>2 for v in sm.values()): raise RuntimeError('同一surface跨3类复制，结构验收停止')

        for h,r,t in x['relations']:

            if not any((ht,r,tt) in legal for ht in sm.get(h,()) for tt in sm.get(t,())): raise RuntimeError('非法关系')

    return {'records': len(pred), 'nonempty_entity_records': sum(bool(x['entities']) for x in pred), 'status': 'passed'}



def main(mode):

    data, schema, cfg = preflight()

    if mode == 'check':

        print('本地检查通过：300条无答案输入，冻结Schema，未加载模型。'); return

    import torch

    from transformers import AutoTokenizer, AutoModelForCausalLM

    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:

        raise RuntimeError('需要至少20GiB CUDA显存；建议RTX4090 24GB')

    model_binding = json.loads((ROOT / 'model/model_binding.json').read_text(encoding='utf-8'))

    if model_binding.get('status') != 'complete' or model_binding['repository'] != cfg['model_repository']:

        raise RuntimeError('官方模型资产未完成核验')

    for name, info in model_binding['files'].items():

        if sha(MODEL / name) != info['sha256']: raise RuntimeError('模型资产指纹变更：' + name)

    selected = data[:10] if mode == 'pilot10' else data

    results = ROOT / 'results_optimized'; results.mkdir(exist_ok=True)

    response_dir = results / 'response_audit' / f'{mode}_{time.time_ns()}'

    response_dir.mkdir(parents=True)

    binding = {'version': VERSION, 'test_sha256': sha(INPUT), 'schema_sha256': sha(SCHEMA),

               'config_sha256': sha(CONFIG), 'script_sha256': sha(__file__), 'model': model_binding}

    manifest = results / 'run_manifest.json'; cache = results / 'raw_cache.jsonl'

    if cache.exists() and not manifest.exists(): raise RuntimeError('缓存缺少指纹文件')

    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['binding'] != binding:

        raise RuntimeError('实验指纹变化，禁止混用缓存；请另建独立结果目录')

    if not manifest.exists(): save(manifest, {'binding': binding, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),

                                            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})

    existing = read_cache(cache, {x['record_id']:x for x in data})

    terminal = results / 'terminal_failures.json'

    failures = json.loads(terminal.read_text(encoding='utf-8')) if terminal.exists() else []

    failed_ids = {x['record_id'] for x in failures}

    if len(failed_ids) != len(failures) or failed_ids & set(existing) or not failed_ids <= {x['record_id'] for x in data}:

        raise RuntimeError('终止失败清单ID不一致')

    pending = [(i,x) for i,x in enumerate(selected) if x['record_id'] not in existing and x['record_id'] not in failed_ids]

    if pending:

        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)

        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map='auto',

                                                   local_files_only=True, low_cpu_mem_usage=True).eval()

        context = int(model.config.max_position_embeddings)

        # 全300条在首轮pilot即检查，绝不静默截断或拆条。

        lengths = [len(tokenizer(wrap(p), add_special_tokens=True)['input_ids']) for x in data for p in prompts(x['text'],schema)]

        if max(lengths) + cfg['max_new_tokens'] > context:

            raise RuntimeError(f'全文+输出预算超长：最大输入{max(lengths)}，输出{cfg["max_new_tokens"]}，模型上限{context}；需用户确认处理方案')

        save(results / 'length_audit.json', {'max_input_tokens': max(lengths), 'context': context, 'records':300})

        def generate(prompt, seed, response_path, task, row, attempt):

            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

            inputs = tokenizer(wrap(prompt), return_tensors='pt', truncation=False).to(model.device)

            with torch.inference_mode():

                out = model.generate(**inputs, do_sample=False,

                                     repetition_penalty=cfg['repetition_penalty'],

                                     max_new_tokens=cfg['max_new_tokens'], pad_token_id=tokenizer.eos_token_id)

            ids = out[0, inputs['input_ids'].shape[1]:]

            details = {'record_id': row['record_id'], 'ann_id': row['ann_id'], 'text': row['text'],

                       'task': task, 'attempt': attempt + 1, 'prompt': prompt, 'seed': seed,

                       'repetition_penalty': cfg['repetition_penalty'],

                       'input_tokens': inputs['input_ids'].shape[1], 'generated_tokens': len(ids),

                       'max_new_tokens': cfg['max_new_tokens'], 'eos_token_id': tokenizer.eos_token_id,

                       'last_token_ids': ids[-12:].tolist(),

                       'raw_response': tokenizer.decode(ids, skip_special_tokens=True).strip(),

                       'response_with_special_tokens': tokenizer.decode(ids, skip_special_tokens=False),

                       'reached_budget_without_eos': len(ids) >= cfg['max_new_tokens'] and ids[-1].item() != tokenizer.eos_token_id,

                       'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}

            return capture_generation(response_path, details)

    started = time.monotonic()

    print(f'已恢复{len(selected)-len(pending)}条，待处理{len(pending)}条', flush=True)

    for done,(ordinal,row) in enumerate(pending,1):

        last = None

        for attempt in range(cfg['max_retries']):

            attempt_dir = response_dir / f'record_{ordinal+1:03d}' / f'attempt_{attempt+1}'

            attempt_dir.mkdir(parents=True)

            stage = 'prepare'

            try:

                prompt = prompts(row['text'],schema)[0]
                stage = 'joint_generation'
                raw = generate(prompt, cfg['seed']+ordinal*2, attempt_dir/'joint.json', 'joint', row, attempt)
                stage = 'normalization'
                result = normalize_joint(row,raw,schema)
                audit([result],[row],schema)

                with cache.open('a',encoding='utf-8',newline='\n') as f:

                    f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())

                existing[row['record_id']] = result; break

            except Exception as exc:

                last = {'record_id':row['record_id'], 'attempt':attempt+1, 'stage':stage, 'error':repr(exc), 'response_directory':str(attempt_dir)}

                save(attempt_dir / 'failure.json', last)

                print('失败/重试：', last, flush=True)

                if isinstance(exc, ValueError) or (stage == 'normalization' and isinstance(exc, RuntimeError)):

                    failures.append(last)

                    print('结构或截断错误已保存；相同种子不重复生成，转入下一条。', flush=True)

                    break

                if attempt+1 < cfg['max_retries']: time.sleep(min(30,2**attempt))

        else: failures.append(last)

        save(terminal, failures)

        save(results / (mode+'_failures.json'), failures)

        elapsed=time.monotonic()-started; total=len(selected); success=sum(x['record_id'] in existing for x in selected)

        print(f'ADELIE {success+len(failures)}/{total} 成功={success} 失败={len(failures)} 待处理={len(pending)-done} 完成={(success+len(failures))/total:.1%} 耗时={elapsed:.1f}s ETA={elapsed/done*(len(pending)-done):.1f}s',flush=True)

    failure_by_id = {x['record_id']:x for x in failures}

    missing=[x['record_id'] for x in selected if x['record_id'] not in existing and x['record_id'] not in failure_by_id]

    if missing: raise RuntimeError(f'{len(missing)}条未处理，不能冻结结果')

    pred=[]

    for row in selected:

        rid=row['record_id']

        if rid in existing:

            item={k:v for k,v in existing[rid].items() if k not in ('raw_entity','raw_relation','raw_joint')}

            item['run_status']='success'

        else:

            item={**row,'entities':[],'relations':[],'method':'ADELIE-DPO-7B-Optimized','run_status':'failed',

                  'parse_status':{'entity_json':False,'relation_json':False},'evaluation_placeholder':True,'failure':failure_by_id[rid]}

        pred.append(item)

    report=audit(pred,selected,schema)

    report.update({'processed':len(pred),'success':sum(x['run_status']=='success' for x in pred),

                   'failed':sum(x['run_status']=='failed' for x in pred),'coverage':sum(x['run_status']=='success' for x in pred)/len(pred),

                   'failure_policy':'empty evaluation placeholder with explicit failure flag; retained in full denominator',

                   'decode_condition':cfg['decode_condition'],'pilot_gate':'user_approved_processing_completion_not_zero_failures',

                   'test_adaptation_disclosure':'Optimized after inspecting baseline outputs on fixed test inputs; exploratory adapted condition, not untouched confirmatory evaluation.'})

    target=results / ('ADELIE_'+mode+'_predictions.jsonl')

    atomic_text(target, ''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pred))

    save(results / (mode+'_audit.json'), {**report, 'binding':binding, 'prediction_sha256':sha(target)})

    print('全部处理完成：',target,'成功=',report['success'],'失败=',report['failed'],'失败未标为成功。')



if __name__ == '__main__':

    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['check','pilot10','full300'])

    main(parser.parse_args().mode)

