"""LLMRE两阶段方法的消渴任务适配；只用标准库，完整300条独立运行。"""
from pathlib import Path
import concurrent.futures as cf
import hashlib,json,re,sqlite3,threading,time,urllib.request,urllib.error
API密钥 = ""
运行模式 = "Formal_300"  # 正式300 / 离线检查
ROOT=Path(__file__).resolve().parent
MODEL="qwen-plus-2025-07-28"
URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
SEED=20260813
WORKERS=5
RETRIES=6
LOCK=threading.Lock()
SCHEMA=json.loads((ROOT/"Inputs/schema.json").read_text(encoding="utf-8-sig"))
TYPES=SCHEMA["entity_types"]
SCHEMAS=SCHEMA["relation_schemas"]
PROMPTS=json.loads((ROOT/"Inputs/Prompts.json").read_text(encoding="utf-8"))
VERSION="LLMRE_method_adapted_resume_fix1_20261006"
PREVIOUS_PROGRAM_SHA256="ac632e805fc496a15523bbda6e66d47126648f1c3de9efa681de810409f0acc2"
NORMALIZATION_BATCH_SIZE=1  # 作者实现逐候选提问；线程池只并发独立请求
RELATION_BATCH_SIZE=1
REQUEST_INTERVAL=1.0  # 所有线程共用；HTTP429触发全局冷却
RATE_LOCK=threading.Lock()
NEXT_REQUEST=0.0
COOLDOWN_UNTIL=0.0
ABORT=threading.Event()
CONTEXT=threading.local()
DB=None
OUT=None

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(path)
def safe_text(value):
    value=str(value)
    if API密钥: value=value.replace(API密钥,"<REDACTED>")
    value=re.sub(r"sk-[A-Za-z0-9_-]+", "<REDACTED>", value)
    return value

def diagnostic(stage,key,attempt,exc,content):
    row={"time":time.strftime("%Y-%m-%d %H:%M:%S"),
         "record_id":getattr(CONTEXT,"record_id","连接检查"),
         "stage":stage,"request_id":key,"attempt":attempt+1,
         "exception_type":type(exc).__name__}
    if isinstance(exc,urllib.error.HTTPError):
        row["http_status"]=exc.code
        try:
            body=json.loads(exc.read().decode("utf-8",errors="replace"))
            err=body.get("error",body)
            if isinstance(err,dict):
                row["service_code"]=safe_text(err.get("code","未知"))
                row["service_message"]=safe_text(err.get("message",""))[:1000]
        except Exception:
            row["service_code"]="错误响应非JSON"
    else:
        row["message"]=safe_text(exc)[:1000]
        if content is not None:
            row["model_response"]=safe_text(content)[:8000]
    with LOCK:
        with (OUT/"Error_Diagnostics.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        print("[诊断] "+json.dumps(row,ensure_ascii=False),flush=True)

def parse_model_json(content,stage,key):
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        original_error=exc
    # 仅删除字符串外多余或类型不匹配的闭合符号；绝不补写缺失内容。
    matches={')':'(',']':'[','}':'{'}
    stack=[];output=[];in_string=False;escaped=False;removed=0
    for char in content:
        if in_string:
            output.append(char)
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':in_string=False
            continue
        if char=='"':in_string=True;output.append(char)
        elif char in '([{':stack.append(char);output.append(char)
        elif char in ')]}':
            if stack and stack[-1]==matches[char]:stack.pop();output.append(char)
            else:removed+=1
        else:output.append(char)
    if in_string or stack or not removed: raise original_error
    repaired=''.join(output)
    try:parsed=json.loads(repaired)
    except json.JSONDecodeError:raise original_error
    event={"time":time.strftime("%Y-%m-%d %H:%M:%S"),"record_id":getattr(CONTEXT,"record_id","连接检查"),
           "stage":stage,"request_id":key,"removed_closing_characters":removed,
           "raw_sha256":hashlib.sha256(content.encode()).hexdigest(),"repaired_sha256":hashlib.sha256(repaired.encode()).hexdigest()}
    with LOCK:
        with (OUT/"JSON_Format_Repair_Log.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(event,ensure_ascii=False)+"\n")
    return parsed

def evidence_filter(stage,reason,value):
    row={"time":time.strftime("%Y-%m-%d %H:%M:%S"),"record_id":getattr(CONTEXT,"record_id","未知"),
         "stage":stage,"reason":reason,"value":value}
    with LOCK:
        with (OUT/"Evidence_Filter_Log.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")

def wait_request_slot():
    global NEXT_REQUEST
    while True:
        if ABORT.is_set(): raise RuntimeError("接口不可恢复错误，已停止新增请求")
        with RATE_LOCK:
            now=time.monotonic()
            wait=max(NEXT_REQUEST,COOLDOWN_UNTIL)-now
            if wait<=0:
                NEXT_REQUEST=now+REQUEST_INTERVAL
                return
        time.sleep(min(wait,0.5))

def rate_cooldown(exc,attempt):
    global COOLDOWN_UNTIL
    delay=min(30*(2**attempt),120)
    value=exc.headers.get("Retry-After") if exc.headers else None
    if value:
        try: delay=max(delay,float(value))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                delay=max(delay,parsedate_to_datetime(value).timestamp()-time.time())
            except Exception: pass
    with RATE_LOCK:
        COOLDOWN_UNTIL=max(COOLDOWN_UNTIL,time.monotonic()+delay)
    print(f"[全局限流] HTTP429：所有新请求暂停至少{delay:.0f}秒。",flush=True)

def ask(stage,text,payload,validator):
    request={"model":MODEL,"temperature":0.0,"seed":SEED,"max_tokens":4096,
      "messages":[{"role":"system","content":PROMPTS["共同要求"]+PROMPTS[stage]},
                  {"role":"user","content":json.dumps({"原文":text,**payload},ensure_ascii=False)}]}
    raw=json.dumps(request,ensure_ascii=False).encode()
    key=hashlib.sha256(raw).hexdigest()
    with LOCK:
        old=DB.execute("select response from requests where id=?",(key,)).fetchone()
    if old and stage!="连接": return validator(json.loads(old[0]))
    for attempt in range(RETRIES):
        content=None
        try:
            wait_request_slot()
            start=time.monotonic()
            req=urllib.request.Request(URL,raw,{"Authorization":"Bearer "+API密钥,"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=180) as response: body=json.load(response)
            choice=body["choices"][0]
            if choice.get("finish_reason")=="length": raise ValueError("输出截断")
            content=choice["message"]["content"].strip()
            if content.startswith("```"): content=re.sub(r"^```(?:json)?\s*|\s*```$","",content)
            parsed=parse_model_json(content,stage,key)
            result=validator(parsed)
            with LOCK:
                DB.execute("insert or replace into requests values(?,?,?,?,?)",(key,json.dumps(parsed,ensure_ascii=False),time.monotonic()-start,json.dumps(body.get("usage",{})),stage))
                DB.commit()
            return result
        except Exception as exc:
            if isinstance(exc,urllib.error.HTTPError) and exc.code==429:
                rate_cooldown(exc,attempt)
            diagnostic(stage,key,attempt,exc,content)
            if isinstance(exc,urllib.error.HTTPError) and exc.code in (401,403,404):
                ABORT.set()
                raise RuntimeError(f"HTTP{exc.code}：停止新增请求，请检查鉴权、权限或接口配置") from None
            if ABORT.is_set(): raise RuntimeError("全局停止：保留缓存") from None
            # 只记录脱敏服务错误和模型响应，不记录密钥或请求头。
            with LOCK: print(f"[{stage}] 请求重试 {attempt+1}/{RETRIES}，异常类型 {type(exc).__name__}",flush=True)
            if attempt+1==RETRIES: raise RuntimeError(stage+"请求或结构校验失败") from None
            time.sleep(min(2**attempt,20))

def entities_valid(obj,text):
    if not isinstance(obj,dict) or not isinstance(obj.get("entities"),list): raise ValueError("缺少实体数组")
    result=[]; seen=set()
    for e in obj["entities"]:
        surface=e.get("surface_form"); typ=e.get("entity_type")
        if not isinstance(surface,str) or not surface or typ not in TYPES: raise ValueError("实体格式")
        positions=[m.start() for m in re.finditer(re.escape(surface),text)]
        if not positions:
            evidence_filter("实体","表面形式不在原文中",surface)
            continue
        # 由程序定位全部逐字提及；不通过Gold修改语义、边界或类型。
        for start in positions:
            key=(surface,typ,start)
            if key not in seen:
                seen.add(key); result.append({"surface_form":surface,"entity_type":typ,"start":start,"end":start+len(surface),"normalized_concept":e.get("normalized_concept",surface)})
    for i,e in enumerate(result): e["entity_id"]=f"e{i+1}"
    return result

def pair_groups(entities):
    unique=list({(e["surface_form"],e["entity_type"]):e for e in entities}.values())
    groups=[]
    for i,head in enumerate(unique):
        for tail in unique[i+1:]:
            if head["entity_type"]==tail["entity_type"]:
                groups.append({"pair_id":f"n{len(groups)+1}","head":head,"tail":tail})
    return unique,groups

def synonym_valid(obj,pairs):
    if not isinstance(obj,dict) or not isinstance(obj.get("positive"),list):
        raise ValueError("同义判别缺少positive数组")
    allowed={p["pair_id"]:p for p in pairs};seen=set();result=[]
    for value in obj["positive"]:
        pid=value.get("pair_id") if isinstance(value,dict) else None
        if pid not in allowed or pid in seen: raise ValueError("同义判别含未知或重复编号")
        seen.add(pid);result.append(allowed[pid])
    return result

def synonym_clusters(unique,positive):
    parent={e["entity_id"]:e["entity_id"] for e in unique}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for pair in positive:
        a=root(pair["head"]["entity_id"]);b=root(pair["tail"]["entity_id"])
        if a!=b:parent[b]=a
    groups={}
    for e in unique: groups.setdefault(root(e["entity_id"]),[]).append(e)
    clusters=[]
    for i,members in enumerate(sorted(groups.values(),key=lambda xs:min(x["start"] for x in xs)),1):
        members=sorted(members,key=lambda x:(x["start"],x["surface_form"]))
        clusters.append({"cluster_id":f"c{i}","entity_type":members[0]["entity_type"],"representative":members[0]["surface_form"],"members":members})
    return clusters

def hypernym_valid(obj,pairs):
    if not isinstance(obj,dict) or not isinstance(obj.get("parents"),list):
        raise ValueError("上下位判别缺少parents数组")
    allowed={p["pair_id"]:p for p in pairs};seen=set();result=[]
    for value in obj["parents"]:
        pid=value.get("pair_id") if isinstance(value,dict) else None
        direction=value.get("parent") if isinstance(value,dict) else None
        if pid in allowed and direction not in ("head","tail"):
            if direction==allowed[pid]["head"]["representative"]:direction="head"
            elif direction==allowed[pid]["tail"]["representative"]:direction="tail"
        if pid not in allowed or pid in seen or direction not in ("head","tail"):
            raise ValueError("上下位判别含未知编号、重复编号或无效方向")
        seen.add(pid);pair=allowed[pid]
        result.append({"parent":pair[direction]["cluster_id"],"child":pair["tail" if direction=="head" else "head"]["cluster_id"]})
    return result

def decision_valid(obj):
    """仅供最小连通检查验证固定的布尔JSON响应。"""
    if not isinstance(obj,dict) or type(obj.get("成立")) is not bool or not isinstance(obj.get("理由"),str):
        raise ValueError("连通检查响应格式错误")
    return obj

def relation_valid(obj,candidates,text):
    if not isinstance(obj,dict) or not isinstance(obj.get("positive"),list):
        raise ValueError("关系批次必须返回positive数组")
    allowed={x["candidate_id"]:x for x in candidates};result=[]
    for value in obj["positive"]:
        if not isinstance(value,dict) or value.get("candidate_id") not in allowed:
            raise ValueError("关系返回未知候选编号")
        cid=value["candidate_id"]
        # 同一编号的多个证据片段允许分条返回，最终三元组仍会去重。
        candidate=allowed[cid]
        head_names={e["surface_form"] for e in candidate["head"]["members"]}
        tail_names={e["surface_form"] for e in candidate["tail"]["members"]}
        supports=value.get("supports")
        if not isinstance(supports,list) or not supports: raise ValueError("成立关系缺少非空supports数组")
        for support in supports:
            if not isinstance(support,dict): raise ValueError("关系证据格式错误")
            evidence=support.get("evidence")
            heads=support.get("head_mentions");tails=support.get("tail_mentions")
            if not isinstance(evidence,str) or not isinstance(heads,list) or not isinstance(tails,list):
                raise ValueError("成立关系缺少原文证据或端点列表")
            if evidence not in text or not evidence.strip():
                evidence_filter("关系","证据不是逐字连续原文",{"candidate_id":cid,"evidence":evidence})
                continue
            supported_heads=sorted({x for x in heads if isinstance(x,str) and x in head_names and x in evidence})
            supported_tails=sorted({x for x in tails if isinstance(x,str) and x in tail_names and x in evidence})
            if not supported_heads or not supported_tails:
                evidence_filter("关系","证据未覆盖合法头尾端点",{"candidate_id":cid,"evidence":evidence,"head_mentions":heads,"tail_mentions":tails})
                continue
            result.append((candidate,evidence,supported_heads,supported_tails))
    return result

def process(row):
    CONTEXT.record_id=row["record_id"]
    text=row["text"]
    entities=ask("实体",text,{"实体类别":TYPES},lambda obj:entities_valid(obj,text))
    unique,pairs=pair_groups(entities)
    synonyms=[]
    if pairs:
        for offset in range(0,len(pairs),NORMALIZATION_BATCH_SIZE):
            batch=pairs[offset:offset+NORMALIZATION_BATCH_SIZE]
            compact=[{"pair_id":x["pair_id"],"head":{"surface_form":x["head"]["surface_form"],"entity_type":x["head"]["entity_type"]},"tail":{"surface_form":x["tail"]["surface_form"],"entity_type":x["tail"]["entity_type"]}} for x in batch]
            synonyms.extend(ask("同义批量",text,{"同类型候选对":compact},lambda obj:synonym_valid(obj,batch)))
    clusters=synonym_clusters(unique,synonyms)
    hypernym_pairs=[]
    for i,head in enumerate(clusters):
        for tail in clusters[i+1:]:
            if head["entity_type"]==tail["entity_type"]:
                hypernym_pairs.append({"pair_id":f"h{len(hypernym_pairs)+1}","head":head,"tail":tail})
    hypernyms=[]
    for offset in range(0,len(hypernym_pairs),NORMALIZATION_BATCH_SIZE):
        batch=hypernym_pairs[offset:offset+NORMALIZATION_BATCH_SIZE]
        compact=[{"pair_id":x["pair_id"],"head":x["head"]["representative"],"tail":x["tail"]["representative"],"entity_type":x["head"]["entity_type"]} for x in batch]
        hypernyms.extend(ask("上下位批量",text,{"不同义概念候选对":compact},lambda obj:hypernym_valid(obj,batch)))
    parent_clusters={x["parent"] for x in hypernyms}
    candidates=[]
    for schema in SCHEMAS:
        for head in clusters:
            if head["entity_type"]!=schema["head_type"]:continue
            for tail in clusters:
                if tail["entity_type"]!=schema["tail_type"] or head["cluster_id"]==tail["cluster_id"]:continue
                candidates.append({"candidate_id":f"r{len(candidates)+1}","head":head,"relation":schema["relation"],"tail":tail})
    batches=[candidates[i:i+RELATION_BATCH_SIZE] for i in range(0,len(candidates),RELATION_BATCH_SIZE)]
    logical_requests=1+(len(pairs)+NORMALIZATION_BATCH_SIZE-1)//NORMALIZATION_BATCH_SIZE+(len(hypernym_pairs)+NORMALIZATION_BATCH_SIZE-1)//NORMALIZATION_BATCH_SIZE+len(batches)
    relations=[];seen_relations=set()
    for batch in batches:
        compact=[{"candidate_id":x["candidate_id"],"head_rep":x["head"]["representative"],"head_aliases":[e["surface_form"] for e in x["head"]["members"]],"head_type":x["head"]["entity_type"],"relation":x["relation"],"tail_rep":x["tail"]["representative"],"tail_aliases":[e["surface_form"] for e in x["tail"]["members"]],"tail_type":x["tail"]["entity_type"],"scene":PROMPTS["关系场景"][x["relation"]]} for x in batch]
        positives=ask("关系批量",text,{"关系候选":compact,"上位概念簇":sorted(parent_clusters)},lambda obj:relation_valid(obj,batch,text))
        for candidate,evidence,heads,tails in positives:
            for head_surface in heads:
                for tail_surface in tails:
                    key=(head_surface,candidate["relation"],tail_surface)
                    if key in seen_relations:continue
                    seen_relations.add(key)
                    relations.append({"head_surface":head_surface,"head_type":candidate["head"]["entity_type"],"relation_type":candidate["relation"],"tail_surface":tail_surface,"tail_type":candidate["tail"]["entity_type"],"evidence_text":evidence,"head_is_hypernym":candidate["head"]["cluster_id"] in parent_clusters,"tail_is_hypernym":candidate["tail"]["cluster_id"] in parent_clusters})
    info={"synonym_pairs":[[x["head"]["entity_id"],x["tail"]["entity_id"]] for x in synonyms],"clusters":[{"cluster_id":c["cluster_id"],"entity_type":c["entity_type"],"members":[e["surface_form"] for e in c["members"]]} for c in clusters],"hypernyms":hypernyms}
    result={**row,"entities":entities,"relations":relations,"normalization":info,"candidate_count":len(candidates),"synonym_pair_count":len(pairs),"hypernym_pair_count":len(hypernym_pairs),"logical_requests":logical_requests,"status":"success"}
    with LOCK:
        DB.execute("insert or replace into records values(?,?)",(row["record_id"],json.dumps(result,ensure_ascii=False)));DB.commit()
    return result

def check():
    test=read(ROOT/"Inputs/Test_300.jsonl"); dev=read(ROOT/"Inputs/Development_100_Text_Only.jsonl")
    assert len(test)==300 and len(dev)==100
    for rows in [test,dev]:
        assert len({r["record_id"] for r in rows})==len(rows)
        assert all(isinstance(r["text"],str) and r["text"] for r in rows)
    assert not {r["record_id"] for r in test}&{r["record_id"] for r in dev}
    expected={("中药","治疗","疾病"),("方药","治疗","疾病"),("治法","主治","疾病"),("病机","导致","疾病"),("病机","涉及","病位"),("症状","提示","病机"),("疾病","表现","症状")}
    assert {(s["head_type"],s["relation"],s["tail_type"]) for s in SCHEMAS}==expected
    print("离线输入核验通过：100/300互斥，ID唯一，七种Schema有效。",flush=True)
    return test,dev

def main():
    global DB,OUT
    test,_=check()
    if 运行模式=="离线检查": return
    if 运行模式!="Formal_300": raise ValueError("运行模式错误")
    if not API密钥.strip(): raise RuntimeError("请在文件顶部手工填写API密钥；当前未调用API。")
    rows=test
    OUT=ROOT/"Results"/"Formal_300"
    OUT.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).read_text(encoding="utf-8")
    code=re.sub(r'(?m)^API密钥\s*=.*$', 'API密钥 = "<USER>"',code)
    code=re.sub(r'(?m)^运行模式\s*=.*$', '运行模式 = "<MODE>"',code)
    fingerprint={"version":VERSION,"request_interval":REQUEST_INTERVAL,"normalization_batch_size":NORMALIZATION_BATCH_SIZE,"relation_batch_size":RELATION_BATCH_SIZE,"api_url":URL,"program_sha256":hashlib.sha256(code.encode()).hexdigest(),"model":MODEL,"temperature":0.0,"seed":SEED,"workers":WORKERS,"retries":RETRIES,"schema":digest(ROOT/"Inputs/schema.json"),"prompts":digest(ROOT/"Inputs/Prompts.json"),"input":digest(ROOT/"Inputs/Test_300.jsonl"),"selected_ids":[r["record_id"] for r in rows]}
    manifest=OUT/"Run_Fingerprint.json"
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding="utf-8"))
        if saved["fingerprint"]!=fingerprint:
            previous=saved["fingerprint"]
            stable=("normalization_batch_size","relation_batch_size","api_url","model","temperature","seed","workers","retries","schema","prompts","input","selected_ids")
            eligible=(previous.get("version")=="LLMRE_method_adapted_final_20261006"
                      and previous.get("program_sha256")==PREVIOUS_PROGRAM_SHA256
                      and previous.get("request_interval")==2.0 and REQUEST_INTERVAL==1.0
                      and all(previous.get(k)==fingerprint.get(k) for k in stable))
            if not eligible: raise RuntimeError("旧缓存不是已核验的对比33原运行，禁止迁移")
            db_path=OUT/"续跑缓存.sqlite"
            backup_db=OUT/"续跑缓存_修复前备份.sqlite"
            backup_manifest=OUT/"Run_Fingerprint_Before_Repair.json"
            if not db_path.exists() or backup_db.exists() or backup_manifest.exists():
                raise RuntimeError("迁移所需缓存缺失或备份已存在，停止以避免覆盖")
            audit=sqlite3.connect(db_path)
            try:
                cached=list(audit.execute("select id,result from records"))
                request_count=audit.execute("select count(*) from requests").fetchone()[0]
                valid_ids={r["record_id"] for r in rows}
                if not cached or len(cached)>300 or request_count<len(cached) or not {rid for rid,_ in cached}.issubset(valid_ids):
                    raise RuntimeError("旧缓存数量或条文ID异常，禁止迁移")
                if any(json.loads(value).get("status")!="success" for _,value in cached):
                    raise RuntimeError("旧缓存包含非成功记录，禁止迁移")
                backup=sqlite3.connect(backup_db)
                try:audit.backup(backup)
                finally:backup.close()
            finally:audit.close()
            atomic(backup_manifest,saved)
            atomic(manifest,{"created_at":saved.get("created_at"),"fingerprint":fingerprint,
                             "migration":"仅修复模型响应格式解析及等价重复项；逐候选方法、输入、提示、模型与评价不变",
                             "cached_records_at_migration":len(cached),"cached_requests_at_migration":request_count})
            print(f"已核验并备份旧缓存：成功条文{len(cached)}、成功请求{request_count}；续跑其余条文。",flush=True)
    if not manifest.exists(): atomic(manifest,{"created_at":time.strftime("%Y-%m-%d %H:%M:%S"),"fingerprint":fingerprint})
    DB=sqlite3.connect(OUT/"续跑缓存.sqlite",check_same_thread=False)
    DB.execute("create table if not exists requests(id text primary key,response text,seconds real,usage text,stage text)")
    DB.execute("create table if not exists records(id text primary key,result text)");DB.commit()
    # 每次启动先做最小连通检查，检查失败不进入批处理。
    ask("连接","",{"检查":"只返回{\"成立\":false,\"理由\":\"连接成功\"}"},decision_valid)
    done={rid:json.loads(s) for rid,s in DB.execute("select id,result from records")}
    if not set(done).issubset({r["record_id"] for r in rows}): raise RuntimeError("缓存出现非本次清单ID")
    pending=[r for r in rows if r["record_id"] not in done]
    print(f"[续跑] 缓存成功{len(done)}条，本轮待处理{len(pending)}条；全局请求间隔{REQUEST_INTERVAL}秒。",flush=True)
    started=time.monotonic(); last_progress=started; failed=[]; failure_reasons={}; completed=0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(process,r):r for r in pending}
        while futures:
            ready,_=cf.wait(futures,timeout=15,return_when=cf.FIRST_COMPLETED)
            for f in ready:
                r=futures.pop(f);completed+=1
                try: done[r["record_id"]]=f.result()
                except Exception as exc:
                    failed.append(r["record_id"])
                    failure_reasons[r["record_id"]]=safe_text(exc)
                    print(f"[条文失败] {r['record_id']}：{safe_text(exc)}",flush=True)
            elapsed=time.monotonic()-started
            eta=elapsed/completed*(len(pending)-completed) if completed else None
            if ready or time.monotonic()-last_progress>=60:
                last_progress=time.monotonic()
                print(f"[{运行模式}] 总数{len(rows)} 完成{len(done)+len(failed)} 成功{len(done)} 失败{len(failed)} 待处理{len(futures)} 进度{(len(done)+len(failed))/len(rows):.1%} 耗时{elapsed/60:.1f}分钟 预计剩余{eta/60 if eta is not None else -1:.1f}分钟",flush=True)
    final=[done[r["record_id"]] for r in rows if r["record_id"] in done]
    atomic(OUT/"Completion_Status.json",{"complete":not failed and len(final)==len(rows),"success":len(final),"failed_ids":failed,"failed_reasons":failure_reasons,"elapsed_seconds":time.monotonic()-started,"version":VERSION,"requests":DB.execute("select count(*) from requests").fetchone()[0],"request_stats":[{"stage":stage,"count":count,"mean_seconds":seconds} for stage,count,seconds in DB.execute("select stage,count(*),avg(seconds) from requests group by stage")]})
    atomic(OUT/"Predictions.json",final)
    if not failed:
        blocks=[]
        for row in final:
            es="\n".join(sorted({e["surface_form"]+" | "+e["entity_type"] for e in row["entities"]})) or "无"
            rs="\n".join(sorted({r["head_surface"]+" | "+r["relation_type"]+" | "+r["tail_surface"] for r in row["relations"]})) or "无"
            blocks.append(f'ann_id: {row["ann_id"]}\nparagraph_id: {row["record_id"]}\n【原文】\n{row["text"]}\n【预标实体】\n{es}\n【预标关系】\n{rs}\n')
        target=OUT/"Predictions_300.txt"
        temp=target.with_suffix(".tmp");temp.write_text(("\n"+"="*80+"\n").join(blocks),encoding="utf-8");temp.replace(target)
        state=json.loads((OUT/"Completion_Status.json").read_text(encoding="utf-8"))
        state["prediction_sha256"]=digest(target)
        atomic(OUT/"Completion_Status.json",state)
    DB.close()
    print("完成。" if not failed else "存在失败，未导出正式评价TXT。原配置再次运行即可续跑。",flush=True)
    if failed: raise SystemExit(1)
if __name__=="__main__": main()
