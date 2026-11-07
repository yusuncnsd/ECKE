from __future__ import annotations

import argparse, hashlib, json, random, time, os
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from d1_centreline import GridNER, RelationClassifier
from d2_d3_models import TCMERE, BSTRGPRel, dynamic_task_loss, focal_multilabel

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"05_Frozen_Data_Indices/data"; MODELS=ROOT/"models"; RESULTS=ROOT/"results"
SEEDS=(20260823,20260824,20260825)
EPOCH_OVERRIDE=None
MODEL_ROOT_OVERRIDE=None
RESULT_ROOT_OVERRIDE=None
def model_root(): return Path(MODEL_ROOT_OVERRIDE) if MODEL_ROOT_OVERRIDE else MODELS
def model_hidden(path): return AutoConfig.from_pretrained(path).hidden_size

def read_jsonl(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def atomic_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+".tmp"); q.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8"); q.replace(p)
def file_hash(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def bind_run(out,method,seed,quick,development,test):
    state={"method":method,"seed":seed,"quick":quick,"development_file":development,"test_file":test,"development_sha256":file_hash(DATA/development),"test_sha256":file_hash(DATA/test),"schema_sha256":file_hash(DATA/"xiaoke_schema_v5.json"),"code_version":"formal_v1_20260823","code_sha256":file_hash(Path(__file__)),"epochs":EPOCH_OVERRIDE}
    path=out/"run_binding.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=state: raise RuntimeError(f"运行指纹不一致，禁止续跑：{out}")
    atomic_json(path,state); return state
def char_ids(tok,text,max_len=510): return torch.tensor([[tok.cls_token_id]+tok.convert_tokens_to_ids(list(text[:max_len]))+[tok.sep_token_id]],dtype=torch.long)
def schema_assets():
    s=json.loads((DATA/"xiaoke_schema_v5.json").read_text(encoding="utf-8")); et=s["entity_types"]; rel=sorted({x["relation"] for x in s["relation_schemas"]}); legal={(x["head_type"],x["relation"],x["tail_type"]) for x in s["relation_schemas"]}; return s,et,rel,legal
def fold_entities(items): return sorted({(x["surface_form"],x["entity_type"]) for x in items})

def windows(text, size=256, overlap=64):
    if len(text)<=size: return [(0,text)]
    starts=list(range(0,max(1,len(text)-size+1),size-overlap)); last=len(text)-size
    if starts[-1]!=last: starts.append(last)
    return [(s,text[s:s+size]) for s in sorted(set(starts))]

def marker_map(types):
    # BERT-base-Chinese预留标记，固定映射并写入运行摘要。
    return {kind:{typ:f"[unused{1+k*len(types)+i}]" for i,typ in enumerate(types)} for k,kind in enumerate(("ss","se","os","oe"))}

def marked_ids(tok,text,head,tail,marks,max_len=510):
    hs,ht,ha,hb=head; ts,tt,ta,tb=tail; events={}
    for pos,val in ((ha,marks["ss"][ht]),(hb,marks["se"][ht]),(ta,marks["os"][tt]),(tb,marks["oe"][tt])): events.setdefault(pos,[]).append(val)
    tokens=[tok.cls_token]; subj=obj=None
    for i,ch in enumerate(text):
        for mark in events.get(i,[]):
            if mark==marks["ss"][ht]: subj=len(tokens)
            if mark==marks["os"][tt]: obj=len(tokens)
            tokens.append(mark)
        tokens.append(ch)
    for mark in events.get(len(text),[]): tokens.append(mark)
    tokens.append(tok.sep_token); tokens=tokens[:max_len+2]
    if subj is None or obj is None or subj>=len(tokens) or obj>=len(tokens): return None
    return torch.tensor([tok.convert_tokens_to_ids(tokens)]),torch.tensor([subj]),torch.tensor([obj])
def marked_batch(items,pad_id,device):
    width=max(x[0].size(1) for x in items); ids=torch.full((len(items),width),pad_id,dtype=torch.long)
    for i,x in enumerate(items): ids[i,:x[0].size(1)]=x[0][0]
    return ids.to(device),torch.tensor([int(x[1]) for x in items],device=device),torch.tensor([int(x[2]) for x in items],device=device)

def dict_lexicon(rows): return sorted({x["surface_form"] for row in rows for x in row["span_entities"]})
def dict_roles(text,lexicon):
    roles=[[0]*5 for _ in range(len(text)+2)]
    for word in lexicon:
        start=0
        while True:
            p=text.find(word,start)
            if p<0: break
            for q in range(p+1,p+len(word)+1): roles[q][4]=1
            if len(word)==1: roles[p+1][3]=1
            else:
                roles[p+1][0]=1; roles[p+len(word)][2]=1
                for q in range(p+2,p+len(word)): roles[q][1]=1
            start=p+1
    return roles

def append_jsonl(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(obj,ensure_ascii=False)+"\n"); f.flush(); os.fsync(f.fileno())

class FGM:
    def __init__(self,model,epsilon=1.0): self.model=model; self.epsilon=epsilon; self.backup={}
    def attack(self):
        for name,p in self.model.named_parameters():
            if p.requires_grad and "word_embeddings" in name and p.grad is not None:
                self.backup[name]=p.data.clone(); norm=torch.norm(p.grad)
                if norm: p.data.add_(self.epsilon*p.grad/norm)
    def restore(self):
        for name,p in self.model.named_parameters():
            if name in self.backup: p.data=self.backup[name]
        self.backup={}

def pos_vocab(rows):
    import jieba.posseg as pseg
    tags={w.flag for row in rows for w in pseg.cut(row["text"])}
    return {x:i+1 for i,x in enumerate(sorted(tags))}
def pos_ids(text,vocab):
    import jieba.posseg as pseg
    result=[]
    for w in pseg.cut(text): result.extend([vocab.get(w.flag,0)]*len(w.word))
    result=(result+[0]*len(text))[:len(text)]
    return [0]+result+[0]
def dynamic_mask(ids,tok,rate=.15):
    x=ids.clone(); eligible=torch.arange(1,max(1,ids.size(1)-1),device=ids.device); count=max(1,int(len(eligible)*rate)); chosen=eligible[torch.randperm(len(eligible),device=ids.device)[:count]]; x[0,chosen]=tok.mask_token_id; return x

def d1_train(seed,dev,test,out,device,quick=False):
    _,types,rels,legal=schema_assets(); tid={x:i+1 for i,x in enumerate(types)}; rid={x:i+1 for i,x in enumerate(rels)}
    model_path=model_root()/"bert-base-chinese"; tok=AutoTokenizer.from_pretrained(model_path); hidden=model_hidden(model_path)
    ner=GridNER(len(tok),len(types),hidden,str(model_path)).to(device); re_model=RelationClassifier(len(tok),len(rels),hidden,str(model_path)).to(device)
    no=torch.optim.AdamW(ner.parameters(),lr=3e-5); ro=torch.optim.AdamW(re_model.parameters(),lr=3e-5); epochs=1 if quick else (EPOCH_OVERRIDE or 12); marks=marker_map(types)
    checkpoint=out/"checkpoint.pt"; start=0
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location=device); ner.load_state_dict(c["ner"]); re_model.load_state_dict(c["re"]); no.load_state_dict(c["no"]); ro.load_state_dict(c["ro"]); start=c["epoch"]+1
    for epoch in range(start,epochs):
        random.Random(seed+epoch).shuffle(dev); ner.train(); re_model.train(); total=0
        for k,row in enumerate(dev,1):
            for offset,piece in windows(row["text"]):
                ids=char_ids(tok,piece,256).to(device); a,b=ner(ids); n=ids.size(1); y=torch.zeros((1,n,n),dtype=torch.long,device=device)
                for x in row["span_entities"]:
                    if offset<=x["start"] and x["end"]<=offset+len(piece): y[0,x["start"]-offset+1,x["end"]-offset]=tid[x["entity_type"]]
                mask=torch.triu(torch.ones(n,n,dtype=torch.bool,device=device)).unsqueeze(0); loss=nn.functional.cross_entropy(a[mask],y[mask])+nn.functional.cross_entropy(b[mask],y[mask]); no.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ner.parameters(),1); no.step(); total+=float(loss.detach())
        # RE使用条文级Gold实体；夹注排除已在转换数据中移除关系。
        for row in dev:
            entities=[(x["surface_form"],x["entity_type"],x["start"],x["end"]) for x in row["span_entities"]]; gold={(h,t):r for h,r,t in row["relations"]}
            batch=[]; labels=[]
            for head in entities:
                for tail in entities:
                    if head==tail: continue
                    packed=marked_ids(tok,row["text"],head,tail,marks)
                    if not packed: continue
                    batch.append(packed); labels.append(rid.get(gold.get((head[0],tail[0]),""),0))
                    if len(batch)==16:
                        ids,si,oi=marked_batch(batch,tok.pad_token_id,device); logits=re_model(ids,si,oi); loss=nn.functional.cross_entropy(logits,torch.tensor(labels,device=device)); ro.zero_grad(); loss.backward(); ro.step(); batch=[]; labels=[]
            if batch:
                ids,si,oi=marked_batch(batch,tok.pad_token_id,device); logits=re_model(ids,si,oi); loss=nn.functional.cross_entropy(logits,torch.tensor(labels,device=device)); ro.zero_grad(); loss.backward(); ro.step()
        torch.save({"epoch":epoch,"ner":ner.state_dict(),"re":re_model.state_dict(),"no":no.state_dict(),"ro":ro.state_dict()},checkpoint); print(f"[D1 seed={seed}] epoch {epoch+1}/{epochs} loss={total/len(dev):.4f}",flush=True)
    ner.eval(); re_model.eval(); cache=out/"prediction_cache.jsonl"; existing={x["record_id"]:x for x in read_jsonl(cache)} if cache.exists() else {}; predictions=list(existing.values())
    with torch.no_grad():
        for idx,row in enumerate(test,1):
            if row["record_id"] in existing: continue
            ents=set()
            for offset,piece in windows(row["text"]):
                ids=char_ids(tok,piece,256).to(device); a,b=ner(ids); la=a.argmax(-1)[0]; lb=b.argmax(-1)[0]; n=ids.size(1)
                for i in range(1,n-1):
                    for j in range(i,min(n-1,i+31)):
                        for z in {int(la[i,j]),int(lb[i,j])}-{0}:
                            gs,ge=offset+i-1,offset+j; ents.add((row["text"][gs:ge],types[z-1],gs,ge))
            folded={(s,t) for s,t,_,_ in ents}; triples=set(); mentions=sorted(ents); pair_items=[]
            for h in mentions:
                for t in mentions:
                    allowed=[r for ht,r,tt in legal if ht==h[1] and tt==t[1]]
                    if not allowed: continue
                    packed=marked_ids(tok,row["text"],h,t,marks)
                    if not packed: continue
                    pair_items.append((packed,h,t,allowed))
            for p in range(0,len(pair_items),32):
                chunk=pair_items[p:p+32]; mids,si,oi=marked_batch([x[0] for x in chunk],tok.pad_token_id,device); labels=re_model(mids,si,oi).argmax(-1).tolist()
                for z,(_,h,t,allowed) in zip(labels,chunk):
                    if z and rels[z-1] in allowed: triples.add((h[0],rels[z-1],t[0]))
            predictions.append({"record_id":row["record_id"],"entities":[{"surface_form":s,"entity_type":t,"start":a,"end":b} for s,t,a,b in sorted(ents)],"folded_entities":sorted(folded),"relations":sorted(triples)})
            append_jsonl(cache,predictions[-1])
            if idx%20==0: print(f"[D1推理] {idx}/{len(test)}",flush=True)
    return predictions

def span_candidates(text,max_width=15,cap=None):
    x=[(i,j) for i in range(len(text)) for j in range(i,min(len(text),i+max_width))]; return x if cap is None else x[:cap]

def d2_train(seed,dev,test,out,device,quick=False):
    _,types,rels,legal=schema_assets(); tid={x:i for i,x in enumerate(types)}; rid={x:i for i,x in enumerate(rels)}; mp=model_root()/"chinese-roberta-wwm-ext"; tok=AutoTokenizer.from_pretrained(mp); model=TCMERE(len(tok),len(types),len(rels),model_hidden(mp),15,str(mp)).to(device); opt=torch.optim.Adam(model.parameters(),lr=8e-4); epochs=1 if quick else (EPOCH_OVERRIDE or 20); lexicon=dict_lexicon(dev); atomic_json(out/"training_fold_dictionary.json",{"items":lexicon})
    checkpoint=out/"checkpoint.pt"; start=0
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location=device); model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"]); start=c["epoch"]+1
    for epoch in range(start,epochs):
        model.train(); total=0
        for row in dev:
            gold={(x["start"],x["end"]-1) for x in row["span_entities"]}; negatives=[x for x in span_candidates(row["text"]) if x not in gold][:max(10,64-len(gold))]; spans=(sorted(gold)+negatives)[:96]; ids=char_ids(tok,row["text"]).to(device); sp=torch.tensor([[(a+1,b+1) for a,b in spans]],device=device); roles=torch.tensor([dict_roles(row["text"],lexicon)[:ids.size(1)]],device=device); e,r=model(ids,roles,sp); ey=torch.zeros_like(e); ry=torch.zeros_like(r)
            emap={};
            for i,(a,b) in enumerate(spans):
                for x in row["span_entities"]:
                    if (a,b)==(x["start"],x["end"]-1): ey[0,i,tid[x["entity_type"]]]=1; emap.setdefault(x["surface_form"],[]).append(i)
            for h,rel,t in row["relations"]:
                if h in emap and t in emap:
                    for hi in emap[h]:
                        for ti in emap[t]: ry[0,hi,ti,rid[rel]]=1
            loss=nn.functional.binary_cross_entropy_with_logits(e,ey)+nn.functional.binary_cross_entropy_with_logits(r,ry); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=float(loss.detach())
        torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict()},checkpoint); print(f"[D2 seed={seed}] epoch {epoch+1}/{epochs} loss={total/len(dev):.4f}",flush=True)
    model.eval(); cache=out/"prediction_cache.jsonl"; existing={x["record_id"]:x for x in read_jsonl(cache)} if cache.exists() else {}; predictions=list(existing.values())
    with torch.no_grad():
        for ix,row in enumerate(test,1):
            if row["record_id"] in existing: continue
            allsp=span_candidates(row["text"]); scored=[]; ids=char_ids(tok,row["text"]).to(device); roles=torch.tensor([dict_roles(row["text"],lexicon)[:ids.size(1)]],device=device)
            for p in range(0,len(allsp),64):
                chunk=allsp[p:p+64]; sp=torch.tensor([[(a+1,b+1) for a,b in chunk]],device=device); e=model.score_entities(ids,roles,sp); score=torch.sigmoid(e[0]).max(-1).values
                scored.extend((float(score[i]),chunk[i]) for i in range(len(chunk)))
            selected=[x[1] for x in sorted(scored,reverse=True)[:64]]; sp=torch.tensor([[(a+1,b+1) for a,b in selected]],device=device); e,r=model(ids,roles,sp); ep=torch.sigmoid(e[0]); rp=torch.sigmoid(r[0]); ents=[]
            for i,(a,b) in enumerate(selected):
                for z in torch.where(ep[i]>=.5)[0].tolist(): ents.append((row["text"][a:b+1],types[z],a,b+1,i))
            folded={(s,t) for s,t,_,_,_ in ents}; triples=set()
            for hs,ht,_,_,hi in ents:
                for ts,tt,_,_,ti in ents:
                    for z in torch.where(rp[hi,ti]>=.5)[0].tolist():
                        if (ht,rels[z],tt) in legal: triples.add((hs,rels[z],ts))
            predictions.append({"record_id":row["record_id"],"entities":[{"surface_form":s,"entity_type":t,"start":a,"end":b} for s,t,a,b,_ in ents],"folded_entities":sorted(folded),"relations":sorted(triples)})
            append_jsonl(cache,predictions[-1])
            if ix%20==0: print(f"[D2推理] {ix}/{len(test)}",flush=True)
    return predictions

def d3_train(seed,dev,test,out,device,quick=False):
    _,types,rels,legal=schema_assets(); tid={x:i for i,x in enumerate(types)}; rid={x:i for i,x in enumerate(rels)}; mp=model_root()/"bert-base-chinese"; tok=AutoTokenizer.from_pretrained(mp); pv=pos_vocab(dev); model=BSTRGPRel(len(tok),len(types),len(rels),model_hidden(mp),str(mp),len(pv)+1).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-5); epochs=1 if quick else (EPOCH_OVERRIDE or 30); atomic_json(out/"pos_vocabulary.json",pv); fgm=FGM(model,1.0); checkpoint=out/"checkpoint.pt"; start=0
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location=device); model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"]); start=c["epoch"]+1
    for epoch in range(start,epochs):
        model.train(); total=0
        for row in dev:
            ids=char_ids(tok,row["text"]).to(device); pos=torch.tensor([pos_ids(row["text"],pv)[:ids.size(1)]],device=device); masked=dynamic_mask(ids,tok); e,h,t=model(masked,pos); ey=torch.zeros_like(e); hy=torch.zeros_like(h); ty=torch.zeros_like(t); sm={}
            for x in row["span_entities"]:
                a,b=x["start"]+1,x["end"]
                if b<ids.size(1)-1: ey[0,tid[x["entity_type"]],a,b]=1; sm.setdefault(x["surface_form"],[]).append((a,b))
            for a,r,b in row["relations"]:
                if a in sm and b in sm:
                    for x in sm[a]:
                        for y in sm[b]: hy[0,rid[r],x[0],y[0]]=1; ty[0,rid[r],x[1],y[1]]=1
            loss=dynamic_task_loss(model,[focal_multilabel(e,ey),focal_multilabel(h,hy),focal_multilabel(t,ty)]); opt.zero_grad(); loss.backward(); fgm.attack(); ae,ah,at=model(masked,pos); dynamic_task_loss(model,[focal_multilabel(ae,ey),focal_multilabel(ah,hy),focal_multilabel(at,ty)]).backward(); fgm.restore(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=float(loss.detach())
        torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict()},checkpoint); print(f"[D3 seed={seed}] epoch {epoch+1}/{epochs} loss={total/len(dev):.4f}",flush=True)
    model.eval(); cache=out/"prediction_cache.jsonl"; existing={x["record_id"]:x for x in read_jsonl(cache)} if cache.exists() else {}; predictions=list(existing.values())
    with torch.no_grad():
        for ix,row in enumerate(test,1):
            if row["record_id"] in existing: continue
            ids=char_ids(tok,row["text"]).to(device); pos=torch.tensor([pos_ids(row["text"],pv)[:ids.size(1)]],device=device); e,h,t=model(ids,pos); e=e[0]; h=h[0]; t=t[0]; ents=[]
            for z in range(len(types)):
                for a,b in torch.nonzero(e[z]>0,as_tuple=False).tolist():
                    if 1<=a<=b<ids.size(1)-1 and b-a<32: ents.append((row["text"][a-1:b],types[z],a-1,b,a,b))
            folded={(s,typ) for s,typ,_,_,_,_ in ents}; triples=set()
            for hs,ht,_,_,ha,hb in ents:
                for ts,tt,_,_,ta,tb in ents:
                    for z,rel in enumerate(rels):
                        if (ht,rel,tt) in legal and h[z,ha,ta]>0 and t[z,hb,tb]>0: triples.add((hs,rel,ts))
            predictions.append({"record_id":row["record_id"],"entities":[{"surface_form":s,"entity_type":typ,"start":a,"end":b} for s,typ,a,b,_,_ in ents],"folded_entities":sorted(folded),"relations":sorted(triples)})
            append_jsonl(cache,predictions[-1])
            if ix%20==0: print(f"[D3推理] {ix}/{len(test)}",flush=True)
    return predictions

def main():
    global EPOCH_OVERRIDE,MODEL_ROOT_OVERRIDE,RESULT_ROOT_OVERRIDE
    ap=argparse.ArgumentParser(); ap.add_argument("method",choices=["D1","D2","D3"]); ap.add_argument("--quick",action="store_true"); ap.add_argument("--development",default="development_100_span_supervision.jsonl"); ap.add_argument("--test",default="independent_test_300_input_only.jsonl"); ap.add_argument("--seed",type=int); ap.add_argument("--epochs",type=int); ap.add_argument("--result-suffix",default=""); ap.add_argument("--allow-cpu-test",action="store_true"); ap.add_argument("--model-root"); ap.add_argument("--result-root"); ap.add_argument("--data-root"); a=ap.parse_args(); EPOCH_OVERRIDE=a.epochs; MODEL_ROOT_OVERRIDE=a.model_root; RESULT_ROOT_OVERRIDE=a.result_root
    if a.data_root:
        globals()["DATA"]=Path(a.data_root)
    if not torch.cuda.is_available() and not a.allow_cpu_test: raise RuntimeError("正式入口要求CUDA")
    dev=read_jsonl(DATA/a.development); test=read_jsonl(DATA/a.test); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.set_float32_matmul_precision("high")
    if a.quick: dev=sorted(dev,key=lambda x:len(x["text"]))[:5]; test=sorted(test,key=lambda x:len(x["text"]))[:3]
    seeds=(a.seed,) if a.seed else (SEEDS[:1] if a.quick else SEEDS)
    for seed in seeds:
        random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); name="gpu_smoke" if a.quick else f"seed_{seed}"; name+=a.result_suffix; root=Path(RESULT_ROOT_OVERRIDE) if RESULT_ROOT_OVERRIDE else RESULTS; out=root/a.method/name; out.mkdir(parents=True,exist_ok=True)
        bind_run(out,a.method,seed,a.quick,a.development,a.test)
        fn={"D1":d1_train,"D2":d2_train,"D3":d3_train}[a.method]; pred=fn(seed,dev,test,out,device,a.quick)
        order={x["record_id"]:i for i,x in enumerate(test)}; pred=sorted(pred,key=lambda x:order[x["record_id"]])
        (out/"predictions_300.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in pred)+"\n",encoding="utf-8"); atomic_json(out/"run_summary.json",{"method":a.method,"seed":seed,"records":len(pred),"quick":a.quick,"complete":len(pred)==len(test)})
if __name__=="__main__": main()
