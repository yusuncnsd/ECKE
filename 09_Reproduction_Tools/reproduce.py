"""Offline reproduction of the submitted ECKE results. No network or API calls.

Python 3.10+ standard library only. Run: python 09_复现与核验程序/reproduce.py
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse, csv, hashlib, json, random, re, statistics, sys, time

ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'02_Experiments'
START=time.monotonic()
CHECKS=[]
def log(s):print(f'[{time.monotonic()-START:6.1f}s] {s}',flush=True)
def read(p):return [json.loads(s) for s in Path(p).read_text('utf-8-sig').splitlines() if s.strip()]
def obj(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def check(name,got,want):
 passed=got==want;CHECKS.append({'name':name,'passed':passed,'observed':got,'expected':want})
 if not passed:log('MISMATCH '+name+': '+str(got)+' != '+str(want))
def section(b,h,hs):
 a='|'.join(map(re.escape,hs));m=re.search(rf'(?ms)^{re.escape(h)}[ \t]*\r?\n(.*?)(?=^(?:{a})[ \t]*\r?$|^[ \t]*={{20,}}[ \t]*\r?$|\Z)',b)
 if not m:raise ValueError('Missing section '+h)
 return m.group(1).strip()
def tuples(s,n):
 if s.strip() in ('','无'):return []
 out=[]
 for line in s.splitlines():
  if not line.strip():continue
  row=[x.strip() for x in re.split(r'[|｜]',line)]
  if len(row)!=n or not all(row):raise ValueError('Malformed annotation')
  out.append(row)
 return out
def txt_records(p,kind='gold'):
 s=p.read_text('utf-8-sig');starts=list(re.finditer(r'(?m)^ann_id:\s*(?:test_|ann_)\d+\s*$',s));out=[]
 hs=('【原文】','【盲标实体】','【盲标关系】') if kind=='gold' else ('【原文】','【译文】','【预标实体】','【校对实体】','【预标关系】','【校对关系】')
 eh,rh=('【盲标实体】','【盲标关系】') if kind=='gold' else (('【校对实体】','【校对关系】') if kind=='development' else ('【预标实体】','【预标关系】'))
 for i,m in enumerate(starts):
  b=s[m.start():starts[i+1].start() if i+1<len(starts) else len(s)]
  out.append({'record_id':re.search(r'(?m)^paragraph_id:\s*(\S+)',b)[1],'ann_id':re.search(r'(?m)^ann_id:\s*(\S+)',b)[1],
   'text':section(b,'【原文】',hs),'entities':tuples(section(b,eh,hs),2),'relations':tuples(section(b,rh,hs),3)})
 return out
def normalized(rows,entity_field='entities'):
 out=[]
 for r in rows:
  es=r[entity_field];rs=r['relations']
  es=[[e['surface_form'],e['entity_type']] if isinstance(e,dict) else list(e) for e in es]
  rs=[[e['head_surface'],e['relation_type'],e['tail_surface']] if isinstance(e,dict) else list(e) for e in rs]
  out.append({'record_id':r['record_id'],'entities':es,'relations':rs,**({'text':r['text']} if 'text' in r else {})})
 return out
def prf(tp,fp,fn):
 p=tp/(tp+fp) if tp+fp else (1.0 if fn==0 else 0.0);r=tp/(tp+fn) if tp+fn else (1.0 if fp==0 else 0.0)
 return {'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.0}
def counts(p,g):return len(p&g),len(p-g),len(g-p)
def evaluate(rows,gold):
 ids=[x['record_id'] for x in rows]
 if len(ids)!=len(set(ids)) or set(ids)!=set(gold):raise ValueError('Prediction ID mismatch or duplicate')
 pm={r['record_id']:r for r in rows};out={};per={};cats={}
 for task,field in [('entity','entities'),('relation','relations')]:
  triples=[];by=defaultdict(lambda:[0,0,0])
  for rid,g in gold.items():
   p={tuple(x) for x in pm[rid][field]};t={tuple(x) for x in g[field]};c=counts(p,t);triples.append(c)
   for typ in {x[1] for x in p|t}:
    q=counts({x for x in p if x[1]==typ},{x for x in t if x[1]==typ})
    for i in range(3):by[typ][i]+=q[i]
  out[task]=prf(*(sum(x[i] for x in triples) for i in range(3)));per[task]=triples;cats[task]={k:prf(*v) for k,v in by.items()}
 return out,per,cats
def bootstrap(rows,n=5000):
 rng=random.Random(20260822);vals=[]
 for _ in range(n):
  chosen=[rows[rng.randrange(len(rows))] for _ in rows]
  vals.append(prf(*(sum(x[i] for x in chosen) for i in range(3)))['f1'])
 vals.sort();return [vals[int(.025*(n-1))],vals[int(.975*(n-1))]]
def csvwrite(p,headers,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(headers);w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'reproduced');ap.add_argument('--integrity-audit',action='store_true',help='Also inspect source offsets and write a detailed local integrity audit.');args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
 log('Reading original Gold, input passages, and frozen predictions (offline).')
 base=MAIN/'02_ECKE';ext=MAIN/'03_External_Comparisons';abl=MAIN/'04_Ablation_Studies'
 gs=txt_records(base/'02_Input_Data/Independent_Test_300/Independent_Test_300_Expert_Gold.txt');gold={g['record_id']:g for g in gs}
 dev=txt_records(base/'02_Input_Data/Development_100/Development_100_Expert_Annotations_V3.txt','development')
 corpus=read(base/'02_Input_Data/Topic_Passage_Corpus/Final_Xiaoke_Passage_Corpus.jsonl');cm={r['record_id']:r for r in corpus}
 test=read(base/'02_Input_Data/Independent_Test_300/independent_test_300_frozen.jsonl');devinput=read(base/'02_Input_Data/Development_100/method_development_100.jsonl')
 check('corpus_unique_records',len(cm),3065);check('test_gold_unique_records',len(gold),300);check('development_annotations',len(dev),100)
 check('gold_entity_count',sum(len(set(map(tuple,g['entities']))) for g in gs),869);check('gold_relation_count',sum(len(set(map(tuple,g['relations']))) for g in gs),508)
 check('development_test_overlap',sorted(set(r['record_id'] for r in devinput)&set(gold)),[])
 check('test_manifest_matches_gold',set(r['record_id'] for r in test)==set(gold),True)
 check('development_annotations_match_manifest',set(r['record_id'] for r in dev)==set(r['record_id'] for r in devinput),True)
 check('all_split_ids_in_corpus',all(r['record_id'] in cm for r in test+devinput),True)
 check('test_text_matches_corpus',all(g['text']==cm[g['record_id']]['text'] for g in gs),True)
 save(out/'development_100_gold.json',dev);save(out/'test_300_gold.json',gs)
 lex=obj(base/'03_Configuration_and_Instructions/xiaoke_lexicon_normalized_v1.json');check('complete_lexicon_concepts',len(lex['items']),680)
 schema=obj(base/'03_Configuration_and_Instructions/xiaoke_schema_v5.json');patterns={(x['head_type'],x['relation'],x['tail_type']) for x in schema['relation_schemas']}
 check('schema_patterns',len(patterns),7)
 check('gold_schema_valid',all(any((h,r,t) in patterns for s,h in g['entities'] if s==head for v,t in g['entities'] if v==tail) for g in gs for head,r,tail in g['relations']),True)
 methods={};details={};runs={};paths={}
 def add(name,files,ef='entities',txt=False):
  vals=[];rr=[]
  for f in files:
   rows=txt_records(f,'prediction') if txt else normalized(read(f),ef)
   m,p,c=evaluate(rows,gold);vals.append(m);rr.append(rows)
   if len(files)==1:details[name]={'per_record_counts':p,'by_type':c}
  methods[name]={t:{k:statistics.mean(v[t][k] for v in vals) for k in ('precision','recall','f1')} for t in ('entity','relation')}
  runs[name]=vals;paths[name]=[f.relative_to(ROOT).as_posix() for f in files]
  save(out/'normalized_predictions'/f'{name}.json',rr)
  log('Evaluated '+name+' ('+str(len(files))+' run(s)).')
 for code,name in [('D1','CENTRELINE'),('D2_repair','TCMERE'),('D3','BSTR-GPRel')]:
  fs=sorted((ext/'04_Results/D1-D3').glob(code+'_seed_*.jsonl'));check(name+'_runs',len(fs),3);add(name,fs)
 care=ext/'CARE_Comparison/01_Experiment_Code/CARE_Balanced/results_calibrated_v1'
 add('CARE',[care/f'seed_{s}/predictions.jsonl' for s in (20260823,20260824,20260825)],'folded_entities')
 add('OneKE-Qwen',[ext/'04_Results/E1/E1_OneKE_Qwen_Final_Predictions_300.jsonl'])
 add('YAYI-UIE',[ext/'04_Results/E3/E3_YAYI-UIE_300_predictions.jsonl'])
 add('LLMRE',[ext/'LLMRE_Comparison/Results/Formal_300/Predictions_300.txt'],txt=True)
 add('ADELIE',[ROOT/'03_Baseline_Packages/08_ADELIE/Experiment_Workspace/results_optimized/ADELIE_full300_predictions.jsonl'])
 add('ECKE',[base/'04_Results/Test_Predictions/Independent_Test_300_Machine_Predictions_V612.txt'],txt=True)
 structured=normalized(read(base/'04_Results/Test_Predictions/independent_test_300_machine_predictions_v612_frozen.jsonl'))
 sm,_,_=evaluate(structured,gold);check('ECKE_TXT_JSONL_metrics_identical',sm,runs['ECKE'][0])
 for code in ['A1','A2','A3','A4']:
  fs=list((abl/'04_Results').glob(f'{code}_*/*Structured_Predictions_300.jsonl'));check(code+'_prediction_files',len(fs),1);add(code,fs)
 oraclefile=base/'05_Evaluation_and_Verification/Gold_Entity_Oracle_and_Error_Propagation/oracle_relation_cache.jsonl';add('Oracle',[oraclefile])
 # Compare all table cells with the submitted manuscript, not a hand-entered expected table.
 claim=obj(ROOT/'01_Documentation/submitted_claims.json');tables=claim['tables'];t3=next(v for v in tables.values() if v[0][:2]==['Task','Method']);t4=next(v for v in tables.values() if v[0][:2]==['Task','Setting'])
 names={'w/o Lexicon Anchoring':'A1','w/o Semantic Supplementation':'A2','Simple Union':'A3','Schema-only RE':'A4'}
 for label,tab in [('Table3',t3),('Table4',t4)]:
  for row in tab[1:]:
   task='entity' if row[0]=='Entity recognition' else 'relation';name=names.get(row[1],row[1]);got=[f'{methods[name][task][k]:.4f}' for k in ('precision','recall','f1')]
   for k,a,b in zip(('P','R','F1'),got,row[2:]):check(f'{label}:{row[1]}:{task}:{k}',a,b)
 for task,expected in [('entity',['0.8869','0.9425']),('relation',['0.7332','0.8936'])]:
  log('Bootstrap '+task+': 5000 passage resamples.');ci=bootstrap(details['ECKE']['per_record_counts'][task]);details['ECKE'].setdefault('ci95_f1',{})[task]=ci
  check('ECKE_'+task+'_CI95',[f'{x:.4f}' for x in ci],expected)
 cats=details['ECKE']['by_type']
 for typ,value in [('疾病','0.9889'),('症状','0.7304'),('治法','0.7143')]:check('entity_F1_'+typ,f"{cats['entity'][typ]['f1']:.4f}",value)
 for typ,k,v in [('表现','recall','0.6721'),('表现','precision','0.8039'),('提示','recall','0.7333'),('提示','precision','0.3667'),('治疗','f1','0.9113'),('导致','f1','0.8352')]:check('relation_'+typ+'_'+k,f"{cats['relation'][typ][k]:.4f}",v)
 check('symptom_FN',cats['entity']['症状']['fn'],39);check('manifestation_FN',cats['relation']['表现']['fn'],40)
 for typ,v in [('疾病','0.9681'),('症状','0.0725')]:check('A2_entity_F1_'+typ,f"{details['A2']['by_type']['entity'][typ]['f1']:.4f}",v)
 for k,v in [('precision','0.7835'),('recall','0.9331'),('f1','0.8518')]:check('Oracle_relation_'+k,f"{methods['Oracle']['relation'][k]:.4f}",v)
 # Full-corpus counts and evidence localization, without changing any original record.
 full=read(base/'04_Results/Full_Corpus_Extraction/xiaoke_full_corpus_predictions_v612.jsonl');fm={r['record_id']:r for r in full}
 check('full_prediction_IDs_match_inputs',set(fm)==set(cm),True)
 check('frozen_test_subset_of_full',all(fm[r['record_id']]==r for r in read(base/'04_Results/Test_Predictions/independent_test_300_machine_predictions_v612_frozen.jsonl')),True)
 en=[e for r in full for e in r['entities']];rel=[e for r in full for e in r['relations']]
 fullstats={'passages':len(full),'entity_mentions':len(en),'relation_instances':len(rel),'distinct_surface_type_entities':len({(e['surface_form'],e['entity_type']) for e in en}),'distinct_surface_relations':len({(e['head_surface'],e['relation_type'],e['tail_surface']) for e in rel}),'entity_types':dict(Counter(e['entity_type'] for e in en)),'relation_labels':dict(Counter(e['relation_type'] for e in rel))}
 for k,v in [('passages',3065),('entity_mentions',8176),('relation_instances',4867),('distinct_surface_type_entities',2378),('distinct_surface_relations',3220)]:check('full_'+k,fullstats[k],v)
 check('full_Disease_mentions',fullstats['entity_types']['疾病'],3664);check('full_Formula_mentions',fullstats['entity_types']['方药'],1285);check('full_Treats_instances',fullstats['relation_labels']['治疗'],2193)
 localization=[];schemaerrors=[]
 for r in full:
  for e in r['entities']:
   if r['text'][e['start']:e['end']]!=e['surface_form']:localization.append({'record_id':r['record_id'],'entity_id':e['entity_id'],'kind':'surface_offset_mismatch'})
  for e in r['relations']:
   if (e['head_type'],e['relation_type'],e['tail_type']) not in patterns:schemaerrors.append({'record_id':r['record_id'],'relation_id':e['relation_id']})
   if r['text'][e['evidence_start']:e['evidence_end']]!=e['evidence_text']:localization.append({'record_id':r['record_id'],'relation_id':e['relation_id'],'kind':'relation_evidence_offset_mismatch'})
 if args.integrity_audit:
  check('full_entity_and_relation_offsets',len(localization),0)
  save(out/'local_integrity_audit.json',{'localization_errors':localization,'schema_errors':schemaerrors})
 check('full_relation_schema_errors',len(schemaerrors),0)
 # Endpoint error propagation uses surface presence, exactly as in the original Oracle audit.
 predmap={r['record_id']:r for r in structured};missing=present=0
 for rid,g in gold.items():
  p=predmap[rid];surfaces={e[0] for e in p['entities']}
  for h,r,t in set(map(tuple,g['relations']))-set(map(tuple,p['relations'])):
   if h not in surfaces or t not in surfaces:missing+=1
   else:present+=1
 check('relation_FN_missing_entity_endpoint',missing,82);check('relation_FN_endpoints_present',present,4)
 # Validate every displayed Top100 edge and source ID against the full frozen output.
 graphpath=ROOT/'04_Figure_Data/Top100_Relations_and_Sources.csv'
 graph=list(csv.DictReader(graphpath.open(encoding='utf-8-sig')));freq=Counter();sources=defaultdict(list)
 for row in full:
  for e in row['relations']:
   key=(e['head_surface'],e['head_type'],e['relation_type'],e['tail_surface'],e['tail_type']);freq[key]+=1;sources[key].append(row['record_id'])
 nodes=set();graphissues=[]
 for row in graph:
  key=(row['头实体'],row['头实体类型'],row['关系'],row['尾实体'],row['尾实体类型']);nodes.update([(key[0],key[1]),(key[3],key[4])])
  if freq[key]!=int(row['出现次数']) or set(sources[key])!=set(row['来源条文ID'].split(';')):graphissues.append(row['序号'])
 check('Figure3_edges',len(graph),100);check('Figure3_nodes',len(nodes),104);check('Figure3_edge_frequencies_and_source_IDs',graphissues,[])
 save(out/'Figure3_verification.json',{'nodes':len(nodes),'edges':len(graph),'incorrect_rows':graphissues,'typed_distinct_relations':len(freq),'note':'Figure uses typed relation keys; manuscript 3220 distinct relations uses surface-only endpoints. These are different deduplication keys.'})
 books=MAIN/'01_Corpus_Preparation'
 for folder in ['02_Input_Data/488_Original_Books','04_Results/488_Cleaned_Books']:check('book_count_'+folder,len(list((books/folder).glob('*.txt'))),488)
 # ADELIE terminal failures stay in the denominator.
 adelie=read(ROOT/'03_Baseline_Packages/08_ADELIE/Experiment_Workspace/results_optimized/ADELIE_full300_predictions.jsonl')
 failure=json.loads((ROOT/'03_Baseline_Packages/08_ADELIE/Experiment_Workspace/results_optimized/terminal_failures.json').read_text('utf-8-sig'))
 check('ADELIE_terminal_failures',len(failure),79)
 save(out/'ADELIE_failure_structure.json',{'type':type(failure).__name__,'entries':len(failure)})
 # Export table values and run-level metrics separately.
 for name,table in [('table3',t3),('table4',t4)]:
  rows=[]
  for r in table[1:]:
   t='entity' if r[0]=='Entity recognition' else 'relation';n=names.get(r[1],r[1]);rows.append([r[0],r[1]]+[f'{methods[n][t][k]:.4f}' for k in ('precision','recall','f1')])
  csvwrite(out/(name+'.csv'),['Task','Method','Precision','Recall','F1'],rows)
 save(out/'metrics_by_run.json',runs);save(out/'metrics_summary.json',methods);save(out/'category_results_and_CIs.json',details);save(out/'full_corpus_counts.json',fullstats)
 save(out/'prediction_sources.json',paths)
 save(out/'verification.json',{'generated_at':datetime.now(timezone.utc).isoformat(),'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':CHECKS,'passed':sum(c['passed'] for c in CHECKS),'failed':sum(not c['passed'] for c in CHECKS),'all_passed':all(c['passed'] for c in CHECKS),'table3_and_table4_passed':all(c['passed'] for c in CHECKS if c['name'].startswith(('Table3:','Table4:'))),'network_calls':0,'inference_rerun':False,'integrity_audit_requested':args.integrity_audit,'note':'Numerical reproduction of supplied frozen outputs. Candidate count is verified separately by verify_screening.py; no historical LLM decisions are regenerated.'})
 log(f"Finished: {sum(c['passed'] for c in CHECKS)}/{len(CHECKS)} checks passed. Output: {out}")
 return 0 if all(c['passed'] for c in CHECKS) else 1
if __name__=='__main__':sys.exit(main())
