"""预测冻结后手动运行；复用现有评价函数，仅将输出重定向至对比2。"""

import importlib.util, sys

from pathlib import Path

ROOT=Path(__file__).resolve().parent

PROJECT=ROOT.parents[1]

COMMON=PROJECT/'02_Experiments/03_External_Comparisons/01_Code/04_Unified_Evaluation'

sys.path.insert(0,str(COMMON))

spec=importlib.util.spec_from_file_location('existing_evaluation',COMMON/'03_统一评价.py')

module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

if __name__=='__main__':

    from importlib import util

    spec2=util.spec_from_file_location('adelie_runner',ROOT/'02_run_adelie_optimized.py')

    runner=util.module_from_spec(spec2); spec2.loader.exec_module(runner)

    source,schema,_=runner.preflight()

    prediction=ROOT/'results_optimized/ADELIE_full300_predictions.jsonl'

    predictions=runner.rows(prediction)

    runner.audit(predictions,source,schema)

    if any(x.get('run_status') not in ('success','failed') for x in predictions): raise RuntimeError('缺少逐条运行状态')

    if any(x.get('run_status')=='failed' and (not x.get('evaluation_placeholder') or x['entities'] or x['relations']) for x in predictions): raise RuntimeError('失败占位不一致')

    module.OUT=ROOT/'evaluation_optimized'

    module.main('ADELIE-DPO-7B-Optimized',str(prediction))



    runner.save(module.OUT/'ADELIE_processing_coverage.json', {'total':len(predictions),'success':sum(x['run_status']=='success' for x in predictions),'failed':sum(x['run_status']=='failed' for x in predictions),'failed_retained_in_denominator':True,'decode_condition':runner.preflight()[2]['decode_condition'],'test_adaptation_disclosure':'Pilot adaptation used fixed test inputs; disclose this limitation.'})

