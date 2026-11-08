"""复用原评价逻辑与归档Gold；只写新复核目录。"""
import importlib.util,sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parent
sys.path.insert(0,str(here))
spec=importlib.util.spec_from_file_location('archived_evaluator',here/'03_Unified_Evaluation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.GOLD=root/'Evaluation_Gold_Snapshot/Independent_Test_300_Expert_Gold.txt'
m.OUT=root/'评价复核'
if __name__=='__main__':
    m.main('ADELIE-DPO-7B-Optimized',str(root/'Experiment_Workspace/results_optimized/ADELIE_full300_predictions.jsonl'))
