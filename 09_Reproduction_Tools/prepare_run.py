"""Prepare a NEW inference workspace without starting inference or training.

No credentials, model weights or frozen predictions are copied as continuation state.
"""
from pathlib import Path
import argparse,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['ecke','screening','ablation','oneke','llmre','supervised','adelie','yayi']);ap.add_argument('--workdir',type=Path,required=True);a=ap.parse_args();out=a.workdir.resolve()
 if out.exists():raise SystemExit('Refusing to overwrite an existing workspace. Choose a new directory.')
 if out==ROOT or ROOT in out.parents:raise SystemExit('Use a workspace outside the frozen package.')
 out.mkdir(parents=True);(out/'08_Licensing_and_Version').mkdir();shutil.copy2(ROOT/'08_Licensing_and_Version/PACKAGE_METADATA.json',out/'08_Licensing_and_Version/PACKAGE_METADATA.json')
 def cp(folder):
  src=ROOT/folder
  def ignore(path,names):
   return [n for n in names if n in {'__pycache__','results_fixed100','results_calibrated_v1','results_optimized','evaluation_optimized','05_Evaluation_and_Verification','05_运行日志'} or n.endswith(('.pyc','.sqlite')) or (n=='04_Results' and a.stage!='screening')]
  shutil.copytree(src,out/folder,ignore=ignore)
 if a.stage in {'ecke','ablation'}:cp('02_Experiments/02_ECKE')
 if a.stage=='screening':cp('02_Experiments/01_Corpus_Preparation')
 if a.stage=='ablation':cp('02_Experiments/04_Ablation_Studies')
 if a.stage in {'oneke','llmre','yayi'}:
  for folder in ['02_Experiments/03_External_Comparisons/01_Code','02_Experiments/03_External_Comparisons/02_Input_Data','02_Experiments/03_External_Comparisons/03_Configuration_and_Instructions']:cp(folder)
 if a.stage=='llmre':
  cp('02_Experiments/03_External_Comparisons/LLMRE_Comparison')
  # Remove copied final result tree, after checking the exact new-workspace boundary.
  target=(out/'02_Experiments/03_External_Comparisons/LLMRE_Comparison/Results').resolve()
  if target.exists() and out in target.parents:shutil.rmtree(target)
 if a.stage=='supervised':
  cp('03_Baseline_Packages/00_Shared_Supervised_Code');cp('02_Experiments/03_External_Comparisons/CARE_Comparison/01_Experiment_Code')
  shutil.copytree(out/'03_Baseline_Packages/00_Shared_Supervised_Code/D2_repair_v1',out/'03_Baseline_Packages/00_Shared_Supervised_Code/AutoDL_D1-D3_Package_v2/D2_repair_v1')
 if a.stage=='adelie':cp('03_Baseline_Packages/08_ADELIE')
 entry={'ecke':'02_Experiments/02_ECKE/01_Code/01_运行V612全库知识抽取.py','screening':'02_Experiments/01_Corpus_Preparation/01_Code/03_运行消渴主题条文筛选.py','ablation':'02_Experiments/04_Ablation_Studies/01_Code/02_运行A1_去除显式词表锚定.py','oneke':'02_Experiments/03_External_Comparisons/01_Code/02_E1_OneKE-Qwen/01_运行E1_OneKE-Qwen.py','llmre':'02_Experiments/03_External_Comparisons/LLMRE_Comparison/01_运行LLMRE任务适配.py'}
 print('Prepared:',out)
 if a.stage in entry:print('After installing dependencies and filling YOUR OWN credential in the prepared script, run:\npython "'+str(out/entry[a.stage])+'"')
 else:print('See docs/ONLINE_RERUN.md for GPU model download, training, and run commands.')
 print('No API calls or training have been executed. Frozen package files were not changed.')
if __name__=='__main__':main()
