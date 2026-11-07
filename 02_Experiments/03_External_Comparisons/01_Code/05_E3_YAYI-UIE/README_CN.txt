E3：YAYI-UIE正式零样本复现包（2026-08-26）

论文：YAYI-UIE: A Chat-Enhanced Instruction Tuning Framework for Universal Information Extraction
官方模型：wenge-research/yayi-uie
实验定位：中文通用信息抽取LLM向中医古籍消渴Schema的严格零样本迁移。

关键边界：
1. 包内只有300条无答案输入和冻结Schema，不含测试Gold、预标答案或100条开发Gold示例。
2. 实体与关系均严格采用官方模型卡训练模板：分别保留“〖实体抽取〗”和“〖关系抽取〗”任务标记。实体一次返回7类字典，关系按冻结关系列表返回三元组；不进行目标领域参数训练。
3. 仅做结构性后处理：JSON/安全Python字面量解析、原文逐字核验、7实体/7关系Schema合法性核验；不以测试答案修正预测。
4. RTX 4090 24GB使用NF4 4bit加载13B模型；这是显存适配，不改变模型参数。

AutoDL运行顺序：
bash 00_install_environment.sh
bash 06_run_all.sh
脚本会先对10条执行结构验收（包括跨类型复制拦截）。人工检查 results/E3_YAYI-UIE_pilot10.jsonl 后：
python 03_run_e3.py full300
python 04_output_audit.py
bash 05_package_results.sh

断点续跑：同一mode重新执行会按record_id跳过已成功条目。输入、Schema或关键参数变化时，指纹门会拒绝混用旧缓存。
正式结果下载：/root/autodl-tmp/E3_YAYI_UIE_Formal_Results.zip及其SHA256文件。
