# ADELIE采用结果：精简原始实验归档

本包为选定适配条件的唯一保留实验数据；非未经适配的首次验证。2026-10-06完成运行与整理。未更改任何预测、原始模型响应、评价数字、运行时间、提示或解码配置。

300条处理完成，221条结构成功、79条失败。实体Micro-F1=19.96%，关系Micro-F1=9.85%。失败带标记的空预测留在300条评价分母。

## 保留的必要材料

- Experiment_Workspace/data：冻结300条输入和7类实体/7模式Schema。
- Experiment_Workspace/results_optimized：最终预测、成功续跑缓存、全部成功与失败的原始响应、终止失败清单、运行指纹、结构审计及上下文容量记录。
- Experiment_Workspace/config_optimized.json与asset_manifest.json：参数及输入指纹。
- Experiment_Workspace/*.py及安装脚本：实际生成、下载、离线核验与原评价入口。
- 统一评价依赖及评价Gold快照：冻结评价程序及本轮评价所用答案。
- Experiment_Workspace/evaluation_optimized：完整指标、误差明细、95%区间与覆盖率。

采用ADELIE-DPO官方权重，联合实体/关系直接输出、greedy解码、repetition_penalty=1.1、768-token预算；不引入Gold示例或词表。模型权重本体不在本包；提交号与权重哈希在运行清单中。此前根据固定测试输出进行了适配，须在论文中披露测试独立性限制。

## 清理范围

删除重复的结果ZIP和上传ZIP及校验副本、旧传输/运行说明、冗余代码清单、打包辅助脚本和与terminal_failures逐字相同的full300_failures副本。原始响应和raw_cache虽含部分重复内容，分别用于审计与续跑，均保留。清理记录仅保存被删文件的哈希，不保留其内容。

## 复核与复现

解压后可执行：python Evaluation_Dependencies/运行归档评价.py。使用包内同一Gold和评价逻辑，写入新目录评价复核，不覆盖原始报告。生成脚本仍使用原工作目录结构；恢复官方模型至Experiment_Workspace/model/ADELIE-DPO和model/model_binding.json后可运行check/full300。原评价入口的历史项目路径保持原样；迁移复核使用上面的独立入口。

文件指纹清单逐项记录SHA-256；原始文件内容没有修改。外部评价TXT仅作为便于查看的摘要副本。无需提交清理前的嵌套交付ZIP。
