# ECKE 实验资料与复现入口

请先阅读本页，不必逐个打开全部文件。资料按论文主线组织；原始记录保留原文件名，避免与实际程序脱节。

## 建议阅读顺序

| 顺序 | 阅读位置 | 对应论文内容 |
|---|---|---|
| 1 | [实验与结果文件对应表](01_Documentation/RESULTS_TO_FILES.md) | 先确认每项结果来自哪个文件 |
| 2 | [条文构建材料](02_Experiments/01_Corpus_Preparation) | 古籍清洗、主题条文筛选、最终3065条 |
| 3 | [ECKE主实验](02_Experiments/02_ECKE) | 实体识别、关系抽取、整体性能与Oracle |
| 4 | [外部对比](02_Experiments/03_External_Comparisons) | 八种外部方法；表3 |
| 5 | [消融实验](02_Experiments/04_Ablation_Studies) | 四种消融设置；表4 |
| 6 | [结果核验](06_Verification_Results) | 先读table3.csv、table4.csv、metrics_summary.json |
| 7 | [论文图表数据](04_Figure_Data) | 仅保留性能与关系网络的底层数据，图片和PPT已移除 |

## 每个实验文件夹怎么读

按“输入数据 → 配置与运行说明 → 代码 → 结果数据 → 评价与核验”查看。优先读结果摘要和可读TXT，再按需要检查JSONL逐条预测。JSONL是复现数据，不是需要从头通读的文章。

- `03_外部方法运行包`：外部方法的服务器运行包和原始响应归档，核查细节时再打开。
- `05_固定数据索引`：100/300条固定划分及候选条文索引。
- `07_文件溯源`：文件哈希和原始来源映射，验证完整性时使用。
- `01_阅读指南`：英文数据说明、方法运行说明、投稿稿件中的实验表述。

## 离线复算

在仓库根目录运行，Python 3.10及以上，无须密钥或联网：

```text
python 09_Reproduction_Tools/verify_files.py
python 09_Reproduction_Tools/reproduce.py
```

结果写入reproduced目录，不覆盖冻结预测。这里复算的是已保存预测的评价结果；新的模型运行另见英文运行说明，不保证未来API返回完全相同的预测。

## English entry points

- [Results-to-files map](01_Documentation/RESULTS_TO_FILES.md)
- [Data dictionary](01_Documentation/DATA_DICTIONARY.md)
- [New inference/training runs](01_Documentation/ONLINE_RERUN.md)
- [Upload and verification](01_Documentation/UPLOAD_AND_VERIFY.md)

## 发布与许可

自有代码采用MIT；自有词表与标注采用CC BY 4.0。第三方权利保持不变。许可、引用与版本文件集中存放于08_许可与版本说明。

这是本地整理版，尚未执行GitHub上传或DOI归档。上传时将本目录内容放到仓库根目录，并保留目录结构。

## 许可与版本文件

- [代码许可证](08_Licensing_and_Version/LICENSE)
- [数据许可证](08_Licensing_and_Version/LICENSE-DATA.md)
- [第三方许可说明](08_Licensing_and_Version/THIRD_PARTY_NOTICES.md)
- [引用信息](08_Licensing_and_Version/CITATION.cff)
- [版本与模型元数据](08_Licensing_and_Version/PACKAGE_METADATA.json)

根目录仅保留本README。许可证与引用文件位于子目录，GitHub可能不自动展示许可或引用按钮。子目录中的.gitignore不再作为整个仓库的忽略规则；上传时请勿加入密钥、新运行缓存和临时文件。
