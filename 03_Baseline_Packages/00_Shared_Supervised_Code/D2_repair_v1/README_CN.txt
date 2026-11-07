D2 TCMERE修复重跑包 v1

修复范围仅限D2：
1. 按论文规格拆分优化器：编码器AdamW 5e-5、抽取模块Adam 8e-4、CompGCN Adam 1e-4。
2. 仅在100条开发集五折内联合选择训练轮次、实体阈值和关系阈值。
3. 若五折平均实体或关系F1仍为0，程序禁止进入正式300条运行。
4. 300条测试Gold不在云端包内，不参与任何选择。
5. 输出写入results/D2_repair_v1，不覆盖原D2、D1或D3结果。

在AutoDL的AutoDL_D1-D3_Package_v2目录中运行一个入口：
python D2_repair_v1/00_run_all_d2_repair.py

该入口按阶段门依次执行五折开发、正式运行和输出审计；任一阶段失败将立即停止，不会错误进入下一阶段。
