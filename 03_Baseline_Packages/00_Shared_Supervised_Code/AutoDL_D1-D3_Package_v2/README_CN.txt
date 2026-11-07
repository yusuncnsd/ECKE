AutoDL单卡运行顺序

推荐：RTX 4090 24GB。若4090租价明显高于3090，可选RTX 3090 24GB；A10 24GB通常更慢，不以“尽快完成”为首选。

1. 上传整个AutoDL_D1-D3实验包，不要另行上传300条专家盲标Gold。
2. 在终端进入本目录，运行：bash 00_install_environment.sh
3. 运行：python 01_preflight.py
4. 运行：python 02_download_models.py
5. 推荐直接运行：bash 06_run_all.sh
6. 该入口依次完成预检、模型冻结，随后对D1、D2、D3分别执行GPU最小验证、五折开发、3随机种子正式训练和300条预测，最后执行完整性审计。
7. 任意中断后重新运行同一命令；检查点、运行指纹和逐条预测缓存会续接，已完成阶段不会重新推断。
8. 每个方法完成后可先下载对应results目录；最终必须下载models/pretrained_model_manifest.json、results全部内容及asset_manifest.json。
9. 确认本地备份完整后停止实例。关闭浏览器不等于停止计费。

正式测试Gold不在云端包内。云端只产生300条预测，回到本地后统一评价。
