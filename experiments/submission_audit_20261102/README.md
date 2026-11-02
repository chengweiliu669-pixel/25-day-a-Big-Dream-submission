# 总分0.81：提交审计、四组训练与可控官方对照（2026-11-02）

用户报告官方总分0.81；尚未取得实际上传CSV和官方分项，不能把单个总分归因于Task2。**本轮没有宣称新官方涨分，也没有采用未通过验收的训练模型。** 原任务模型及默认horizon0.82723选择保持不变；0.82815平滑门槛仅作为原先候选另存。

## 为什么本地高分不等于总分高

官方总分=0.35×S_state+0.30×S_queue+0.15×S_physics+0.20×S_ODME。Task3的LWR占总分10%，Task4非S_link组件占总分15%，此前的高FD和接近1的S_link不能替代这25%。Task2本地0.827225→0.828149即使全部迁移，也只增加总分0.000277。Task4存在明显不可识别性：903条路径可能只有42个独立计数方程；计数拟合近1不是路径真值准确的证据。完整证据见[审计报告](reports/REPORT.md)、[任务4秩](reports/task4_identifiability.csv)、[官方评分合同](../task4/robust_odme_20261027/outputs/official_reference/docs/SCORING_SPEC.md)。

## 已完成的训练

|实验|选择阶段结果|结论|
|---|---|---|
|密度残差头，两条道路|十二月父模型S_state0.945922，最好非零密度修正约0.944551|状态精度下降；拒绝|
|速度/流量联合树修正|十二月最轻混合0.945816，低于0.945922|未通过状态改善条件；拒绝|
|时序残差CNN+积累量变化损失|无物理损失控制组仅改善S_state约0.000023、dN误差约0.11%；未达到预设1%变化量改善条件；两个物理损失候选反而退步|选择parent；不把极小开发增益当稳定提升|
|噪声感知ongoing软目标残差|十月锁定sigma.02、两个残差强度；一月仅选门槛；二月最好Overall0.825709|低于原0.827225和保留候选0.828149；拒绝|

所有原模型保留。Task1试点6—11月拟合，十二月选配置，一月确认、二月报告；仅2道路抽样日期，不冒充十道路55天成绩。Task2最好onset不动，6—11月拟合，一月只选门槛，不用二月调参。噪声模型独立256行推理与训练缓存概率误差0.0；外部初始分数、软标签梯度、时序零初始化/已知单元不变检查通过。训练代码快照见training_archive/，其中部分依赖原研究缓存，不宣称在此目录一键从零重训已验证。

## 真正可上传的文件

已在授权公开数据上运行完整模型推理，严格合并6,980,503行：state6,735,795、queue174,000、odme70,708。源码和三个Task2父链清单与本库匹配；Task4通过123个输入哈希和17份70708行预测完整键检查。队列只读已发布历史与masked-T；R127导出与原Release逐行相同，32项未来目标扰动检查通过。与R127相比，horizon两月只改变402/429个队列单格，smooth再改变71/53个，不能期待最后一点改动显著拉高总分。

[下载完整提交与对照压缩包](https://github.com/chengweiliu669-pixel/25-day-a-Big-Dream-submission/releases/tag/submission-audit-20261102)。解压CSV后上传；校验原CSV和ZIP的SHA256，见[清单](reports/download_manifest.json)。这些文件尚未获得新的官方分数，不把用户先前的0.81归到本次生成文件上。

- submission_horizon082723.csv：本库声明模型的完整重建，作为对照基准；SHA256为5b7d516ec02262628bfcc0f9c9dc5bd9e74cf1400e368d0bf664ed921244fa69。
- submission_R127_queue_only.csv：只把Task2换回R127，全部state/odme行与基准逐字节相同。若两份文件分别获得官方总分，差值÷0.30就是对应S_queue差值，能检查队列改进是否真实迁移。
- submission_task4_v9_only.csv：只把D7_I210_W的Task4换为本库既有v9保守方案，改变6642行，全部状态和队列行逐字节相同。原策略的S_dev未证实稳健提升，仅用于官方单任务诊断，不是本轮训练的新赢家。若与基准分别获得官方总分，差值÷0.20就是S_ODME差值。
- submission_smooth082815_queue_only.csv：只换此前平滑门槛候选，其他行逐字节不变。该候选旧窗口退步、主窗口CI跨零，不默认推荐。

若没有分项报告，优先核对原上传文件与基准的内容，再做单任务对照；不要同时换多个任务后把总分差全归给Task2。当前机器未配置Kaggle API，官方复评分仍需上传后取得，未在本轮消耗提交名额。

## 补齐原库缺少的Task2官方导出器

```powershell
python tools/export_task2_horizon_official.py --data-root D:/data/kaggle_public --package D:/models/ConditionalPosition-R127 --output D:/results/task2 --split both
```

依赖目录保持原相对位置，包括同级新归档的best_overall_082815_smooth_20261101。每个方案每个split87000行，先核对verification.json，再使用原严格合并器。不要用历史evaluate输出替代官方模板预测。原始比赛数据、运行环境和账号凭据未上传。新导出器已在本提交库独立运行完整validation：八道路、每个方案87000行与研究工作区逐字节一致，另16项未来标签扰动通过。

## 已有贡献者检查问题

原tools/check_contributors.py在本轮基线的两条既有提交上已失败。本轮新增提交按团队已核实账号署名，没有增加新的身份失败；原检查脚本和既有历史均未改写。完整历史检查仍不通过，不把新增提交的署名合规说成全库已通过；见reports/inherited_contributor_check.json。

## 按用户要求独立重跑官方提交准则

用户确认原上传文件不在其手上，不再等待该文件。从本库声明的模型和同机授权数据独立重建提交，直接下载并运行官方main分支的原版src/merge_submissions.py。最新脚本与已归档官方参考仅行结束符不同，下载脚本SHA256为a8cd6c4c9f2cea777fde8f6b163c00835a4fb45672998b9fc71368810e792ee1。

官方合并完整跑过6,980,503行，打印“Every scored cell was filled from your files.”，无漏键补零警告。与本轮已发布基准按submission_id、task和四个数值列逐行比较，顺序完全一致，四个数值列最大绝对差均为0.0。见[官方合并复核](reports/official_merge_recheck.json)。这确认的是本机重建文件的完整性和映射正确，不冒充对队友原上传文件的比对。

另直接运行官方Task4公开计数评分器，validation S_link=0.999939410001、private发布计数S_link=0.999923233550，均无缺失、重复、非法路径。这里的private仅指发布的private期合成计数，不是Kaggle隐藏私榜。完整S_ODME、S_queue、S_physics及隐藏总分仍需要未公开真值或官方返回；不填入估算数字。
