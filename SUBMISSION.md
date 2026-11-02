# 完整提交步骤

## 1. 取得数据与权重

从比赛官方页面按规则取得完整kaggle_public。按Task1完整复现说明下载已发布0.9428权重；Task2按horizon README下载未改动的ConditionalPosition-R127完整父链。保留experiments内原相对路径，校验各包清单。不要把原始数据或访问凭据提交到Git。

## 2. 生成Task1状态预测

在 `experiments/task1/best_road_combination_20261026` 执行：

```bash
python -X utf8 reproduction/pipeline.py submit --data-root /data/kaggle_public --work-dir /results/task1 --device cuda
```

先按其README安装对应环境、下载权重。可使用CPU，但耗时更长。预期validation/private合计6,735,795行。Task3复用这些状态值，没有独立提交字段。

## 3. 准备Task2二值队列

按 `experiments/task2/best_overall_082723_horizon_20261031/README.md` 准备父链与特征，以固定配置组合early_full Onset与horizon Ongoing。现成evaluate入口重算历史代理分，**不是完整validation/private提交导出器**。特征推理入口需要按官方队列模板映射回全部时距/节点并导出：

`panel,window_id,timestamp,link_id,queue_pred`

queue_pred必须为0/1，validation/private合计174,000行。当前整理包未凭空生成这174,000行；在拥有原始数据的运行环境执行后，才能称为完整Kaggle提交。不要用零值占位替代真实模型输出。

## 4. 准备Task4路径流量

默认使用 `experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv` 冻结基准，覆盖validation/private合计70,708行。先核对输入manifest是否与本次官方数据一致。如需重算：

```bash
cd experiments/task4/robust_odme_20261027
python reproduce.py --release-root /data/kaggle_public --stage baseline
python reproduce.py --release-root /data/kaggle_public --stage verify
```

`--stage latest`会生成v9研究候选，不能将它与冻结基准混为同一模型。其他v11等候选保留作研究，不自动替换默认选择。

## 5. 严格合并

回到库根目录：

```bash
python -X utf8 experiments/task1/best_road_combination_20261026/reproduction/merge_submission.py \
  --data-root /data/kaggle_public \
  --state-dir /results/task1/submissions \
  --queue /results/queue-validation.csv \
  --queue /results/queue-private.csv \
  --odme experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv \
  --output /results/submission.csv
```

列顺序：`submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow`。按submission_key.csv的submission_id顺序，预期总数6,980,503行。合并器拒绝缺失、重复、多余输入、非有限值、负路径流量及非二值队列。全部验证通过后，再由团队在Kaggle上传。

## 已核实与待运行

- 已核实：原历史成绩文件、三分支源码及依赖目录、三位贡献者身份、提交字段契约。
- 当前环境待完成：下载数据/大型Release权重、真实完整模型推理、Task2完整模板导出、最终合并与Kaggle评分。
- 本库是提交整理仓库，不把未生成的submission.csv声称为已完成提交。
