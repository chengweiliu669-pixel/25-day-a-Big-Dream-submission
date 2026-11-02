## 2026-11-02：总分0.81审计与完整提交对照

[分析、四组实际训练结果、完整6,980,503行提交下载](experiments/submission_audit_20261102/README.md)。补齐Task2官方模板导出器，新增单任务可控对照；本轮未发现通过验收的替换模型，也不宣称获得新官方总分。原模型和默认选择保持不变。

# 25-day-a-Big-Dream · 提交整理版

团队：[OscarZH2005030](https://github.com/OscarZH2005030) · [JaydenWang9](https://github.com/JaydenWang9) · [chengweiliu669-pixel](https://github.com/chengweiliu669-pixel)

本库整合原仓库 main、oscar/task1-training-stack、task4-reproducible-odme-20261027 分支中各任务的成绩证据、选用模型源码、复现说明和提交合并器。原仓库及其历史不作修改。本库保留选定文件的真实人类提交历史，统一三人的 GitHub 身份，并移除 AI 共同作者尾注；来源与导入策略见 [provenance/history-import.json](provenance/history-import.json)。上游许可证与文件内作者归属保留。

## 各任务最好成绩与证据

| 任务 | 目前可核实的成绩 | 口径与对应方案 |
|---|---:|---|
| Task 1 | **S_state = 0.942828631970** | 共同55天、十方向、6,636,705目标；8路Jayden186＋1路适配LightGBM＋1路原Transformer/孤立点回退 |
| Task 2 | **代理 Overall IoU = 0.827225403** | 二月新口径367窗口；early_full Onset＋六时距残差Ongoing；Onset 0.776032314、Ongoing 0.878418492 |
| Task 3 | **官方完整 S_physics 暂无** | 直接评估Task1同一份状态输出。历史十方向诊断中LightGBM的FD=0.990807391最高，A_dynamic_fixed的LWR代理=0.101851360最高，两者属于不同模型，不拼接 |
| Task 4 | **S_link = 0.999939410001** | validation公开合成计数；冻结官方目标等价求解基准。private公开计数为0.999923233550，仅为Task4的25%组件 |

以上均不能当作 Kaggle 官方隐藏测试成绩或全赛总分。Task1留出集与Task2二月数据已多轮用于开发。Task3最佳组合0.9428尚无对应完整物理分；历史FD/LWR表不代表该组合的物理成绩。Task4的S_od/S_dev/S_attr真值未公开，完整比赛分无法在本地核实。

[完整分数表 CSV](scores/best-scores.csv) · [机器可读 JSON](scores/best-scores.json) · [提交步骤](SUBMISSION.md)

### 各任务入口

- **Task 1：**[完整复现说明](experiments/task1/best_road_combination_20261026/reproduction/README_完整复现.md)、[发布权重实际复评分](experiments/task1/best_road_combination_20261026/reproduction/acceptance/published-full55-evaluation.json)。历史0.942828632692与发布权重重算0.942828631970差约7.23e-10，表中采用实际复算值。代码及权重下载器保留原相对目录。
- **Task 2：**[默认horizon方案](experiments/task2/best_overall_082723_horizon_20261031/README.md)。依赖同级0.82639父模型、0.81796 Onset与R127复现目录；本库原样保留依赖树。R127完整父链需从原Release下载。
- **Task 2其他口径：**相同horizon模型在旧410窗口为0.772054927；12—2月去重1479窗口的[shadow候选](experiments/task2/shadow_pending_dedup_20261031/README.md)为0.790893192093。二者均单独报告，不与367窗口成绩混排。
- **Task 3：**[十方向FD/LWR代理原表](experiments/task3/ten_dir_fd_lwr_proxy.csv)、[Oscar报告](docs/TASK3_Oscar_LightGBM_报告_CN.md)。单走廊三月FD=0.992333与全十方向留出诊断范围不同，不直接排名。Task3无需额外预测文件。
- **Task 4：**[完整实验与复现](experiments/task4/robust_odme_20261027/README.md)、[validation基准分](experiments/task4/robust_odme_20261027/outputs/official_validation_link_score.csv)、[数据性质更正](experiments/task4/robust_odme_20261027/DATA_VALIDATION_CORRECTION.md)。默认提交整理方案保留冻结基准；v11人工evaluation综合0.707161892290作为研究记录，不将其宣称为隐藏真值上的更好模型。

## 文件及依赖边界

原始比赛数据、凭据和运行环境不入库；Task1和R127的大型发布权重沿用原GitHub Release及其SHA256清单。历史诊断parquet数组未迁入，本库保留相关代码和成绩表；若重跑旧Task3诊断，可从原仓库取回相应数组，或重新生成。

本库提供提交代码与模型选择，未在当前Mac上执行完整训练或完整6,980,503行推理，也未向Kaggle提交。具体就绪检查见[SUBMISSION.md](SUBMISSION.md)。

## 后续贡献者保护

每次推送前运行 `python3 tools/check_contributors.py`，会检查全部可达提交的作者、提交者及共同作者尾注，要求恰好只有三人的已核实身份。本地可启用 `git config core.hooksPath .githooks` 自动拦截额外身份。新克隆需自行启用；此检查不授予任何GitHub访问权限，也不替代网站端最终核验。
