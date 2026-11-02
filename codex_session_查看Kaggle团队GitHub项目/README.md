# Oscar 的 Codex 会话「查看Kaggle团队GitHub项目」记录与产出

来源：Oscar 本机 Codex 里的项目「IEEE Big data 2026 3」下唯一的会话「查看Kaggle团队GitHub项目」（2026-10-16 至 10-26，约 700 行记录）。Codex 里其它与比赛无关的会话（UE5/Blender、审批记录）没有收录。
完整时间线（含之后 Claude Code 上的工作）见 `docs/Oscar 若cord.md`。

**所有分数都是本地代理分，不是 Kaggle 官方分数。** Task3 的组织方边界流量、Task2 的官方 `queue_true` 从未公开。

## 内容
- `会话完整记录_2026-10-16至10-26.txt`：该会话的对话全文（已把微信账号 ID 替换为 `wxid_REDACTED`；其余本地路径保留原样）。
- `docs/`：这个会话里写的 6 份报告
  - `OSCAR_TASK1_TASK2_BASELINE.md`、`中文基线复跑总结.md`：10-17 Task1 历史均值与 Task2 persistence 基线。
  - `TASK1_TRAINING_STACK_PILOT_CN.md`：10-19 图 Transformer 训练栈试跑（1 epoch，仅验证流程通）。
  - `TASK3_Oscar_LightGBM_报告_CN.md`：Task3 诊断（含 10-20/21 复跑后的修改版）。
  - `同批样本_Task1与物理一致性对照_CN.md`：速度/流量误差、ΔN、FD 同批对照。
  - `TASK2_LightGBM_初版报告_CN.md`：10-20 Task2 LightGBM 初版。
- `results/`：对应的小型结果表（同批样本对照、Task2 窗口代理 IoU、Task3 复跑 CSV 与 provenance）。
- `scripts/`：这个会话写的脚本（历史均值、同批对照、Task2/Task3 LightGBM、Task3 本地代理评估）；含本机硬编码路径。
- `artifacts_small/`：Task2 LightGBM 候选的说明、特征重要性、代理 IoU 汇总、provenance。

## 没有放进来的
预测大文件（Task2 候选 validation/private 各 87,000 行 CSV 与概率 parquet、Task3 的 19MB 结果包、LightGBM 模型文本）、数据、`outputs/` 本地缓存都没放；它们在工作仓库 `OscarZH2005030/25-day-a-Big-Dream` 的 `oscar/task1-training-stack` 分支里（提交 `0ef9f48`、`9b3fce2`、`b1f9841`、`a27a66d`、`7a2a5ac`）。这些结果已被队友后续的 Task1 0.9428、Task2 R127 取代，只作历史记录。

## 已知限制（来自会话本身）
- 早期 smoke Transformer 的 Task3 结果已作废。
- Task2 初版代理 IoU 0.5008 只基于 19 个窗口，不能代表后续版本。
- Codex 说过没有代发微信消息，记录里的"群消息"只是草稿。
