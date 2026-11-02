# Oscar 若cord：从 Codex 到 Claude 的完整工作时间线

整理日期：2026-11-01。来源：Oscar 在 Codex 上的比赛会话（2026-10-16 至 10-26，已排除与比赛无关的会话），以及之后在 Claude Code 里的全部工作（2026-10-26 至 11-01）。
**所有分数都是本地代理分，不是 Kaggle 官方分数。** 官方 `queue_true` 与 Task3 组织方边界流量从未公开，3、4 月标签不存在。

## 一、角色与规则（先说清楚）
- 比赛：2026 IEEE Big Data – Traffic Flow Bench，10 个方向，5 分钟粒度合成数据，`S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME`。
- 分工：Task1 全队共同；Task3 Oscar 主力；Task2 由 Steven 的 R127 线主做，Oscar 协助；Task4 第三位成员。
- Task3 没有提交行，直接用 Task1 的速度/流量预测评分；公开数据没有组织方边界流量，本地 `S_LWR` 对所有方案触底，本地 Task3 只能做诊断。
- 时间：Kaggle 11/6 截止；11/10 代码 + PDF（复现误差 1% 内）；奖项登记邮件 10/25（需本人核实是否完成）；内部冻结 10/30–10/31，内部提交 11/5。

## 二、时间线

### 阶段 A：Codex（10-16 至 10-26）
| 日期 | 做了什么 | 结果 / 提交 |
|---|---|---|
| 10-16 | 读 Kaggle 说明与官方基线仓库，拆解四项任务、评分和三人分工 | 明确无需部署服务，只需可复现流程 |
| 10-17 | 数据包解压（9,688 文件，约 11.5GiB，不入 Git）；跑 Task1 历史均值基线与 Task2 persistence 基线；写中文基线报告 | Task1 历史均值 `S_state` 0.730602（留出 2 月，3,443,944 条目标）；推到 `test1` 分支，提交 `0ef9f48` |
| 10-19 | 接手 Task1 训练栈：D12_I5_N 上图 Transformer 试跑；核对 GitHub 三条路线（LightGBM/GRIN/门控残差）；因切分不同不能横比 | `oscar/task1-training-stack` 提交 `9b3fce2` |
| 10-19 | Task3 核查：官方规则确认；第一次 smoke Transformer 作废；补训 LightGBM（`S_state` 0.934785，对时间插值 0.909445、历史均值 0.573799） | 提交 `b1f9841` |
| 10-20 | 按队友建议做同批样本三项对照（速度/流量误差、ΔN、FD）；D12_I5_N，1,004,145 目标 | 提交 `a27a66d`；LightGBM 直接 `S_FD` 0.9934；ΔN 不是官方 LWR 误差 |
| 10-20 | Task2 第一版 LightGBM（onset/ongoing）；19 窗口代理 IoU 0.5008，persistence 0.3302（onset 0.257，ongoing 0.745） | 提交 `7a2a5ac`；预测各 87,000 行 |
| 10-21 | 队友报告 0.9398 为当时最好；复跑 Task3（D12_I5_N LightGBM，`S_FD` 0.992333，物理代理分 0.330778）并给组员表格 | 本地代理，标明不是官方 |
| 10-23 | 给出各走廊 Transformer `S_state`（0.9105–0.9599，十走廊均值 0.9358，孤立点回退 LightGBM 后 0.9369）；T1-dynamic 6 方向交接，I10/I405 smoke 通过后开始正式训练 | Jayden 的 Dell 跑 6 个方向，我们负责补剩余方向并整合 |
| 10-26 | Codex 额度用完 | 转到 Claude Code 接手 |

（Codex 里另有与比赛无关的 UE5/Blender 会话，没有收录。）

### 阶段 B：Claude Code（10-26 至 11-01）
| 日期 | 做了什么 | 结果 / 提交 |
|---|---|---|
| 10-26 | 读取并提炼 Codex 全部历史，写接手摘要 | 摘要含待确认项（见第四节） |
| 10-26 | 发现 Task1 剩余方向已在 `25-day-a-Big-Dream-main` 跑完；整合 T1-dynamic 8→10 方向并推 GitHub | main `0718f9d`；8 方向 `A_dynamic_fixed` 0.939788 |
| 10-26 | Task3 诊断：下载 Strong v4（1.67GB，校验 SHA256）算 10 方向基本图；做守恒（ΔN）代理；尝试平滑后处理 | 基本图 FD ≈ 0.99，几乎无提升空间；本地代理 `S_LWR` 仅约 0.1；平滑使代理与 `S_state` 都变差，作为负面结果放弃；脚本与 CSV 在 `oscar/task1-training-stack` |
| 10-26 | 复现 Task2 R41（validation/private 各 87,000 行，与发布包逐字节一致）并推送 | main `fcbdd29` |
| 10-26 | 跑 Task4 官方基线（35,354 行 ×2）并推送 | main `1a0c4f8`；validation `S_link` 0.964566 |
| 10-26–29 | 看 Kaggle 页面核对冻结/时间线；分析队友 Task2 截图；队友已到 Task1 0.9428、Task2 R127、Task4 更稳版本，我们的 R41 与旧 Task4 数字随之被取代 | 见下 |
| 10-30 | 写 Task2 新增 Onset feature 方案（F1–F7 + 验证协议 + 防泄漏清单）并推送 | main `28adbfe` |
| 10-30–11-01 | 下载 R127 两个包（877MB + 1.16GB；中途一次断流，一次续传拼坏文件已删除重下，最终 SHA256/CRC 全部通过） | — |
| 11-01 | 实现 F1（到阈值距离、斜率、外推穿越时间）与 F2（邻居去噪速度比）；36 次"改未来字段特征不变"检查通过；匹配对照预筛 | 加权对数损失几乎不变，AP 变化小且方向不一致 → 无可靠增益；**Task2 停在 R127**；main `23ebdfa` |

### 当前各线最新状态
- Task1：队友最佳道路组合 `S_state` 0.942829（55 天留出，本次未重测）。
- Task2：R127（二月 367 窗口整体代理 IoU 0.7960，Onset 0.7321，Ongoing 0.8599；410 窗口 0.7546）。二月窗口多次被看过，不是独立盲测。
- Task3：仍只有诊断，没有可信官方分数；基本图无头部空间，收益只能来自更好的 Task1（尤其 ΔN 一致性）。
- Task4：本地只能算 `S_link`。

## 三、已验证无效 / 已放弃
1. Task3 平滑后处理：代理和 `S_state` 都变差。
2. Task2 F1/F2 特征：预筛无可靠增益（单种子；未做 12 月 IoU；I210 东向条件头未单独评估）。
3. 队友 R131（道路条件残差）、R133（追加精确正例）、R138（条件累计 CDF）：均未过其自设门槛。

## 四、未完成 / 需要本人确认
- 10/25 奖项登记邮件是否完成；Kaggle 每日提交次数上限。
- Task4 `S_link` 对不上：我算 0.9646，队友报 0.99994，未查清；不要引用我这个数。
- Task3 诊断仍只在 `oscar/task1-training-stack`，没进 main；需要在 0.9428 模型上重做。
- Task2 预测需要更新到 R127（我们早先推的是 R41）；整份合并提交的格式检查；这些都没有上传 Kaggle。
- 本地有意未动：Codex 仓库的未提交改动（Task3 LightGBM 报告、rerun CSV/JSON）与 `task3_shockwave.py`、`dl.log`。

## 五、关键仓库
- 工作仓库：`OscarZH2005030/25-day-a-Big-Dream`（main；分支 `test1`、`oscar/task1-training-stack`）。
- 本仓库为组装后的提交仓库；上面各提交号对应工作仓库。
