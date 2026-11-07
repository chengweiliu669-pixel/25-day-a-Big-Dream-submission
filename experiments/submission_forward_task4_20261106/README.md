# 2026-11-06 完整提交：Jayden 新 Task4（低强度正向重投影）

在 Jayden 仓库 [35ed471](https://github.com/JaydenWang9/IEEE-Compitition/commit/35ed47190f246923f80bc0ef21e9c2acfdb1bf65) 上组包，用其默认 Task4 `forward_candidate_path_flow.csv`。Task1/Task3 用 Oscar 严重缺失混合校正后的状态，Task2 用旧 0.827225 方案的队列预测。组包命令只有最后一步：

```
python -X utf8 tools/build_submission.py --data-root <kaggle_public> --state-dir <已校正状态目录> --output-dir <输出目录>
```

## 与已上传的冻结 Task4 版本对比

两份提交的状态和队列行逐行相同，只有 Task4 的 70,708 行中 4,158 行不同，总流量不变（见 [comparison_result.txt](comparison_result.txt)）。已上传的冻结 Task4 版本的公开分约 0.82，官方基线 0.81135。本版尚无官方分数，Jayden 的模拟估计总分最多约 +0.0003，公开榜上可能看不出差别。

## 文件

- [build_acceptance.json](build_acceptance.json)：本版验收，zip sha256 `b711168c6eecc993d905cc662b1c2f5331971e3d62d141426997808f3d639ec3`，共 149,857,120 字节，csv sha256 `ff017582bf6f9991072cb2627121f3f25e9efe3c9a103aae368b125e0019941c`。
- [build_acceptance_frozen_task4_uploaded.json](build_acceptance_frozen_task4_uploaded.json)：已上传版本的验收，zip sha256 `4a7b7bed…1df6`。
- zip 约 150MB，超过 GitHub 单文件 100MB 限制，没有入库，需要时用 Release 附件放置，以上 sha256 为准。
- 未上传 Kaggle 的官方评分：无。
