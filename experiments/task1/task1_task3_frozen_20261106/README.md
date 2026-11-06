# Task1 / Task3 冻结版复现

本版对应十条道路的 2031 年 2 月本地验收。底座和逐路校正沿用十路完整重训，D7_I10_W 使用通过验收的追加校正。发布权重含 81 个文件。任务使用的数据日期来自比赛合成日历。

| 十路等权指标 | 发布版参考值 |
|---|---:|
| Task1 S-state | 0.945900260 |
| FD | 0.987981540 |
| 五分钟归一化守恒残差 | 0.370463622 |
| 五分钟守恒代理分 | 0.629536378 |
| Task3 本地代理分 | 0.749018098 |
| 五分钟宏 RMSE，辆 | 1.063188672 |
| 各路合并 RMSE 的十路均值，辆 | 1.137554961 |

Task3 本地代理使用公开参考存量差。正式计分使用主办方持有的边界流量。历史 2 月数据也用于过开发，这里报告同批公开数据的固定验收。

## 在另一台电脑上复现

先从 [Kaggle 官方数据页](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data) 下载并解压比赛数据。数据根目录是含有 `config`、`corridors` 和 `task1` 的 `kaggle_public`。下面用 Windows x64 和 Python 3.12 举例。冻结权重复现可使用 CPU。

~~~powershell
git clone --branch task1-task3-frozen-20261106 --single-branch https://github.com/chengweiliu669-pixel/25-day-a-Big-Dream-submission.git
cd 25-day-a-Big-Dream-submission
py -3.12 -m venv .venv
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$package = (Resolve-Path .\experiments\task1\task1_task3_frozen_20261106).Path
& $python -m pip install -r "$package\requirements.txt"
& $python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
$data = 'D:\datasets\kaggle_public'

& $python "$package\reproduce.py" fetch
& $python "$package\reproduce.py" reproduce --data-root $data --device cpu --output .\runs\frozen-february
~~~

`fetch` 从本仓库的 [GitHub Release](https://github.com/chengweiliu669-pixel/25-day-a-Big-Dream-submission/releases/tag/task1-task3-frozen-20261106) 下载权重。已有压缩包可加 `--archive`，也可以用 `--weights-dir` 指定单独的权重目录。`weights.json` 固定压缩包与解包后的逐文件 SHA256。Git 里保存代码、参数清单、参考指标和数据指纹。

`reproduce` 先用公开掩码和冻结权重生成十条路 28 天的预测，然后校验 900 个输入文件的 SHA256，读取 2 月标签评分。结果在 `runs/frozen-february/february-metrics.json`，含逐路结果、十路均值以及与发布参考的误差。目标键、行数和数值有效性逐日核对。平台差异的验收容差由代码逐项检查。

只生成状态预测时，可用下面的入口。`--split` 支持 `february`、`validation` 和 `private`。输出是按官方模板键排列的逐日 Parquet。

~~~powershell
& $python "$package\reproduce.py" predict --data-root $data --split validation --device cpu --output .\runs\validation
~~~

已有完整 2 月预测时，可用同一个输出目录单独运行 `evaluate`。不同运行用不同输出目录。

## 从公开数据重新训练

训练程序固定本版采用的底座路由、残差特征、校正结构和 D7_I10_W 日期折。训练与选型的日期范围是 2030 年 6 月至 2031 年 1 月，具体划分见 `training-split.json`。从随机初始化训练底座需要支持 CUDA 的 NVIDIA GPU。原运行使用 PyTorch 2.9.1 的 CUDA 12.8 wheel。训练依赖用新的虚拟环境安装。

~~~powershell
py -3.12 -m venv .venv-train
$trainPython = (Resolve-Path .\.venv-train\Scripts\python.exe).Path
& $trainPython -m pip install -r "$package\requirements-training.txt"
& $trainPython -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

& $trainPython "$package\train_frozen.py" --data-root $data --device cuda --work-dir .\runs\new-training
& $trainPython "$package\reproduce.py" reproduce --data-root $data --device cuda --trained-run .\runs\new-training --output .\runs\new-training-february
~~~

重新训练会创建独立权重和冻结清单。硬件和训练算子的差异会影响新权重，重训成绩以新生成的报告为准。只重训下游阶段时可给 `train_frozen.py` 指定 `--published-base` 为下载解包后的权重目录。`--road` 是单路诊断入口。

主要来源见 `provenance.json`。基础特征和底座模型代码保留已有团队实现，物理计算沿用比赛评分口径。
