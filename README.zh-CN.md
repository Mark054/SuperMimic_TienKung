# SuperMimic_TienKung

[English](README.md) | 中文

面向天工 3.0（Tiangong 3.0）人形机器人的 SuperMimic 独立实现。项目包含天工机器人模型、动作数据集、动作数据转换、基于可微世界模型的训练、策略回放和评估工具。

![SuperMimic 天工 3.0 动作跟踪效果](SuperMimic_tiangong.gif)

## 环境要求

- Linux
- Python 3.12
- NVIDIA GPU 与兼容的 CUDA 驱动
- [uv](https://docs.astral.sh/uv/)

```bash
git clone <your-repository-url> SuperMimic_TienKung
cd SuperMimic_TienKung
uv sync
```

依赖锁定到项目生成的 `uv.lock`。`mjlab` 固定到 `8bb2bc9e187797689ec82c78c35019afd2fd0376`。

## 项目结构

```text
assets/tiangong3/              天工 3.0 MuJoCo 模型与网格
scripts/tiangong3/             动作转换、查看、训练结果播放和评估脚本
src/tiangong_supermimic/       训练、环境、世界模型、策略与运动学实现
data/npz/tiangong3/            天工动作数据集（NPZ）
logs/                          本地训练输出（Git 忽略）
```

## 动作数据

训练输入是天工关节顺序的 NPZ 文件，默认目录为 `data/npz/tiangong3`。

### CSV 转 NPZ

CSV 每行依次为根位置 `xyz`、根四元数 `xyzw` 和 31 个天工关节角：

```bash
uv run python scripts/tiangong3/csv_to_npz.py \
  data/raw/lafan1/tiangong3 \
  --input-fps 30 \
  --output data/npz/tiangong3
```

转换会重采样到 50 Hz，并生成训练所需的根状态、关节状态和刚体状态。

### 从 LAFAN1 重定向（可选）

`scripts/tiangong3/retarget_lafan1_gmr.py` 依赖外部 [GMR](https://github.com/YanjieZe/GMR) 工作区。通过 `GMR_ROOT` 指定路径；未设置时默认查找本仓库同级的 `GMR`：

```bash
GMR_ROOT=/path/to/GMR uv run python scripts/tiangong3/retarget_lafan1_gmr.py
```

该脚本应在安装了 GMR 依赖的环境中运行，输出默认写入 `data/raw/lafan1/tiangong3`。

### 查看参考动作

```bash
uv run python scripts/tiangong3/view_motion.py walk1_subject1
# 或传入完整文件路径
uv run python scripts/tiangong3/view_motion.py data/npz/tiangong3/walk1_subject1.npz
```

## 训练

```bash
uv run tg-sm-train
```

常用参数：

```bash
uv run tg-sm-train \
  --data.npz-dir data/npz/tiangong3 \
  --env.num-envs 4096 \
  --env.collision-mode none \
  --device cuda:0 \
  --run-name tiangong3
```

多 GPU 训练：

```bash
uv run torchrun --standalone --nproc-per-node=2 \
  -m tiangong_supermimic.scripts.train \
  --env.num-envs 512
```

检查点默认保存到 `logs/tiangong3/<run-name>/`。恢复训练：

```bash
uv run tg-sm-train --resume logs/tiangong3/<run-name>/ckpt_latest.pt
```

## 播放与评估

播放策略：

```bash
uv run python scripts/tiangong3/play.py \
  --ckpt logs/tiangong3/<run-name>/ckpt_latest.pt \
  --npz-dir data/npz/tiangong3
```

在动作集上评估：

```bash
uv run python scripts/tiangong3/eval_benchmark.py \
  --ckpt logs/tiangong3/<run-name>/ckpt_latest.pt \
  --npz-dir data/npz/tiangong3
```

## 碰撞模式

训练用 `--env.collision-mode` 切换，`play.py` / `eval_benchmark.py` 用 `--collision` 切换，默认 `none`。

| 模式 | 碰撞几何 | 自碰撞 | 24 GB 显卡参考环境数 |
|---|---|---|---|
| `none` | 简化凸包，仅与地面接触 | 关闭 | 16384 |
| `simplified` | 每个零件一个简化凸包 | 开启 | 8192 |
| `mesh` | 原始网格（按凸包处理） | 开启 | 4096 |

所有模式下，每只脚均使用两根半径 5 mm 的纵向胶囊轨道，圆角外轮廓为
270 x 104 mm；脚部网格仅用于显示，平地接触不再依赖 STL 凸包。

`filter_clips.py` 会在地面对齐前删除部分重定向 LAFAN 轨迹开头的三帧平移接缝。

自碰撞排除了相邻部件，列表见 `src/tiangong_supermimic/robot.py` 中的 `SELF_COLLISION_EXCLUDES`。

`play.py` 加 `--colliders` 可用半透明红色显示碰撞体，加 `--no-ghost` 隐藏蓝色参考动作。

## 许可

MIT，详见 [LICENSE](LICENSE)。天工模型及网格的再分发与使用还应遵循其原始来源的许可和使用条款。
