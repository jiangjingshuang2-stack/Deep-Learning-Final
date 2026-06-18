# 深度学习与空间智能期末作业第一个任务

本仓库对应题目一：基于 3DGS 与 AIGC 的多源资产生成与真实场景融合。任务完成了真实物体 A 的 COLMAP + 3DGS 重建、文本生成物体 B、单图生成物体 C、Mip-NeRF 360 garden 背景重建，以及 ABC 与 garden 的世界静止融合渲染。

## 目录结构

```text
.
├── README.md
├── environment.yml
├── 模型权重说明.md
├── code/
│   ├── build_garden_a_table_static_v2.py
│   ├── composite_bc_world_static_v2.py
│   ├── debug_project_world_candidates.py
│   └── debug_select_table_from_garden.py
├── docs/
│   └── HW3_深度学习与空间智能.pdf
├── logs/
│   ├── objectA_3dgs_train_7000.log
│   ├── garden_3dgs_train_30k_images2_0611.log
│   ├── objectB_dreamfusion_sd15_gpu3_5k_v3.log
│   ├── objectC_sunscreen_zero123_train_0611_gpu0.log
│   └── parsed_loss_summary.txt
├── models/
│   ├── objectA_3dgs_iteration_7000/
│   ├── objectB_threestudio_it5000/
│   └── objectC_zero123_it600/
├── videos/
│   ├── objectA_3DGS_orbit.mp4
│   ├── objectB_threestudio_orbit.mp4
│   ├── objectC_zero123_orbit.mp4
│   ├── background_garden_3DGS_orbit.mp4
│   └── final_ABC_garden_world_static_v2.mp4
└── 报告/
    ├── report_cn.tex
    ├── report_en.tex
    └── figures/
```

## 环境配置

服务器环境：`jjs_server_2`，训练和渲染均在已有环境中完成。3DGS 使用官方 3D Gaussian Splatting 代码环境，threestudio 使用独立 conda 环境。提交目录提供 `environment.yml` 作为复现实验的统一依赖参考。

```bash
conda env create -f environment.yml
conda activate hw3-task1-3dgs-aigc
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
wandb login
```

实际服务器上的主要环境和路径：

```text
3DGS: /root/HW3_task1/third_party/gaussian-splatting
threestudio: /root/HW3_task1/third_party/threestudio
COLMAP 数据: /root/HW3_task1/data/object_A_rot90_upload
garden 数据: /root/HW3_task1/data/background/garden
```

## 数据准备

Object A 使用手机/相机环绕拍摄的真实物体图像。原始抠图图片先统一顺时针旋转 90 度，再上传到服务器进行 COLMAP 特征提取、匹配和稀疏重建。最终使用 relaxed full-frame pinhole 设置获得 71 张有效位姿。

Object B 使用文本 prompt 生成：

```text
a small blue ceramic dragon statue, detailed scales, glossy surface, studio lighting
```

中文含义：一个小型蓝色陶瓷龙雕像，具有细致的鳞片、光滑有光泽的表面，工作室灯光。

Object C 使用单张防晒霜图片，去背景后通过 Zero123 进行单图到 3D 生成。背景场景选择 Mip-NeRF 360 数据集中的 `garden`。

## 训练命令

Object A 位姿提取和 3DGS 训练：

```bash
cd /root/HW3_task1/third_party/gaussian-splatting
CUDA_VISIBLE_DEVICES=2 python convert.py \
  -s /root/HW3_task1/data/objectA_rot90_recon_0612_relaxed_fullframe_pinhole

CUDA_VISIBLE_DEVICES=2 python train.py \
  -s /root/HW3_task1/data/objectA_rot90_recon_0612_relaxed_fullframe_pinhole \
  -m /root/HW3_task1/data/objectA_rot90_recon_0612_relaxed_fullframe_pinhole/gs_eval_7000 \
  --iterations 7000 \
  --test_iterations 7000 \
  --save_iterations 7000
```

Object B 文本到 3D：

```bash
cd /root/HW3_task1/third_party/threestudio
CUDA_VISIBLE_DEVICES=3 python launch.py \
  --config configs/dreamfusion-sd.yaml \
  --train --gpu 0 \
  system.prompt_processor.prompt="a small blue ceramic dragon statue, detailed scales, glossy surface, studio lighting" \
  trainer.max_steps=5000 \
  name=objectB_dreamfusion_sd15_gpu3_5k_v3
```

Object C 单图到 3D：

```bash
cd /root/HW3_task1/third_party/threestudio
CUDA_VISIBLE_DEVICES=0 python launch.py \
  --config configs/zero123.yaml \
  --train --gpu 0 \
  data.image_path=/root/HW3_task1/data/object_C/IMG_7047.png \
  trainer.max_steps=600 \
  name=objectC_sunscreen_zero123_0611_gpu0
```

garden 背景 3DGS 训练：

```bash
cd /root/HW3_task1/third_party/gaussian-splatting
CUDA_VISIBLE_DEVICES=2 python train.py \
  -s /root/HW3_task1/data/background/garden \
  -m /root/HW3_task1/data/background/gs/garden \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000
```

## 渲染与融合

最终采用世界静止融合：ABC 与 garden 固定在同一世界坐标系中，渲染时相机环绕，物体与背景保持相对静止。A 和 garden 均为显式 3D Gaussian，B/C 为 threestudio 导出的 mesh。为了统一渲染，先将 A 的高斯点云过滤、缩放并并入 garden 高斯背景，再用官方 3DGS 渲染得到 `garden + A` 帧；随后将 B/C mesh 归一化、坐标系转换并投影到相同相机轨迹上进行带深度的 mesh 合成。

```bash
python code/build_garden_a_table_static_v2.py
python code/composite_bc_world_static_v2.py
```

关键产出：

```text
videos/final_ABC_garden_world_static_v2.mp4
```

## 提交链接

- GitHub 仓库：<https://github.com/jiangjingshuang2-stack/Deep-Learning-Final>
- 模型权重网盘：<https://drive.google.com/drive/folders/154-pIcfsTkYXpTeUgTcfOjBJ5Sep0xIy?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto>

## W&B 与日志

训练过程已登录 W&B 并记录。已知 run 信息：

| 实验 | W&B 项目/Run |
|---|---|
| Object A 3DGS | `HW3-task1-3dgs / 68pj7ckl` |
| Object B threestudio | `HW3-task1-threestudio-objectB / 7txhb0k5` |
| Object C Zero123 | `HW3-task1-zero123 / 4azkmnu1` |
| garden 3DGS | 服务器 W&B/offline 日志位于 `/root/HW3_task1/data/background/gs/garden*/wandb` |

本交付目录同时保存了训练控制台日志。`logs/parsed_loss_summary.txt` 是从日志解析得到的曲线摘要。Object B 的控制台日志主要保留 tqdm 进度和 prompt，标量 loss 以 W&B 页面为准。

## 结果文件

| 文件 | 内容 |
|---|---|
| `videos/objectA_3DGS_orbit.mp4` | Object A 单独 3DGS 渲染 |
| `videos/objectB_threestudio_orbit.mp4` | Object B 文本到 3D 单独渲染 |
| `videos/objectC_zero123_orbit.mp4` | Object C 单图到 3D 单独渲染 |
| `videos/background_garden_3DGS_orbit.mp4` | garden 背景单独 3DGS 渲染 |
| `videos/final_ABC_garden_world_static_v2.mp4` | 最终 ABC + garden 世界静止融合渲染 |

## 模型权重

A/B/C 的模型文件已放入 `models/`。garden 3DGS 权重约 1.1GB，未直接复制进交付目录，服务器路径和 Google Drive 权重链接见 `模型权重说明.md`。报告首页已写入 GitHub 仓库和模型权重链接。
