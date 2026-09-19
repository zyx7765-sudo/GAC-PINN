# GAC-PINN: Geometry-Adaptive and Constraint-Enhanced Physics-Informed Neural Networks

## 1. 项目简介

GAC-PINN 是一个面向具有陡峭梯度、强非线性和多尺度特征的物理场模拟问题的 PINN 求解框架。

该框架通过四个组件协同工作，克服了标准 PINN 的三个关键瓶颈：

| 瓶颈                            | 解决方案                                   | 对应组件                  |
| :------------------------------ | :----------------------------------------- | :------------------------ |
| 频谱偏差（高频特征收敛慢）      | 高斯傅里叶特征嵌入，重塑 NTK 频谱          | FFM                       |
| 几何不灵活性（无法自适应加密）  | 梯度驱动的微分同胚映射，等分布正则化       | AGM                       |
| 边界/初值冲突（软约束梯度竞争） | 自适应硬约束 + 算子感知路由 + 带宽门控网络 | Adaptive Hard Constraints |

### 基准模型引用

本项目对比的基线方法来源于以下工作：

| 方法         | 文献                     |
| :----------- | :----------------------- |
| Vanilla PINN | Raissi et al. (2019) [1] |
| RAR-PINN     | Mao & Meng (2023) [8]    |
| GPINN        | Yu et al. (2022) [2]     |
| Causal PINN  | Wang et al. (2024) [3]   |
| RAMS-PINN    | Ouyang et al. (2026) [4] |

### 代码声明

本项目基于 **gPINN**（Yu et al., 2022）的开源代码进行二次开发，PDE 设置、基准数据集和对比基线均与 gPINN 保持一致。gPINN 官方仓库地址：https://github.com/lu-group/gpin

在本项目中：
- 三个基准问题的 PDE 设置、初始/边界条件、参考解均与 gPINN 完全相同；
- 所使用的 `Burgers.npz` 和 `usol_D_0.001_k_5.mat` 数据集直接来源于 gPINN 的官方仓库；
- 核心创新（AGM、FFM、自适应硬约束、算子感知路由）为本工作的独立贡献。

## 2. 基准实验结果

以下结果为 5 个独立随机种子（1234, 5678, 9012, 3456, 7890）下的均值 ± 标准差：

| 问题                                      | 相对 L² 误差                     |
| :---------------------------------------- | :------------------------------- |
| 1D Burgers 方程（粘性冲击波，ν = 0.01/π） | **(1.747 ± 0.450) × 10⁻⁴**      |
| 2D Poisson 方程（a = 10 极端尖峰）        | **(2.868 ± 0.947) × 10⁻⁵**      |
| 1D Allen-Cahn 方程（ε = 0.001 尖锐相场）  | **(1.756 ± 0.712) × 10⁻³**      |
| 2D Navier-Stokes 方程（圆柱绕流，压力场） | **2.66 × 10⁻²**                 |

在 Burgers 和 Poisson 问题上，GAC-PINN 的精度分别比最优基线高出约 62 倍和 18 倍（基于平均相对 L² 误差）。

## 3. 运行环境配置

### 硬件要求
- **GPU**：建议显存 ≥ 8GB（RTX 2080 Ti / RTX 3070 / Tesla V100 或同等配置）
- **系统**：Linux / Windows / macOS（已测试 Ubuntu 20.04）

### 创建 Conda 环境
```bash
conda create -n gac_pinn python=3.9
conda activate gac_pinn
```

### 安装依赖
```bash
git clone https://github.com/[你的用户名]/[仓库名].git
cd [仓库名]
pip install -r requirements.txt
```

本项目使用 **PyTorch 后端**，已在代码中通过 `os.environ["DDE_BACKEND"] = "pytorch"` 强制指定。请确保 PyTorch 版本与 CUDA 兼容。

## 4. 数据准备

### 1D Burgers 方程
需要 `Burgers.npz` 文件，包含 `t`、`x`、`usol` 三个字段，放置于 `dataset/` 目录下。

- 下载：[DeepXDE 官方 Burgers 数据集](https://github.com/lululxvi/deepxde/blob/master/examples/dataset/Burgers.npz)

### 1D Allen-Cahn 方程
需要 `usol_D_0.001_k_5.mat` 文件（MATLAB 格式），包含 `u`、`x`、`t` 三个变量，放置于 `dataset/` 目录下。

- 来源：[gPINN 官方仓库](https://github.com/lu-group/gpin) 的基准数据集

### 2D Poisson 方程
解析解已知，无需额外数据文件。

## 5. 快速复现

### 选择运行的方程

打开 `main.py`，修改配置开关：

```python
# ==============================================================================
# 用户全局运行配置开关
# ==============================================================================
CURRENT_EQUATION = "1D_Burgers"        # 可选: "1D_Burgers" / "2D_Poisson" / "1D_Allen_Cahn"
POISSON_A_PARAM = 10                  
EPOCHS_PHASE1 = 12000                  # Phase 1 训练步数
```

### 运行训练
```bash
python main.py
```

### 预期输出

训练完成后，控制台输出性能报告（以 Burgers 为例）：

```
============================================================
      ACADEMIC PERFORMANCE REPORT (UNIFIED FRAMEWORK)
============================================================
算子感知路由判定 : 时间依赖型 (time-dependent)
总计物理优化步数   : 17000 steps
最终相对 L2 误差 : 1.747e-04
============================================================
```

同时生成误差分布图：`error_distribution_spatial_hardness.png`（Burgers）或 `allencahn_unified_error.png`（Allen-Cahn）。

## 6. 项目文件结构

```
根目录/
├── README.md                    # 说明文档
├── requirements.txt             # 依赖清单
├── LICENSE                      # MIT License
├── main.py                      # 主训练入口
├── models.py                    # 网络架构（AGM、FFM、自适应硬约束等）
├── pde.py                       # PDE 定义与硬约束绑定
├── dataset/                     # 数据文件
│   ├── Burgers.npz
│   └── usol_D_0.001_k_5.mat
└── output/                      # 运行时输出
    ├── error_distribution_spatial_hardness.png
    └── ... 
```

## 7. 许可证

本项目采用 [MIT License](LICENSE) 开源。
