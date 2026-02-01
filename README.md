# Cubed Sphere DG Solver (High-Performance Advection)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Backend](https://img.shields.io/badge/Backend-NumPy%20%7C%20JAX-orange)](https://github.com/google/jax)

這是一個高效能的立方體球面 (Cubed Sphere) 不連續伽略金 (Discontinuous Galerkin) 求解器。專為求解球面上的雙曲型偏微分方程 (如平流方程) 而設計。本專案採用現代軟體架構，並支援 **NumPy** 與 **JAX** 雙後端，可無縫切換 CPU 模擬與 GPU 加速運算。

## 🌟 核心特色 (Key Features)

- **高效能架構**: 支援 JAX JIT 編譯與 XLA 加速，在 GPU 上可獲得顯著效能提升。
- **高階數值方法**: 採用譜元素法 (Spectral Element Method) 與 LGL 積分點，具備指數收斂特性。
- **混合並行策略**: 針對 Apple M1/M2 Metal 與 NVIDIA GPU 最佳化的向量化運算。
- **模組化設計**: 數值核心、網格幾何與時間積分器完全解耦，易於擴充。

---

## 📦 安裝說明 (Installation)

本專案採用標準 Python 套件結構。建議在虛擬環境中安裝。

```bash
# 1. 複製專案
git clone <repo_url>
cd DG_method_on_cube_sphere/套件化1

# 2. 安裝依賴與專案 (編輯模式)
pip install -e .

# 3. (選用) 若需 GPU 加速，請安裝 JAX
# pip install "jax[cpu]"      # For CPU only
# pip install "jax[cuda12]"   # For NVIDIA GPU
# pip install "jax-metal"     # For Apple Silicon
```

---

## 🚀 快速開始 (Quick Start)

只需 5 行程式碼即可執行一個完整的球面平流模擬：

```python
from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig

# 1. 設定參數 (設定 N=32, 模擬時間 T=1.0)
config = AdvectionConfig(N=32, CFL=1.0, T_final=1.0, backend='numpy')

# 2. 初始化求解器與初始條件
solver = CubedSphereAdvectionSolver(config)
u0 = solver.get_initial_condition(type="gaussian")

# 3. 執行模擬 (自動處理時間步進)
final_state = solver.solve((0.0, 1.0), u0)
print("Simulation Complete!")
```

您可以參閱 `examples/run_advection.py` 獲得更完整的繪圖範例。

---

## ⚙️ 後端切換 (Backend Switching)

本專案核心優勢在於能夠切換運算後端。

### 1. 使用 NumPy (預設, CPU)
適合除錯、開發與小規模測試。完全基於記憶體內原地運算 (In-place operations) 優化。

```python
config = AdvectionConfig(..., backend='numpy')
```

### 2. 使用 JAX (高效能, GPU/TPU)
適合大規模高解析度模擬。利用 JIT (Just-In-Time) 編譯技術將時間迴圈融合為單一 XLA 內核。

```python
config = AdvectionConfig(..., backend='jax')
```

**注意事項 (macOS / Apple Silicon)**:
若在 Mac 上遇到 JAX `float64` 或 Metal 後端相容性問題，可透過環境變數強制使用 CPU 進行 JAX 運算：
```bash
JAX_PLATFORMS=cpu python examples/run_jax.py
```

---

## 📊 效能基準 (Benchmarks)
我們提供了自動化的基準測試腳本。詳細報告請見 [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)。

執行測試：
```bash
python benchmarks/run_benchmark.py
```

## 📂 專案結構
- `cubed_sphere/`: 核心套件原始碼
  - `numerics/`: 多項式與積分算子
  - `geometry/`: 立方體球網格生成與投影
  - `solvers/`: 時間積分器與 PDE 求解器
- `examples/`: 使用範例腳本
- `tests/`: 單元測試 (Unit Tests)

---

## License
MIT License
