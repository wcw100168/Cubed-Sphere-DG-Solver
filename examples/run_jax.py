from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils import plot_cubed_sphere_state
import numpy as np

def main():
    # 1. 配置 (Backend='jax')
    # 若無安裝 JAX，會自動 Fallback 回 NumPy，但會顯示警告
    config = AdvectionConfig(
        N=24,
        R=1.0,
        u0=2 * np.pi,
        alpha0=0.0,
        CFL=1.0,
        T_final=1.0,
        backend='jax' # 啟用 JAX 加速
    )
    
    print(f"Testing Backend: {config.backend}")
    
    # 2. 初始化求解器
    # 這裡會自動進行 JIT 編譯 (首次執行可能會稍慢)
    solver = CubedSphereAdvectionSolver(config)
    
    # 3. 建立初始條件 (GPU Array if JAX)
    initial_state = solver.get_initial_condition(type="gaussian")
    
    # 4. 執行模擬
    final_state = solver.solve((0.0, config.T_final), initial_state)
    
    # 5. 把資料轉回 NumPy 以便繪圖 (若是 JAX array，這裡會自動處理)
    final_state_np = np.array(final_state)
    
    print("Optimization Complete.")
    plot_cubed_sphere_state(solver, final_state_np, title=f"JAX Advection Result")

if __name__ == "__main__":
    main()
