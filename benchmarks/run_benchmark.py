import time
import pandas as pd
import sys
import os

# Enable JAX x64 and Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"
try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

# 確保可以 import cubed_sphere 套件
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig

def run_benchmark_suite():
    # 增加 128 以展示 GPU 優勢
    n_values = [32, 64, 128, 256]
    results = []

    print("========================================")
    print("     CUBED SPHERE BENCHMARK SUITE       ")
    print("========================================")

    for N in n_values:
        print(f"\n--- Testing N={N} ---")
        
        # 固定模擬時間，讓運算量隨 N 增加
        T_final = 0.05 
        
        row = {'N': N}
        
        for backend in ['numpy', 'jax']:
            try:
                print(f"Running {backend}...", end=" ", flush=True)
                
                # 設定配置
                config = AdvectionConfig(N=N, CFL=1.0, T_final=T_final, backend=backend)
                solver = CubedSphereAdvectionSolver(config)
                u0 = solver.get_initial_condition("gaussian")
                
                # 計時開始
                start_time = time.time()
                
                # 執行模擬 (包含 JAX 的 JIT 編譯時間，模擬真實使用情境)
                solver.solve((0.0, T_final), u0)
                
                end_time = time.time()
                duration = end_time - start_time
                
                print(f"Done! ({duration:.4f}s)")
                row[f'{backend}_time'] = duration
            except Exception as e:
                print(f"FAILED: {e}")
                row[f'{backend}_time'] = float('nan')
        
        # 計算加速倍率 (Speedup = CPU / GPU)
        if row.get('jax_time') and row['jax_time'] > 0:
            speedup = row['numpy_time'] / row['jax_time']
        else:
            speedup = 0.0
            
        row['speedup'] = speedup
        results.append(row)

    # 建立與顯示表格
    df = pd.DataFrame(results)
    
    # 格式化欄位
    df = df.rename(columns={
        'numpy_time': 'NumPy (s)',
        'jax_time': 'JAX (s)',
        'speedup': 'Speedup (x)'
    })
    
    print("\n\n========================================")
    print("           FINAL COMPARISON             ")
    print("========================================")
    pd.options.display.float_format = '{:.4f}'.format
    print(df.to_string(index=False))

if __name__ == "__main__":
    # 檢查是否安裝 pandas
    try:
        import pandas
        run_benchmark_suite()
    except ImportError:
        print("Error: Pandas is required for this benchmark script.")
        print("Please run: pip install pandas")