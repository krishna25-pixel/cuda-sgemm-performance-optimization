from pathlib import Path
import pandas as pd
r = Path(__file__).resolve().parents[1] / "results"
files = [r/f"sgemm_run{i}.csv" for i in (1,2,3)]
assert all(f.exists() for f in files), "Missing independent-run CSVs"
frames = [pd.read_csv(f) for f in files]
for i, f in enumerate(frames,1):
    if not (f["correct"] == 1).all(): raise RuntimeError(f"SGEMM correctness failure run {i}")
for i, f in enumerate([r/f"reduction_run{j}.csv" for j in (1,2,3)],1):
    d = pd.read_csv(f)
    if not (d["correct"] == 1).all(): raise RuntimeError(f"Reduction correctness failure run {i}")
rows = pd.concat([f[(f.N == 1024) & (f.kernel.isin(["naive","tile64_reg4","cublas"]))] for f in frames])
s = rows.groupby("kernel")["median_ms"].median()
print("N=1024 median times (ms):",s.to_dict())
print("Tile64 speedup versus naive: %.3fx" % (s["naive"]/s["tile64_reg4"]))
print("All supplied CSV correctness flags passed.")
