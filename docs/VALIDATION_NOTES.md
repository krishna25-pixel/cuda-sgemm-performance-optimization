# Validation status

Confirmed from user-supplied Kaggle outputs: source compiled/executed on Tesla T4, SGEMM and reduction tests reported PASS, three independent run CSVs were supplied, plus one seed-941 N=2048 stress run. All reported PASS flags in CSVs were set to 1.

N=1024 Tile64/Reg4 independent times: 0.884256, 0.871696, 0.904602 ms. Median 0.884256 ms, 2428.58 GFLOPS.

The benchmark's SGEMM numerical oracle is cuBLAS with tolerance `(absolute <= 0.002 OR relative <= 0.0005)` (both thresholds exceeded causes failure). This can hide some errors, so correctness is *within specified tolerance* rather than bitwise identical or independently proven.

At N=1024 cuBLAS recorded 0.650227, 0.344474, 0.339366 ms in the three independent runs. The relative performance claim varies and is not suitable as a fixed headline figure.

What has NOT been established: controlled GPU clock consistency, comprehensive independent CPU FP64 SGEMM oracle validation, verified Nsight Compute microarchitectural counters, or reproducible cuBLAS percentage.
