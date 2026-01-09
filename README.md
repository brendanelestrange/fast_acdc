# Fast ACDC: High-Performance Graph Matching Solver

**Fast ACDC** is a highly optimized C++ implementation of the **Alternating Convex/Discrete Coupling (ACDC)** algorithm for solving large-scale Graph Matching (GM) problems.

This project was specifically engineered to solve the **VNC (Ventral Nerve Cord) Connectome Matching Challenge** ($N \approx 25,000$ nodes) on commodity hardware with limited RAM (24GB), achieving competitive solutions in under an hour.

## 🚀 Key Features

* **Zero-Copy Architecture:** Implements custom Eigen wrappers and in-place matrix modifications to prevent memory explosion. Capable of handling $25k \times 25k$ dense matrices on <24GB RAM.
* **Hybrid Solver:**
    * **Phase 1:** Continuous Relaxation using the **Frank-Wolfe** algorithm with an optimized **LAPJV** (Linear Assignment Problem) backend.
    * **Phase 2:** Discrete Local Search using a parallelized greedy strategy to refine the permutation via pairwise swaps.
* **Parallelized:** Extensive use of **OpenMP** to parallelize gradient computations and discrete swap scanning, achieving a ~100x speedup over standard Python implementations.
* **Sparse Optimized:** Efficient handling of sparse adjacency matrices (`Eigen::SparseMatrix`) for graph operations.

## 🛠️ Prerequisites

* **C++ Compiler** supporting C++17 (GCC, Clang, or MSVC).
* **CMake** (Version 3.7 or higher).
* **Eigen3:** A header-only C++ library for linear algebra.
* **OpenMP:** For multithreading support.

## 📦 Build Instructions

1.  **Clone the repository** (and ensure `libraries/lapjv.cpp` and `lapjv.h` are present).
2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```
3.  **Configure with CMake:**
    ```bash
    cmake ..
    ```
    *(Note: On macOS, the CMake script automatically checks standard Homebrew paths for `libomp`)*.
4.  **Compile:**
    ```bash
    make
    ```

## 🏃 Usage

Ensure your input data files are in the data folder and the executable is in the project folder(or update the paths in `acdc.cpp`).

### Required Data Files
The solver expects the following CSV files:
1.  **`male_connectome_graph.csv`**: Source graph adjacency list (u, v, weight).
2.  **`female_connectome_graph.csv`**: Target graph adjacency list (u, v, weight).
3.  **`vnc_matching_submission_benchmark_XXXX.csv`**: Initial permutation/alignment file.

### Running the Solver
```bash
./acdc