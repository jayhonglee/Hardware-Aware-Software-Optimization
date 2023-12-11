# General Matrix Multiply Optimization on X86 CPU

This project focuses on optimizing the General Matrix Multiply (GEMM) algorithm on a 10-core X86 CPU using C. The goal was to enhance performance through various software optimization techniques.

## Project Overview

-   **Objective**: Optimize GEMM algorithm for maximum performance on X86 architecture.
-   **Platform**: 10-core X86 CPU.
-   **Language Used**: C

## Optimizations Implemented

1. **Data Tiling Optimization**

    - Employed data tiling technique with a tile size of 16.
    - Achieved an initial speedup of 88.14%.

2. **X86 SIMD Intrinsics for Vectorization**

    - Implemented SIMD (Single Instruction, Multiple Data) intrinsics specific to X86 architecture.
    - Combined with data tiling, resulted in a 98.39% speedup.

3. **OpenMP Multithreading**

    - Utilized OpenMP for multithreading purposes in conjunction with data tiling and vectorization.
    - Achieved a speedup of 99.81%.

4. **Loop Unrolling**
    - Integrated loop unrolling with the aforementioned optimizations.
    - Final speedup reached 99.84%, reducing computation time from 8 minutes to 0.73 seconds.

## Results

The optimizations conducted on the GEMM algorithm led to a significant enhancement in performance, demonstrating the effectiveness of the employed strategies.

## Optimization Report

You can find the detailed optimization report [here](./optimizationReport.pdf).

<!-- ## Usage -->

## Acknowledgments

I collaborated on this project with my partner Denzel, whose contributions were instrumental to the success of these optimizations.

## Future Improvements

In the future, I am keen on exploring optimization techniques tailored for various specifications of CPUs and hardware architectures. Experimenting with optimizations on different types of PCs and architectures could provide valuable insights into tailoring performance enhancements for diverse computing environments.

## License

This project was completed as part of the coursework for Introduction to Computer Organization (ENSC 254) under the guidance of Professor Zhenman Fang. All rights reserved under the terms of the course project guidelines.
