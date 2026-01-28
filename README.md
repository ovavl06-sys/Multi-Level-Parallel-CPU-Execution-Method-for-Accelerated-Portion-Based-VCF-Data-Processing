# Multi-Level-Parallel-CPU-Execution-Method-for-Accelerated-Portion-Based-VCF-Data-Processing
This repository contains Python implementations for the paper "Multi-Level Parallel CPU Execution Method for Accelerated Portion-Based Variant Call Format Data Processing".
The code demonstrates a multi-level parallel CPU-oriented method for processing Variant Call Format (VCF) data, including parsing, feature construction, and mutation pathogenicity classification using Random Forest. It includes sequential and parallel versions, extensions for multi-gene analysis, generalization tests, and statistical/hyperparameter analysis.
Key features include sequential and parallel (multi-core) processing of VCF portions, resource coordination to avoid oversubscription, JIT compilation with Numba for performance, experiments on BRCA1 and other genes (e.g., MLH1, TP53, CDKN2A), and up to 5.69× speedup on multi-core CPUs.

##Repository Structure

sequential_algorithm.py: A baseline implementation that executes VCF processing and Random Forest training in a single-threaded mode. It serves as the primary control for speedup and efficiency benchmarks.
parallel_algorithm.py: The core implementation of the proposed multi-level parallel framework. It integrates block-based data partitioning, task-level feature decomposition, and JIT-compiled numerical kernels with an explicit resource-coordination mechanism.
generalizability.py: An extension applied to multiple genomic regions (MLH1, TP53, CDKN2A). It demonstrates the robustness and scalability of the method across different chromosomes and varying mutation densities.
statistical_analysis.py: A dedicated benchmarking suite that performs repeated measurements (N=5) to calculate statistical significance. It generates mean runtimes, standard deviations, coefficients of variation, and performs paired t-tests to validate performance gains.
data/: Directory containing the filtered VCF dataset (N=5073 variants) for the BRCA1 region, extracted from Ensembl Release 110.

##Technical Methodology
This work utilizes a hierarchical execution model to maximize CPU throughput:
Level I (Data): Asynchronous partitioning of VCF portions into localized blocks.
Level II (Task): Parallel execution of feature construction workflows (normalization, encoding, and annotation).
Level III (Execution): Runtime specialization of numerical kernels via Numba JIT to optimize low-level loop execution.
Requirements & Environment
The code is written in Python 3.10+ and requires the following scientific libraries:
Data Analysis: pandas, numpy, scipy
Machine Learning: scikit-learn (Random Forest, Hyperparameter Optimization, Validation Metrics)
Optimization: numba (JIT compilation and parallel prange)
System: multiprocessing, joblib, logging
