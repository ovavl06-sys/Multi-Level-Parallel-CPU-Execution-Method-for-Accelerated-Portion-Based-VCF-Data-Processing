# Multi-Level Parallel CPU Execution Method for Accelerated Portion-Based VCF Data Processing

This repository contains Python implementations for the paper 
"Multi-Level Parallel CPU Execution Method for Accelerated Portion-Based Variant Call Format Data Processing".

The code demonstrates a multi-level parallel CPU-oriented method for processing Variant Call Format (VCF) data, including parsing, feature construction, and mutation pathogenicity classification using Random Forest. It includes sequential and parallel versions, extensions for multi-gene analysis, generalization tests, and statistical/hyperparameter analysis.

Key features include sequential and parallel (multi-core) processing of VCF portions, resource coordination to avoid oversubscription, JIT compilation with Numba for performance, experiments on BRCA1 and other genes (MLH1, TP53, CDKN2A), and up to 5.69× speedup on multi-core CPUs.

---

## Repository Structure

- **sequential_algorithm.py**: Baseline single-threaded VCF processing and Random Forest training; used for speedup and efficiency benchmarks.  
- **parallel_algorithm.py**: Multi-level parallel framework with block-based partitioning, task-level decomposition, and JIT numerical kernels.  
- **generalizability.py**: Extension for multiple genomic regions (MLH1, TP53, CDKN2A) demonstrating scalability and robustness.  
- **statistical_analysis.py**: Benchmarking suite performing repeated measurements, calculating statistics, and paired t-tests.  
- **data/**: Filtered VCF dataset for BRCA1 region (N=5073 variants), from Ensembl Release 110.

---

## Technical Methodology

This work utilizes a hierarchical execution model to maximize CPU throughput:

- **Level I (Data):** Asynchronous partitioning of VCF portions into localized blocks.  
- **Level II (Task):** Parallel execution of feature construction workflows (normalization, encoding, annotation).  
- **Level III (Execution):** Runtime specialization of numerical kernels via Numba JIT to optimize low-level loop execution.

---

## Requirements & Environment

- **Python:** 3.10+  
- **Data Analysis:** pandas, numpy, scipy  
- **Machine Learning:** scikit-learn (Random Forest, Hyperparameter Optimization, Validation Metrics)  
- **Optimization:** numba (JIT compilation and parallel `prange`)  
- **System:** multiprocessing, joblib, logging  

> **Note:** The code is currently configured to run on the author's local environment; file paths may need to be adjusted for other systems.
