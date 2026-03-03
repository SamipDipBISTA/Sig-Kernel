# Sig-Kernel
```text
SIG-KERNEL/
├── experiments/
│   └── spx_vix_data/
│
├── src/
│   ├── pycache/
│   ├── init.py
│   ├── helper.py : not very useful
│   ├── optimisation_functions.py : optimisation functoins, ridge+efm, ridge+window, 
                                    localreg+efm, localreg+window
│   ├── pipeline.py               : summary of all the functions
│   ├── regression_functions.py   : regression + local regression pipelines
│   ├── signature_functions.py    : compute efm sig, sliding sig, sliding sig optimized
│   └── spxvix.py                 : efm vs guyon playground
│
├── nwsig/                        : local reg with efm sig included
│
├── playground/                   : for fun
│
├── regime_cluster_detect/        : regime detection sig+umap;                         
                                    implementation of mmd
│
├── sigkernelplay/                : Goursat pde vs truncated sig convergence
│
├── signature heatmap/
│   ├── pycache/
│   ├── heaps_sig_umap.py         
│   ├── heatmap.py
│   ├── heatmaps_sig.py
│   ├── helper.py
│   ├── run_pipeline.py
│   └── signature_data_5min.csv
│
└── slidesig/                     : Nina's code with same experiments adapted for efm 
