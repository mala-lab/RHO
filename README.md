<div align="center">
  <h2><b> Semi-supervised Graph Anomaly Detection via Robust Homophily Learning </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/mala-lab/RHO?color=green)
![](https://img.shields.io/github/stars/mala-lab/RHO?color=yellow)
![](https://img.shields.io/github/forks/mala-lab/RHO?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)
[![arXiv](https://img.shields.io/badge/arXiv-2506.15448-b31b1b)](https://arxiv.org/abs/2506.15448)
</div>


# Overview
In this work, we propose RHO, the very first GAD approach designed to learn heterogeneous normal patterns on a set of labeled normal nodes. RHO is implemented by two novel modules, AdaFreq and GNA. AdaFreq learns a set of adaptive spectral filters in both the cross-channel and channel-wise view of node attribute to capture the heterogeneous normal patterns from the given limited labeled normal nodes, while GNA is designed to to enforce the consistency of the learned normal patterns, thereby facilitating the learning of robust normal representations on datasets with different levels of homophily in the normal nodes. 

![image](https://github.com/mala-lab/RHO/blob/main/figures/RHO.png)

# Environment Settings
This implementation is based on Python3. To run the code, you need the following dependencies: <br>
* torch==2.0.0+cu117
* torch-geometric==2.3.0
* scipy==1.10.1
* numpy==1.19.5
* tqdm==1.24.4
* dgl==1.0.0+cu117
* seaborn==0.12.2
* scikit-learn==1.2.2
# Datasets
For convenience, some datasets can be obtained from [google drive link](https://drive.google.com/drive/folders/1x0cEGRCGtPGGbKY9S1x4rvDqJrxlLB27?dmr=1&ec=wgc-drive-hero-goto) . We sincerely thank the researchers for providing these datasets. Due to the Copyright of DGraph-Fin, you need to download from [DGraph-Fin](https://dgraph.xinye.com/introduction).

# Run experiments:
    $ sh run.sh
    
By running the following scripts with the provided checkpoints, you should be able to reproduce the results reported in Table 1 of our paper. <br>
    
    $ python reproduction.py --dataset name

To facilitate reproducibility, we provide a recording of the experiment execution process in **Run process**. This video demonstrates the procedure to run the code and reproduce the reported results.

## 📖 Citation
    
If you find this work useful, please cite our paper:

```bibtex
@article{ai2026semi,
  title={Semi-supervised Graph Anomaly Detection via Robust Homophily Learning},
  author={Ai, Guoguo and Qiao, Hezhe and Yan, Hui and Pang, Guansong},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={133988--134013},
  year={2025}
}



