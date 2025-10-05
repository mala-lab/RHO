# RHO
Official PyTorch implementation of ''Semi-supervised Graph Anomaly Detection via Robust Homophily Learning'' <br>
# Overview
In this work, we propose RHO, the very first GAD approach designed to learn heterogeneous normal patterns on a set of labeled normal nodes. RHO is implemented by two novel modules, AdaFreq and GNA. AdaFreq learns a set of adaptive spectral filters in both the cross-channel and channel-wise view of node attribute to capture the heterogeneous normal patterns from the given limited labeled normal nodes, while GNA is designed to to enforce the consistency of the learned normal patterns, thereby facilitating the learning of robust normal representations on datasets with different levels of homophily in the normal nodes. 
![image](https://github.com/GGA23/GrokFormer/blob/main/GrokFormer_demo.gif)

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
You can download the datasets from (https://drive.google.com/drive/folders/1x0cEGRCGtPGGbKY9S1x4rvDqJrxlLB27?dmr=1&ec=wgc-drive-hero-goto)

# Run experiments:
    $ sh run.sh
    
By running the following scripts with the provided checkpoints, you should be able to reproduce the results reported in Table 1 of our paper. <br>
    
    $ python reproduction.py --dataset name

To facilitate reproducibility, we provide an recording of the experiment execution process in **Run process**. This video demonstrates the procedure to run the code and reproduce the reported results.

## 📖 Citation
    
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ai2025RHO,
  title     = {Semi-supervised Graph Anomaly Detection via Robust Homophily Learning},
  author    = {Ai, Guoguo and Qiao, Hezhe and Yan, Hui and Pang, Guansong},
  booktitle = {Proceedings of the Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}



