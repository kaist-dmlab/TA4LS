# TA4LS: Time-series domain Adaptation for Mitigating Label Shfits

[KDD2025] Mitigating Source Label Dependency in Time-Series Domain Adaptation under Label Shifts

This is the implementation of TA4LS published in KDD 2025 [paper](https://dl.acm.org/doi/10.1145/3711896.3737050)

## Overview 
Time-series unsupervised domain adaptation (TS-UDA) is essential in fields such as healthcare and manufacturing, where data often consists of distinct entities, such as individual patients or machinery. This heterogeneity leads to discrepancies not only in feature distributions but also in label distributions, posing a significant challenge for domain adaptation. However, prior studies have mostly focused on alleviating covariate shifts, resulting in predicted target labels that are often biased toward the source domain's label distribution. To address this issue, we propose **T**ime-series domain **A**daptation **for** mitigating **L**abel **S**hifts (**TA4LS**), a novel label refinement approach. TA4LS leverages the consistency between predicted labels and clustering information obtained from the unique characteristics that differentiate each label in the target domain. Furthermore, our approach as a plug-in module achieves performance improvements across diverse existing unsupervised domain adaptation methods, particularly in scenarios with significant discrepancies between source and target label distributions. In experiments on four benchmark datasets with label shifts, TA4LS demonstrates superior performance across six unsupervised domain adaptation methods and six label shift handling modules.


## How to use TA4LS

### (1) Environment Setting
- Python Version: 3.9.13
- Torch Version: 1.13.1+cu117
- Package List
  - Please refer to the `envs.txt` in config folder

### (2) Download the dataset
- [HAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [HHAR_SA](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO)
- [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B)
- [EEG](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)


### (3) Example of the running code

* Parameter options
```
--data_path: set the path of the datasets
--dataset : dataset name (e.g., HAR, HHAR_SA, WISDM, and EEG)
--device : GPU device
--label_shift : True (TA4LS), False (Original UDA)
--backbone : backbone network (e.g., CNN, TCN, and ResNet18)
--exp_name : your experiment name
--da_method : existing UDA method
--num_runs: repetitions
```


* Script
```
python main.py --dataset WISDM --device cuda:0 --label_shift True --exp_name YOUR_EXP_NAME  --da_method DSAN --num_runs 3
```


---

### Code Reference
We refer to the implementation of [AdaTime(TKDD 2023)](https://github.com/emadeldeen24/AdaTime)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
