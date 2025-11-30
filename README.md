# **HFS-OPE (Hybrid Feature Selection Optimized Private Expert Model)**

## **1. Introduction**

Developed an enhanced multi-task learning model based on the PLE framework to address negative transfer and improve task-specific representation quality.

### **Key Features**

- **Independent Embeddings**: Constructed task-specific and shared embeddings for each feature, providing sufficient parameter space to capture task-level personalized patterns.

- **Hybrid-granularity Feature Selection**: Used PLE-based offline feature importance to identify task-relevant features, and introduced a FeatureGate module to apply dimension-wise soft masking on task-specific embeddings.

## **2. Model Architecture**
<img src="assets/HFSOPE.png" alt="HFS-OPE" width="320" height="240" />

## **3. Requirements**
- **numpy**
- **torch**
- **pandas**
- **scikit-learn**

## **4. Getting started**
```
python main.py \
--task_name=census_income \
--seed=42 \
--model_name=esmm \ #mmoe、ple、hfs_ope
--train_batch_size=1024 \
--val_batch_size=1024 \
--test_batch_size=1024 \
--device=cuda \
--mtl_task_num=2 \
--model_path='/share/home/u17518/yhn/application/HFS-OPE_v2/experiments/census_income_ple_seed42_best_model_2.pth' # use it when model_name = hfs_ope
--dataset_path='./data/KuaiRand/kuairand_pure_input.csv'
```


## **5. Datasets**
| Name | users | items | interactions | Features | link |
|----------|----------|----------|----------|----------|----------|
| census-income   | -  | -  | 299,285  |42|https://archive.ics.uci.edu/dataset/117/census+income+kdd|
| Kuairand-Pure   | 27,285  | 7,583  | 1,186,059  |30(user) + 62(item)|https://kuairand.com/|
| TenRec   | 1M  | 1,948,388  | 86,642,580  |3(user) + 2(item) + User's last 10 interactions|https://tenrec0.github.io/|

Note: Kuairand-Pure only user 10(user) + 8(item) features

## **6. Results**
| Dataset | Metric | ESMM | MMOE | PLE | ours |
|---------|--------|------|------|-----|------|
| census-income | Task1-AUC<br>Task2-AUC | 0.986<br>0.911 | 0.979<br>0.938 | 0.982<br>0.935 | **0.995<br>0.947** |
| Kuairand-Pure | click-AUC<br>like-AUC | 0.601<br>0.839 | **0.744**<br>0.834 | 0.691<br>0.795 | 0.739<br>**0.866** |
| TenRec | click-AUC<br>like-AUC | 0.559<br>0.922 | -<br>- | -<br>- | **-<br>-** |

(The remaining experimental results will be released soon.)
