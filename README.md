# **HFS-OPE (Hybrid Feature Selection Optimized Private Expert Model)**

## **1. Introduction**

Developed an enhanced multi-task learning model based on the PLE framework to address negative transfer and improve task-specific representation quality.

### **Key Features**

- **Independent Embeddings**: Constructed task-specific and shared embeddings for each feature, providing sufficient parameter space to capture task-level personalized patterns.

- **Hybrid-granularity Feature Selection**: Used PLE-based offline feature importance to identify task-relevant features, and introduced a FeatureGate module to apply dimension-wise soft masking on task-specific embeddings.

## **2. Model Architecture**
<img src="HFSOPE.png" alt="HFS-OPE" width="320" height="240" />

## **3. Requirements**
- **numpy**
- **torch**
- **pandas**
- **scikit-learn**

## **4. Datasets**
| Name | Instances | Features | Labels | link |
|----------|----------|----------|----------|----------|
| census-income   | 299285  | 42  | 2  |https://archive.ics.uci.edu/dataset/117/census+income+kdd|
## **5. Results**
### **census-income**
| Models | Task1AUC | Task2AUC |
|----------|----------|----------|
| MMOE   | 0.938   | 0.979   |
| PLE   | 0.935   | 0.982   |
| OPE   | 0.939   | 0.992   |
| HFS-OPE   | 0.945   | 0.994   |
