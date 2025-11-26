# **HFS-OPE (Hybrid Feature Selection Optimized Private Expert Model)**

## **1. Introduction**

Developed an enhanced multi-task learning model based on the PLE framework to address negative transfer and improve task-specific representation quality.

### **Key Features**

- **Independent Embeddings**: Constructed task-specific and shared embeddings for each feature, providing sufficient parameter space to capture task-level personalized patterns.

- **Hybrid-granularity Feature Selection**: Used PLE-based offline feature importance to identify task-relevant features, and introduced a FeatureGate module to apply dimension-wise soft masking on task-specific embeddings.

## **2. Model Architecture**
*(To be added)*

## **3. Requirements**
*(To be added)*

## **4. Results**
| Models | Task1AUC | Task2AUC |
|----------|----------|----------|
| MMOE   | 0.938   | 0.979   |
| PLE   | 0.935   | 0.982   |
| OPE   | 0.939   | 0.992   |
| HFS-OPE   | 0.945   | 0.994   |
