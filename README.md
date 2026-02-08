# NXP-EdgeAI-Wafer-Defect-Detection
# 🚀 Edge-AI Semiconductor Wafer Defect Classification
### **NXP IESA Hackathon 2026 | Phase 1 Submission**

This project implements a high-speed, lightweight Deep Learning system for real-time defect detection in semiconductor wafer images. It is specifically optimized for deployment on **NXP i.MX RT** series hardware using the **eIQ® AI Development Environment**.

---

## **📊 Performance Highlights (Projected)**
| Metric | Result | Environment |
| :--- | :--- | :--- |
| **Inference Latency** | **3.55 ms** | TFLite (INT8 Emulated) |
| **Throughput** | **~281.8 FPS** | Optimized for Inline Inspection |
| **Model Size** | **1.95 MB** | Ultra-lightweight for Edge SRAM |
| **Validation Accuracy** | **97.5%** | Robust 8-Class Classification |

---

## **🛠️ Technical Architecture**
[cite_start]To meet the "Real-Time" and "Low-Power" requirements[cite: 39, 43], the system utilizes:
* **Backbone:** MobileNetV3-Small for high feature-extraction efficiency.
* **Custom Layer:** **Spatial Attention Module** to focus on microscopic defect structures.
* [cite_start]**Optimization:** INT8 Post-Training Quantization (PTQ) to ensure compatibility with NXP NPU/eIQ[cite: 58].
* **Loss Function:** Focal Loss + Label Smoothing to handle class imbalance.

---

## **🔍 Model Explainability & Results**

### **Explainable AI (Grad-CAM)**
[cite_start]We utilize Grad-CAM heatmaps to verify the model focuses on actual defect morphology (Bridges, Opens, Malformed Vias) rather than background noise[cite: 49, 63].

![Grad-CAM Results](results/image_a7f775.jpg)

### **Confusion Matrix**
[cite_start]The model demonstrates high precision across all 8 required categories[cite: 49].

![Confusion Matrix](results/image_a7a540.png)

### **Training Convergence**
Stable convergence over 50 epochs, balancing accuracy with effective regularization.

![Accuracy and Loss](results/image_a7a521.png)

---

## **📁 Repository Structure**
```text
├── data/               # Dataset samples (Clean + 7 Defect types)
├── models/             # Final ONNX and TFLite (INT8) model files
├── notebook/           # Complete development and training code
├── results/            # Performance tables and visualization plots
├── src/                # Custom Python modules (Attention, Preprocessing)
├── README.md           # Project documentation
└── requirements.txt    # Environment dependencies
