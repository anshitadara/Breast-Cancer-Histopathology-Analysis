# Breast Cancer Histopathology Analysis (BreakHis Dataset)

This project develops a deep learning pipeline to classify histopathological breast tissue images into benign and malignant classes using the BreakHis dataset. It includes patient-wise data splitting, EfficientNet-B0 fine-tuning, weighted loss to handle imbalance, and detailed failure analysis.

---

## 1. Dataset

**Dataset:** BreakHis – Breast Cancer Histopathological Image Dataset  
**Total images:** 7,909  
**Patients:** 82  
**Classes:** Benign (4 subtypes), Malignant (4 subtypes)  
**Magnification levels:** 40×, 100×, 200×, 400×
**Dataset Structure**
- BreaKHis_v1
  - histology_slides
    - breast
      - benign
        - SOB
          - adenosis
            - patient folders
              - magnification folders (40×, 100×, 200×, 400×)
                - image files (.png)
      - malignant
        - SOB
          - ductal_carcinoma
            - patient folders
              - magnification folders
                - image files (.png)


A **patient-wise split** (not image-wise) is performed to avoid information leakage.

---

## 2. Why Patient-Wise Split

Many online tutorials report >95% accuracy because they incorrectly split images randomly.  
This allows patches from the same patient to appear in both train and test sets, causing the network to memorize features.

This project instead uses:
- 70% train
- 15% validation
- 15% test

Assignment is done at **patient ID level**, ensuring evaluation reflects real-world generalization.

---

## 3. Model

**Architecture:** EfficientNet-B0 (ImageNet pretrained)  
**Training Strategy:** Two-phase fine-tuning  
- Phase-1: Backbone frozen, train classifier only  
- Phase-2: Full model fine-tuning  

**Loss:** Class-weighted CrossEntropy (to penalize malignant false-negatives)  
**Optimizer:** AdamW  
**Scheduler:** ReduceLROnPlateau  
**Early Stopping:** Enabled  

---

## 4. Results (Test Set – Patient-Wise)

| Metric       | Score |
|--------------|--------|
| Accuracy     | 83.64% |
| Precision    | 81.95% |
| Recall       | 90.02% |
| F1-Score     | 85.79% |

### Interpretation
The model is tuned to avoid missing malignant cases, achieving high recall. Benign tissue is sometimes predicted malignant (false positives), which is acceptable in screening scenarios.

---

## 5. Confusion Matrix

<img width="568" height="455" alt="image" src="https://github.com/user-attachments/assets/42aa37e1-fc8b-4f97-95f0-bf147bbc338d" />

---

## 6. Failure Analysis

### Most misclassified subtypes:
phyllodes_tumor 92 (benign)
mucinous_carcinoma 56 (malignant)
tubular_adenoma 49 (benign)
ductal_carcinoma 27 (malignant)


### Worst magnification:
40× → most errors (lacks cellular detail)
200× → best performance (good texture + context balance)


![Misclassified Samples](results/misclassified_example1.png)
![Misclassified Samples](results/misclassified_example2.png)


---

## 7. How To Run

### Option A — Use Google Colab
1. Open the notebook: `notebooks/BreakHis_EfficientNet.ipynb`
2. Ensure Drive is mounted
3. To skip retraining, enable:
FAST_RESUME = True
4. The notebook will automatically load:
   - trained model weights (`models/efficientnet_b0_best.pt`)
   - patient-wise metadata

### Option B — Local Environment

Install dependencies:
pip install -r environment.txt

---

## 8. Model Weights

Best checkpoint (EfficientNet-B0):
models/efficientnet_b0_best.pt

## 9. Future Improvements (Planned)

Upcoming experiments to include in future paper revision:
- Stain normalization (Macenko / Reinhard)
- Multi-scale modeling (40× + 200× fusion)
- EfficientNet-B3 and ConvNeXt-Tiny comparison
- Subtype-aware loss or focal loss
- Grad-CAM interpretability overlays

---

## 10. Citation – If you reference this work

BreakHis Dataset:
https://www.kaggle.com/datasets/ambarish/breakhis

