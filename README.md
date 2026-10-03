# Facial Expression Recognition (HOG + Linear SVM)

Module for classical machine learning baseline using Histogram of Oriented Gradients (HOG) and Linear Support Vector Classifier (LinearSVC) on the clean FER2013 dataset.

## How to Run Real Training & Evaluation

To execute HOG feature extraction, train LinearSVC on the clean training set, and evaluate on the untouched test set:

```bash
python experiments/run_hog_svm.py
```

## Artifact & Output Locations

- **Trained Model:** `results/models/hog_svm_model.joblib` (bundles trained classifier + HOG config + class names)
- **Metrics Table:** `results/tables/hog_svm_test_metrics.csv` (and `.json`)
- **Confusion Matrix Figure:** `results/figures/confusion_matrix.png`
- **Reproducibility Log:** `results/reproducibility.json`
