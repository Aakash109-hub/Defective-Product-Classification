# Defective-Product-Classification
## 1. Introduction

Casting defects in industrial manufacturing can lead to significant financial losses due to rejected orders and limitations in manual inspection. The objective of this project is to develop a machine learning model to classify casting products as **Defective** or **Non-defective**, automating the quality assurance process. This project employs a deep learning approach using a Convolutional Neural Network (CNN) to classify images of submersible pump impellers based on visible casting defects.

## 2. Methodology

### 2.1 Dataset
The dataset used in this project consists of grayscale images (300x300 pixels) of submersible pump impellers, provided with data augmentation to improve model generalization.

- **Dataset source**: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/data
- **Classes**:
  - **Defective (def_front)**: Contains images of impellers with defects (e.g., blowholes, shrinkage).
  - **Non-defective (ok_front)**: Images of acceptable, defect-free impellers.

The data was split into training and test sets:
- Training set:
  - Defective: 3,758 images
  - Non-defective: 2,875 images
- Test set:
  - Defective: 453 images
  - Non-defective: 262 images

### 2.2 Data Preprocessing
To prepare the dataset for training, the following preprocessing steps were applied:
- **Normalization**: Scaled pixel values to a range between 0 and 1.
- **Data Augmentation**: Applied techniques such as flipping, rotation, and zooming to increase variability and reduce overfitting.

### 2.3 Model Architecture
A Convolutional Neural Network (CNN) was chosen for its effectiveness in image classification. The CNN architecture included:
- **Convolutional Layers**: Extracted essential features like edges, textures, and shapes.
- **Pooling Layers**: Reduced the spatial dimensions and helped generalize the model.
- **Dropout Layers**: Randomly deactivated neurons during training to reduce overfitting.
- **Fully Connected Layer**: Combined extracted features for classification.
- **Output Layer**: A softmax layer for binary classification.

### 2.4 Training and Evaluation
- **Loss Function**: Binary cross-entropy was used.
- **Optimizer**: Adam optimizer for its adaptability and efficiency.
- **Metrics**: Accuracy, precision, recall, and F1-score.
- **Early Stopping and Model Checkpointing**: Stopped training when performance plateaued and saved the model with the best validation accuracy.

## 3. Model Performance

### 3.1 Accuracy
The model achieved high accuracy on the test set, demonstrating its effectiveness in distinguishing between defective and non-defective casting products.
- **Test Accuracy**: 99.4%

### 3.2 Confusion Matrix
The confusion matrix below summarizes the model's predictions:

|               | Predicted Non-defective | Predicted Defective |
|---------------|-------------------------|----------------------|
| **Actual Non-defective** | 451                    | 2                    |
| **Actual Defective**     | 2                      | 260                  |

This indicates that the model made only four errors in total: two false positives (non-defective items misclassified as defective) and two false negatives (defective items misclassified as non-defective). These low error rates underscore the model’s reliability for real-world application.

### 3.3 Classification Report
The classification report provides further insight into the model's precision, recall, and F1-score for each class:

| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| Non-defective   | 0.996     | 0.996  | 0.996    | 453     |
| Defective       | 0.992     | 0.992  | 0.992    | 262     |
| **Overall Accuracy** | **0.994** | -      | -        | 715     |
| **Macro Avg**   | 0.994     | 0.994  | 0.994    | 715     |
| **Weighted Avg**| 0.994     | 0.994  | 0.994    | 715     |

### Interpretation of Classification Metrics:
- **Precision**: High precision (0.996 for non-defective, 0.992 for defective) indicates that when the model predicts an item as defective, it is highly likely to be correct.
- **Recall**: High recall (0.996 for non-defective, 0.992 for defective) shows that the model is adept at identifying both defective and non-defective items correctly.
- **F1-score**: The high F1-scores for both classes indicate a balanced performance with low rates of both false positives and false negatives.

## 4. Insights and Conclusion

### 4.1 Key Insights
- **High Accuracy and Reliability**: The model’s performance metrics demonstrate its capability to accurately detect defects, which can significantly reduce errors associated with manual inspection.
- **Improved Efficiency**: Automating defect detection can expedite the inspection process, which is traditionally time-consuming when done manually..
- **Industrial Application Potential**: Given its high accuracy and reliability, the model could be deployed in an industrial setting to improve quality assurance processes.

### 4.2 Limitations
- **Binary Classification**: The current model only classifies images as defective or non-defective, without identifying specific defect types. For practical use, additional subclassifications of defects could be useful.
- **Consistency in Lighting and Image Quality**: The model’s accuracy depends on the consistent quality of images and lighting conditions.

### 4.3 Conclusion
The project successfully demonstrates the use of deep learning in automating quality control for casting products. The model achieved a high accuracy of 99.4%, indicating its potential to improve inspection accuracy and streamline industrial processes. This work highlights the value of AI in enhancing manufacturing quality assurance.
