# Plant Disease Classification Project

This project focuses on classifying plant diseases based on leaf images using **Transfer Learning** with a **ResNet50** model. The approach leverages a subset of the publicly available **PlantVillage** dataset. Below is a concise overview of the project’s goals, data handling, and implementation details.

## Table of Contents
1. [Project Goal](#project-goal)  
2. [Data Overview](#data-overview)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Model and Training](#model-and-training)  
5. [Results and Evaluation](#results-and-evaluation)  
6. [Key Insights](#key-insights)  
7. [Additional Notes](#additional-notes)  
8. [References](#references)  

---

## Project Goal
The main objective is to accurately classify various plant diseases from leaf images. By employing **ResNet50** (pre-trained on ImageNet), we reduce the computational cost and training time while maintaining high accuracy.

---

## Data Overview
- **Source:** A subset of the [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
- **Number of Classes:** 15 classes of plant diseases.
- **Total Images:** Approximately 20,000 leaf images.
- **Reason for Subset:** Given limited computational resources (e.g., Google Colab), the dataset was restricted to selected classes for faster and more efficient training.
- **Data Location:** The leaf dataset was **previously uploaded to Google Drive** within this project environment.

---

## Data Preprocessing
1. **Folder Structure:** Each class (disease category) is kept in a dedicated folder.  
2. **Splitting Data:**  
   - **Training Set:** 80%  
   - **Validation Set:** 10%  
   - **Test Set:** 10%  
3. **Transformations:**
   - **Image Resize:** 128×128 pixels.  
   - **Data Augmentation:** Random horizontal flips applied to training images.  
   - **Normalization:** Aligned with ImageNet statistics (mean and std), supporting smooth transfer learning.

---

## Model and Training
1. **Architecture:** ResNet50 (imported from `torchvision.models`).  
2. **Transfer Learning Setup:**  
   - **Freezing Layers:** All layers except the last were frozen initially, then the final layer was adjusted for the new number of classes.  
3. **Hyperparameters:**  
   - **Learning Rate (LR):** `0.001` (SGD)  
   - **Momentum:** `0.9`  
   - **Scheduler:** StepLR reducing LR every 7 epochs (gamma=0.1).  
4. **Training Process:**  
   - 8 epochs total.  
   - Monitored loss and accuracy on both training and validation sets each epoch.  
   - Saved the best model based on validation accuracy.

---

## Results and Evaluation
- **Training Loss:** ~0.4284  
- **Validation Loss:** ~0.4117  
- **Training Accuracy:** ~0.8812  
- **Validation Accuracy:** ~0.8794  
- **Test Accuracy:** ~0.8749  

The model generalized well, showing consistent performance across training, validation, and test sets.

---

## Key Insights
1. **Imbalanced Data Handling:**  
   - Techniques such as **oversampling** or **undersampling** can help mitigate bias toward dominant classes.  
   - Transfer learning from ImageNet helps because the model already has generic feature representations.  
2. **Learning Rate Tuning:**  
   - A small LR is essential when fine-tuning pre-trained layers to avoid disrupting previously learned weights.  
   - Gradually increasing LR during a full fine-tuning phase can sometimes improve performance.  
3. **Alternative Models:**  
   - **EfficientNet** might yield higher accuracy and efficiency due to compound scaling and lower resource usage.

---

## Additional Notes
- The **leaf database** for this project was **uploaded beforehand to Google Drive** as part of the workflow, making it easier to mount and access in platforms like Google Colab.
- By leveraging Transfer Learning, the training phase is faster and requires fewer data samples compared to training a model from scratch.

---

## References
- [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)  
- [PyTorch ResNet50 Documentation](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)  
- [Oversampling and Undersampling Techniques](https://www.kaggle.com/code/residentmario/undersampling-and-oversampling-imbalanced-data)  
- [Transfer Learning Insights](https://www.restack.io/p/transfer-learning-answer-unet-learning-rate-cat-ai)
