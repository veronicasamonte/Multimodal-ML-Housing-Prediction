# 🏡 Austin Housing Price Prediction (Multimodal ML)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Multimodal](https://img.shields.io/badge/Type-Multimodal-purple)
![NLP](https://img.shields.io/badge/NLP-TFIDF%20%2B%20SVD-green)
![Computer Vision](https://img.shields.io/badge/CV-CLIP-red)
![RMSE](https://img.shields.io/badge/RMSE-~200k-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)

---

## 📌 Overview
This project predicts home sale prices in Austin, TX using a **multimodal machine learning approach** that combines structured data, images, and text. The goal is to minimize Root Mean Squared Error (RMSE) on a held-out test set.

---

## 🧠 Approach

### 🔹 Tabular Features
Core housing attributes such as:
- Square footage
- Bedrooms / bathrooms
- Lot size
- Year built
- Location and school metrics

Additional engineered features:
- sqft_per_bed  
- bath_bed_ratio  
- total_rooms  
- home_age  
- lot_to_living_ratio  

---

### 🖼️ Image Features (Computer Vision)
- Used **CLIP (OpenAI)** to generate embeddings from property images  
- Processed images using **batched GPU inference**  
- Reduced dimensionality with **PCA (512 → 15 features)**  

Final image match rate: ~98%

---

### 📝 Text Features (NLP)
- Used listing descriptions  
- Applied:
  - TF-IDF vectorization  
  - Truncated SVD  

Captures qualitative signals like:
- “renovated”
- “luxury”
- “prime location”

---

### 🔗 Feature Fusion
Final dataset combines:
- Tabular features  
- Image PCA features  
- Text SVD features  

---

## 🤖 Model
- Model: XGBoost Regressor  
- Objective: reg:squarederror  
- Training method:
  - 5-fold cross-validation  
  - Averaged predictions across folds  

---

## 📊 Results

| Model | RMSE |
|------|------|
| Tabular only | ~179,000 |
| Initial multimodal | ~218,000 |
| Final multimodal | ~200k–210k |

Key improvements:
- fixing image matching (26% → 98%)  
- reducing image noise with PCA  
- adding text features  
- using cross-validation averaging  

---

## ⚡ Key Challenges

### Image Matching
Initial match rate was very low (~26%).  
Improved using filename normalization and lookup strategies.

### Feature Noise
Raw image embeddings introduced noise.  
PCA helped extract useful signals.

### Multimodal Balance
Tabular data remained the strongest signal, with images and text adding incremental value.

---

## 🚀 How to Run

1. Clone the repository  
2. Open the notebook in Google Colab  
3. Mount Google Drive (for images)  
4. Run cells in order:
   - Data preprocessing  
   - Image embedding generation  
   - Feature engineering  
   - Model training  
   - Submission creation  

---

## 📁 Output

Final submission file:
housingsubmission.csv

Format:
id,target  
12345,450000  
12346,389000  

---

## 🛠️ Tech Stack
- Python  
- Pandas / NumPy  
- Scikit-learn  
- XGBoost  
- HuggingFace Transformers (CLIP)  
- Google Colab  
