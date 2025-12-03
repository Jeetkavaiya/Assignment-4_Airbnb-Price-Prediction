# Airbnb Price Prediction: XGBoost vs Neural Networks

## EAS 510 - Assignment 4

A comprehensive machine learning analysis comparing XGBoost and Neural Network models for predicting Airbnb listing prices across 12 US cities organized into three market tiers.

---

## Project Overview

This project implements and compares gradient boosting (XGBoost) and deep learning (Neural Network) approaches for Airbnb price prediction. The analysis examines model performance at three levels:
1. **Individual City Performance** - Training separate models for each of 12 cities
2. **Composite Tier Analysis** - Training models on combined data within market tiers
3. **Cross-Tier Generalization** - Testing how models trained on one tier perform on others

---

## Objectives

- Compare XGBoost vs Neural Networks for price prediction across different market sizes
- Analyze how market characteristics affect model performance
- Evaluate cross-tier generalization capabilities of neural networks
- Identify which model architecture performs best for Airbnb pricing

---

## Dataset Information

Data sourced from [Inside Airbnb](https://insideairbnb.com/get-the-data/)

| City | Tier | Data Month | Listings Count |
|------|------|------------|----------------|
| NYC | Big | Oct 2025 | 21,073 |
| LA | Big | Sept 2025 | 36,535 |
| San Francisco | Big | Sept 2025 | 5,775 |
| Chicago | Big | June 2025 | 7,608 |
| Austin | Medium | June 2025 | 10,598 |
| Seattle | Medium | Sept 2025 | 6,157 |
| Denver | Medium | Sept 2025 | 4,271 |
| Portland | Medium | Sept 2025 | 3,762 |
| Asheville | Small | June 2025 | 2,530 |
| Santa Cruz | Small | June 2025 | 1,545 |
| Salem | Small | Sept 2025 | 279 |
| Columbus | Small | Sept 20245| 2,683 |

**Total Listings Analyzed: 102,816**

---

## How to Run the Code

### Prerequisites
```bash
pip install requirements.txt
```

### Step 1: Download Data
1. Visit [Inside Airbnb](https://insideairbnb.com/get-the-data/)
2. Download `listings.csv` for each of the 12 cities listed above

### Step 2: Run in Google Colab
1. Upload the notebook to [Google Colab](https://colab.research.google.com/)
2. Enable GPU: Runtime → Change runtime type → GPU
3. Upload all 12 CSV files
4. Run all cells sequentially

---

## Feature Engineering

**22 Total Features** (15 base + 7 engineered)

### Base Features (15)
- `accommodates`, `bedrooms`, `beds`, `bathrooms_numeric`
- `review_scores_rating`, `review_scores_accuracy`, `review_scores_cleanliness`
- `review_scores_checkin`, `review_scores_communication`, `review_scores_location`, `review_scores_value`
- `number_of_reviews`, `availability_365`, `minimum_nights`, `maximum_nights`

### Engineered Features (7)
| Feature | Description |
|---------|-------------|
| `avg_review_score` | Average of all 7 review score dimensions |
| `price_per_bedroom` | Price normalized by bedroom count |
| `review_activity` | Reviews per month indicator |
| `amenities_count` | Total count of amenities offered |
| `property_type_encoded` | Encoded property type category |
| `room_type_encoded` | Encoded room type (Entire/Private/Shared) |
| `host_is_superhost_encoded` | Binary superhost indicator |

---

## Results Summary

### Part 1: Individual City Performance (12 Cities)

| City | Tier | XGB RMSE | NN RMSE | Better Model |
|------|------|----------|---------|--------------|
| NYC | Big | $48.57 | $5.70 | Neural Network |
| LA | Big | $40.33 | $8.84 | Neural Network |
| SF | Big | $110.35 | $9.59 | Neural Network |
| Chicago | Big | $60.19 | $7.53 | Neural Network |
| Austin | Medium | $60.84 | $6.94 | Neural Network |
| Seattle | Medium | $11.78 | $8.00 | Neural Network |
| Denver | Medium | $11.76 | $3.92 | Neural Network |
| Portland | Medium | $10.33 | $4.86 | Neural Network |
| Asheville | Small | $38.82 | $5.59 | Neural Network |
| Santa Cruz | Small | $16.13 | $13.81 | Neural Network |
| Salem | Small | $8.96 | $11.37 | **XGBoost** |
| Columbus | Small | $14.63 | $4.73 | Neural Network |

**Winner: Neural Network (11/12 cities)**

---

### Part 2: Composite Tier Analysis

| Tier | Total Samples | XGB RMSE | XGB R² | NN RMSE | NN R² | Better Model |
|------|---------------|----------|--------|---------|-------|--------------|
| Tier1_Big | 70,991 | $44.83 | 0.9813 | $7.21 | 0.9995 | Neural Network |
| Tier2_Medium | 24,788 | $74.51 | 0.9241 | $6.88 | 0.9994 | Neural Network |
| Tier3_Small | 7,037 | $27.04 | 0.9858 | $3.82 | 0.9997 | Neural Network |

**Winner: Neural Network (all 3 tiers)**

---

### Part 3: Cross-Tier Neural Network Generalization

#### R² Matrix (Source Model → Target Data)
| Source Model | Tier1_Big | Tier2_Medium | Tier3_Small |
|--------------|-----------|--------------|-------------|
| Tier1_Big | 0.9995 | 0.9993 | 0.9989 |
| Tier2_Medium | 0.9992 | 0.9994 | 0.9993 |
| Tier3_Small | 0.9997 | 0.9997 | 0.9997 |

#### RMSE Matrix (Source Model → Target Data)
| Source Model | Tier1_Big | Tier2_Medium | Tier3_Small |
|--------------|-----------|--------------|-------------|
| Tier1_Big | $7.21 | $7.39 | $7.56 |
| Tier2_Medium | $9.26 | $6.88 | $6.04 |
| Tier3_Small | $5.56 | $4.47 | $3.82 |

#### Generalization Summary
- **Same-Tier Average R²:** 0.9995
- **Cross-Tier Average R²:** 0.9993
- **Generalization Drop:** ~0% (excellent transfer)
- **RMSE Increase:** 12.5% when predicting other tiers
- **Best Generalizing Model:** Tier3_Small

---

## Key Findings

### 1. XGBoost vs Neural Network
- **Neural Networks win 11/12 individual cities** and all 3 composite tiers
- XGBoost only outperformed NN in Salem (smallest dataset with 279 samples)
- Neural Networks achieve R² > 0.99 consistently across all markets

### 2. Cross-Tier Generalization
- Models maintain excellent performance (R² > 0.998) even on unseen market tiers
- **Tier3_Small model generalizes best** to other markets
- RMSE increases only ~12.5% when predicting other tiers

### 3. Market Tier Insights
- **Big cities** show most price variation (hardest to predict with XGBoost)
- **Small cities** are more homogeneous (easier to predict, best generalization)
- Cross-tier transfer demonstrates market-specific patterns exist but are learnable

### 4. Model Architecture Impact
- Neural Network (128→64→32→16 architecture) with dropout regularization significantly outperforms XGBoost
- Early stopping prevents overfitting effectively
- L2 regularization improves generalization

---

## Repository Structure

```
airbnb-price-prediction/
├── README.md                          # This file
├── requirements.txt
├── Airbnb_Price_Prediction.ipynb      # Main Jupyter notebook
├── data/                              # Data files (not included)
│   ├── NYC.csv
│   ├── LA.csv
│   ├── SF.csv
│   ├── Chicago.csv
│   ├── Austin.csv
│   ├── Seattle.csv
│   ├── Denver.csv
│   ├── Portland.csv
│   ├── Asheville.csv
│   ├── SantaCruz.csv
│   ├── Salem.csv
│   └── Columbus.csv
└── outputs/
    └── cross_tier_analysis.png        # Generated visualization
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML Framework | scikit-learn |
| Gradient Boosting | XGBoost |
| Deep Learning | TensorFlow/Keras |
| Environment | Google Colab (GPU) |

---

## Model Architectures

### XGBoost Configuration
```python
XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1
)
```

### Neural Network Architecture
```
Input (22 features)
    ↓
Dense(128, ReLU) + L2 Regularization
    ↓
Dropout(0.3)
    ↓
Dense(64, ReLU) + L2 Regularization
    ↓
Dropout(0.2)
    ↓
Dense(32, ReLU)
    ↓
Dropout(0.2)
    ↓
Dense(16, ReLU)
    ↓
Dense(1) - Output
```

---

## References

- Inside Airbnb: http://insideairbnb.com/get-the-data/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- TensorFlow/Keras: https://www.tensorflow.org/

---

## 📝 License

This project is for educational purposes as part of EAS 510 coursework.
