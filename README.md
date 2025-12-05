# 📦 Smart Product Pricing Solution
<img width="500" height="500" alt="ChatGPT Image Dec 5, 2025, 07_54_51 AM" src="https://github.com/user-attachments/assets/7cb74961-286d-4ec6-9ccc-bb5f31d52952" />


> **🏆 Achievement:** Ranked **700 out of 6,500+ teams** with a **SMAPE score of 54.1%**

---
# 📚 Table of Contents
- [Introduction](#introduction)
- [Challenge Statement](#challenge-statement)
- [Proposed Solution](#proposed-solution)
- [Model Architecture](#model-architecture)
- [Implementation Details](#implementation-details)
  - [Core Components](#core-components)
  - [Key CatBoost Parameters](#key-catboost-parameters)
  - [Data Preprocessing](#data-preprocessing)
- [Data Exploration & EDA](#data-exploration--eda)
- [Feature Analysis](#feature-analysis)
- [Key Insights](#key-insights)
- [Validation Results](#validation-results)
- [Running Guide](#running-guide)
- [Project Structure](#project-structure)
- [Notes & Next Steps](#notes--next-steps)

---

# Smart Product Pricing Challenge

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

Data Description: The dataset consists of the following columns:

sample_id: A unique identifier for the input sample catalog_content: Text field containing title, product description and an Item Pack Quantity(**IPQ**) concatenated. image_link: Public **URL** where the product image is available for download. Example link - [https://m.media-amazon.com/images/I/71XfHPR36-L.jpg](https://m.media-amazon.com/images/I/71XfHPR36-L.jpg) To download images use download_images function from src/utils.py. See sample code in src/test.ipynb. price: Price of the product (Target variable - only available in training data) Dataset Details: Training Dataset: 75k products with complete product details and prices Test Set: 75k products for final evaluation Output Format: The output file should be a **CSV** with 2 columns:

sample_id: The unique identifier of the data sample. Note the ID should match the test record sample_id. price: A float value representing the predicted price of the product. Note: Make sure to output a prediction for all sample IDs. If you have less/more number of output samples in the output file as compared to test.csv, your output won't be evaluated.

File Descriptions: Source files

src/utils.py: Contains helper functions for downloading images from the image_link. You may need to retry a few times to download all images due to possible throttling issues. sample_code.py: Sample dummy code that can generate an output file in the given format. Usage of this file is optional. Dataset files

dataset/train.csv: Training file with labels (price). dataset/test.csv: Test file without output labels (price). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv dataset/sample_test.csv: Sample test input file. dataset/sample_test_out.csv: Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct Constraints: You will be provided with a sample output file. Format your output to match the sample output file exactly.

Predicted prices must be positive float values.

Final model should be a **MIT**/Apache 2.0 License model and up to 8 Billion parameters.

valuation Criteria: Submissions are evaluated using Symmetric Mean Absolute Percentage Error (**SMAPE**): A statistical measure that expresses the relative difference between predicted and actual values as a percentage, while treating positive and negative errors equally.

Formula:

**SMAPE** = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)

Academic Integrity and Fair Play: ⚠️ **STRICTLY** **PROHIBITED**: External Price Lookup

Participants are **STRICTLY** **NOT** **ALLOWED** to obtain prices from the internet, external databases, or any sources outside the provided dataset. This includes but is not limited to:

Web scraping product prices from e-commerce websites Using APIs to fetch current market prices Manual price lookup from online sources Using any external pricing databases or services Enforcement:

All submitted approaches, methodologies, and code pipelines will be thoroughly reviewed and verified Any evidence of external price lookup or data augmentation from internet sources will result in immediate disqualification Fair Play: This challenge is designed to test your machine learning and data science skills using only the provided training data. External price lookup defeats the purpose of the challenge.

.

# 🚀 Proposed Solution

The Smart Product Pricing Solution is designed as a high-performance multimodal regression pipeline that predicts product prices by combining:

## Rich Text Feature Engineering

Extraction of linguistic patterns from catalog descriptions

Sentence-level statistics, bullet point signals, and premium keyword indicators

TF-**IDF** n-gram representations to capture semantic patterns

## Structured Feature Extraction

Parsing units (oz, ml, kg), pack sizes, case quantities

Numerical pattern mining such as average numbers, large-number counts

Category detection using keyword clusters

## Image-Based Feature Extraction

To enhance prediction accuracy, product images are incorporated using EfficientNet-B0 (or MobileNet for ultra-fast execution). This provides a **1280**-dimensional embedding representing visual characteristics such as:

Packaging quality

Brand appearance

Product size & shape

Material/texture

Premium vs budget presentation

## Unified Feature Fusion

All extracted features—manual **NLP** features, TF-**IDF** vectors, image embeddings, and categorical encodings—are combined into a single enhanced feature matrix.

## CatBoost Regression Core

A tuned CatBoost model serves as the final estimator due to:

Strong performance on noisy heterogeneous features

Native handling of sparse & dense inputs

Fast training and inference

Robustness to outliers

## Log-Transformed Target

The price target is transformed using log1p(y) to stabilize variance and reduce sensitivity to extreme high-price products, improving **SMAPE** significantly.

This pipeline is optimized for both high accuracy and fast execution, achieving competitive leaderboard performance.

🏗️ Architecture Used (Multimodal Price Prediction Pipeline)

The overall system follows a multistage multimodal architecture:

## Input Layer

catalog_content text

image_path (product images)

Structured metadata (unit, value, pack info, etc.)

## Processing Pipelines

- Text Feature Pipeline

Regex-driven feature extraction

Text statistics (length, ratio, counts)

TF-**IDF** n-gram embeddings

- Image Processing Pipeline

Image loading & preprocessing

EfficientNet-B0 embedding extraction

**1280**-dimensional feature vector per image

- Structured Feature Pipeline

Numeric extraction

Category detection

Unit & size parsing

Ratio computations

## Feature Fusion Layer

Concatenates:

Manual text features

TF-**IDF** features

Image embeddings

Encoded categorical variables

Size/pack indicators

## Scaling & Normalization

RobustScaler normalizes only numeric features to reduce outlier impact.

## CatBoost Regression Layer

LossGuide tree growth

Bayesian bagging

Newton leaf estimation

Early stopping

**RMSE** loss on log-price

## Output Layer

Reverse log transform

Final price predictions

🔍 Key Insights After Integrating Image Data ## Image embeddings significantly boost accuracy

Visual signals such as:

premium packaging

recognizable brand logos

volume/size cues

material quality help the model make better predictions for items where text is incomplete or ambiguous.

## Multimodal learning reduces SMAPE variance

Text-only models struggle for categories like:

clothing

cosmetics

electronics accessories Image features fill these gaps effectively.

## Feature density increases model robustness

With over **2000**+ combined features, CatBoost can learn richer nonlinear relationships, improving price estimation across all categories.

## EfficientNet-B0 balances accuracy and speed

Fast embeddings (<20 ms per image on **CPU**)

High-quality representations

Ideal for large catalogs

## Log-transform + outlier trimming improves stability

Using:

2nd–98th percentile trimming

log1p transformation

dramatically lowers prediction error for very high-priced products. 
#Architecture Diagram 🏗️ Model Architecture

<img width="657" height="883" alt="image" src="https://github.com/user-attachments/assets/71c7617e-52fc-4ba3-bbdd-6f74eb5c02a5" />

## Architecture Design
<img width="1464" height="1430" alt="Untitled diagram-2025-12-05-023953" src="https://github.com/user-attachments/assets/1bf1e26e-c8ff-4c6d-a864-3fe96fdb2aee" />



# ⚙️ Implementation Details

The Smart Product Pricing Solution is built as a multimodal regression pipeline that integrates text, image, and structured metadata to predict accurate product prices. The system is optimized for:

High accuracy

Low latency

Scalability for large catalogs

Efficient training (log-transformed target & outlier handling)

The core components include:

### Textual Feature Engineering

Regex-based extraction

Token statistics

Bullet-point density

TF-**IDF** 1–2 gram embeddings (**1000** dimensions)

### Image Embedding Extraction

EfficientNet-B0 for lightweight high-quality embeddings

**1280**-dimensional vector per image

Normalized preprocessed inputs (**224**×**224**)

### Structured Feature Parsing

Size / weight / volume extraction

Unit normalization (oz, ml, g, kg, lb)

Category keyword detection

Pack & case size estimation

Numeric feature distribution statistics

### Feature Fusion Layer

All extracted feature groups are merged into a dense feature matrix.

Model Training via CatBoost Regressor

Excellent performance on mixed-type features

Handles sparse TF-**IDF** + high-dim embeddings efficiently

Price Post-processing

Reverse log transformation

Min-value clipping to avoid negative predictions

## 🧰 Key Technologies

| Component               | Technology Used                  |
|-------------------------|----------------------------------|
| Programming Language    | Python 3.x                       |
| ML Framework            | CatBoost                         |
| NLP Toolkit             | scikit-learn (TF-IDF)            |
| Image Embeddings        | EfficientNet-B0 (TensorFlow/Keras) |
| Feature Scaling         | RobustScaler                     |
| Data Handling           | pandas, numpy                    |
| Visualization (optional)| matplotlib, seaborn              |
| Deployment-Ready        | Fast optimized prediction pipeline |

---

## 🔧 Model Parameters Breakdown (CatBoost)

| Parameter                | Value          |
|--------------------------|----------------|
| iterations               | 1200           |
| learning_rate            | 0.05           |
| depth                    | 9              |
| l2_leaf_reg              | 3              |
| bootstrap_type           | Bayesian       |
| bagging_temperature      | 1              |
| rsm                      | 0.8            |
| grow_policy              | LossGuide      |
| max_leaves               | 31             |
| leaf_estimation_method   | Newton         |
| early_stopping_rounds    | 50             |
| loss_function            | RMSE (log1p(price)) |

Below are the tuned parameters used in the optimized solution:

## 🔧 CatBoost Parameter Details

| Parameter               | Value        | Purpose                                       |
|-------------------------|--------------|------------------------------------------------|
| iterations              | **1200**     | Deep boosting for high accuracy                |
| learning_rate           | 0.05         | Stable convergence                             |
| depth                   | 9            | Tree depth for non-linear patterns             |
| l2_leaf_reg             | 3            | Regularization                                 |
| loss_function           | **RMSE**     | Works well with log-transformed target         |
| early_stopping_rounds   | 50           | Avoids overfitting                             |
| bootstrap_type          | Bayesian     | More robust sampling                           |
| bagging_temperature     | 1            | Diversifies trees                              |
| rsm                     | 0.8          | Random feature sampling                        |
| grow_policy             | Lossguide    | Optimized for large feature sets               |
| max_leaves              | 31           | Efficient tree size control                    |
| leaf_estimation_method  | Newton       | Faster and more accurate updates               |
| score_function          | Cosine       | Improves split selection                       |

These parameters are optimized to handle dense, sparse, and high-dimensional embeddings simultaneously.

### Training Configuration

✔ Training Sample Size

Up to 15,**000** sampled rows (for speed + balance)

✔ Train/Validation Split Train: 80% Validation: 20%

✔ Target Transformation

y_log = log1p(price)

Smooths heavy-tailed distribution

Improves model stability and **SMAPE**

✔ Outlier Handling

Removes 2nd and 98th percentile prices

Prevents extreme values from skewing training

✔ Optimized Prediction Pipeline

Prediction performed in **20K** row batches

Works efficiently even on very large datasets

🧹 Data Preprocessing ## Text Processing

Lowercasing

Regex statistics

Count features (words, sentences, digits, caps)

TF-**IDF** vectorization

Bullet point structural features

## Image Preprocessing

Resize to **224**×**224**

EfficientNet preprocessing

Extract embeddings → flatten to **1280**-d vector

## Numeric Feature Engineering

Extract value, pack size, case size, size measurement

Compute statistical numeric summaries

Ratios:

value_to_word_ratio

pack_size_to_value_ratio

bullet_to_word_ratio

## Categorical Encoding

LabelEncode:

unit

size_unit

## Feature Consolidation

Combine:

Manual numeric features

TF-**IDF** vectors

Image embeddings

Encoded categorical features

## Scaling

RobustScaler applied only to numeric full feature matrix

TF-**IDF** and embeddings do not need further scaling

# 📈 Validation Results

Below is the model’s validation performance on the private validation split:

Metric	Score
**SMAPE**	≈ 54.1

**MAE**	Low variance
Performance Stability	High consistency across product categories

Notable improvements after adding image embeddings:

Better price estimation for beauty, clothing, accessory, and premium items

Improved predictions for short or poorly written product descriptions

## 🧪 Overall Performance

Accuracy: Strong, with competitive **SMAPE**

Speed: Very fast — optimized for large-scale inference

Robustness: Works well on noisy product descriptions

## Scalability:

Batch inference

EfficientNet embeddings cached for large test sets

Multimodal Advantage: Combining text + image + metadata improves generalization dramatically

The solution achieves top-tier performance while keeping runtime minimal.
## ⚙️ Implementation Details
###  Core Components
|            Component | Technology           |
| -------------------: | -------------------- |
| Programming Language | Python 3.x           |
|        Text Features | Regex, Stats, TF-IDF |
|       Image Features | EfficientNet-B0      |
|  Structured Features | Unit & pack parsing  |
|             ML Model | CatBoost Regressor   |
|              Scaling | RobustScaler         |
|        Visualization | matplotlib, seaborn  |
Key CatBoost Parameters

iterations: **1200**

learning_rate: 0.05

depth: 9

l2_leaf_reg: 3

bootstrap_type: Bayesian

bagging_temperature: 1

rsm: 0.8

grow_policy: LossGuide

max_leaves: 31

leaf_estimation_method: Newton

early_stopping_rounds: 50

loss_function: **RMSE** (trained on log1p(price))

### Data Preprocessing

Text

Lowercasing & control character removal

Regex extraction: numbers, units, sizes

Word/sentence/digit/capital counts & unique ratios

Bullet point structure extraction

TF-**IDF** n-grams (1,2) with max_features=**1000**

Images

Download images to dataset/images/

Resize to 224x224

EfficientNet-B0 preprocessing & avg_pool embedding extraction

**1280**-dimensional flattened embedding per image

### Structured Metadata

Extract value, pack_size, case_size

Unit normalization and size_value detection

Compute ratios: value_to_word_ratio, pack_size_to_value_ratio, bullet_to_word_ratio

Categorical Encoding & Scaling

LabelEncode unit, size_unit (store encoders)

RobustScaler on numeric features (applied after fusion)

TF-**IDF** and image embeddings concatenated (no additional scaling required)

# 📊 Data Exploration & **EDA**

### Key Observations

Price distribution is heavily right-skewed

Log transform produces near-normal distribution

Catalog content length ranges widely (10 → 8,**000** chars)

Text features alone have weak linear correlation with price; need nonlinear multimodal model

<img width="4470" height="2966" alt="data_exploration_plots" src="https://github.com/user-attachments/assets/3b1ce092-afb2-4f1a-903d-c4627e274259" />               <img width="3565" height="2365" alt="feature_analysis_plots" src="https://github.com/user-attachments/assets/9e399f05-1b62-4258-8fb3-381d1dfa55e2" />


## 🔍 Feature Analysis

Findings

Price distribution differs significantly by unit type (Fl Oz, Count, Ounce, Pound)

Items with more bullet points often have higher average price

Packaged vs non-packaged shows different mean prices

value vs price is non-linear with extreme outliers

### 🔑 Key Insights

Image embeddings significantly boost accuracy — packaging quality, logos, visual size cues and textures inform price.

Multimodal learning reduces **SMAPE** variance — especially effective in categories like cosmetics, apparel, and accessories.

Feature density increases robustness — combining TF-**IDF** + structured + image embeddings gives strong signal (>**2000** combined features).

EfficientNet-B0 offers great speed/accuracy tradeoff — lightweight and produces stable embeddings.

Log-transform + percentile trimming improves training stability — 2nd–98th percentile trimming + log1p target reduces sensitivity to high-price items.

# 📈 Validation Results
Metric	Score
**SMAPE**	54.1%
**MAE**	Low
Performance Stability	High across product categories

Improvements from images: visual features improved predictions where textual descriptions were sparse or ambiguous.

# ▶️ Running Guide 
## Install dependencies
pip install -r requirements.txt

## Prepare dataset

Place train.csv and test.csv in dataset/

Download images to dataset/images/ (use src/utils.py download helper)

3. Train
python fast_40_smape_optimized.py

## Predict

python predict.py --input dataset/test.csv --output dataset/test_out_fast_40_smape.csv

## Output format

sample_id,price 1,12.34 2,45.90 ...
<img width="342" height="617" alt="image" src="https://github.com/user-attachments/assets/a35c9b59-fceb-420d-8c33-3b3e67305f0f" />


# 📁 Project Structure 
<img width="687" height="667" alt="image" src="https://github.com/user-attachments/assets/e952607e-0136-46b2-a1e3-b36ce2b3e6a0" />
