# Overview

This project focuses on developing a robust Multi-Modal Machine Learning model to predict Airbnb prices in Melbourne. The dataset comprises 275 diverse features, including property details, location attributes, and host information. The process began with Exploratory Data Analysis (EDA) to identify trends, correlations, and patterns, followed by Data Transformation, where comprehensive feature engineering, scaling, and encoding were applied to prepare the data.

To incorporate textual features like property descriptions, BERT embeddings with max pooling were employed to extract meaningful contextual representations, which were seamlessly integrated with numerical and categorical features to enhance model performance. Machine Learning models, such as XGBoost, were trained and fine-tuned using Optuna to achieve optimal results. The model's performance, evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on a testing dataset, demonstrated competitive results that outperformed benchmark scores from the author’s [GitHub repository](https://github.com/georgian-io/Multimodal-Toolkit). 

# Dataset

The Airbnb Prices Dataset can be found [here](https://github.com/georgian-io/Multimodal-Toolkit/tree/master/datasets/Melbourne_Airbnb_Open_Data). The dataset consists of 18,297 training samples, 2,288 validation samples, and 2,288 test samples.

# Exploratort Data Analysis(EDA)

This section presents a straightforward Exploratory Data Analysis (EDA) to explore features related to pricing. The following visualizations offer valuable insights into the data:

**Example of Numerical Features**

- **Left Image: The price range does not increase consistently with the number of bedrooms**
- **Right Image: There appears to be no clear correlation between the number of rooms and price, as properties with 2 rooms can have prices comparable to those with 8 rooms. This suggests that other factors might play a more significant role in determining price.**
  
  <table>
<tr>
    <td><img src="https://github.com/user-attachments/assets/dc91f5b3-008b-4d82-a30c-46ab94778ec4" alt="Image 1" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/cdcc3fb0-e29a-46d3-9731-e02cc1786293" alt="Image 2" width="400"/></td>
</tr>
</table>

# Data Proprcessing and Transformation

## 1) Target Variable (Price):

- Checked minimum and maximum values for outliers.
- Removed prices in the top 5% and bottom 95% of the dataset to eliminate extreme outliers.

## 2) Feature Selection and Transformation:
**Numerical Features:**
- Removed features with variance inflation factors (VIF) > 20 to address multicollinearity.
- Applied Box-Cox transformation to normalize skewed features (skewness > 0.8).
- Retained features with statistical significance (p-value < 0.05).
  
**Categorical Features:**
- Evaluated importance using Chi-Square test
- Retained the top 70 features based on scores.

## 3) Textual Features:
- Used BERT-base-multilingual-cased to generate embeddings for property descriptions.
- Aggregated embeddings using Max Pooling for better contextual representation.

# Machine Learning

A series of experiments were conducted to predict Airbnb prices using different data modalities (text, metadata, and combined features). We also explored a range of machine learning architectures, including hybrid models that combined text-based embeddings with structured metadata. 

Below is a breakdown of the features used, methods applied, and the evaluation metrics (Mean Absolute Error - MAE, and Root Mean Squared Error - RMSE) for each experiment.

## Numerical & Categorical Features only

| **#** | **Features**       | **Methods**                        | **Test MAE** | **Test RMSE** |
|-------|---------------------|------------------------------------|--------------|---------------|
| 2    | Meta Only          | 2-Layer MLP                      | **46.577**     | **136.621**      |
| 3     | Meta Only          | Fine-Tuned XGB                    | **44.115**     | **138.052**      |

## Text Features only

| **#** | **Features**       | **Methods**                        | **Test MAE** | **Test RMSE** |
|-------|---------------------|------------------------------------|--------------|---------------|
| 1     | Combined Text Only          | BERT Embedding → Linear                     | **77.353**     | **178.627**      |

## Combined Text & Numerical & Categorical Features only

| **#** | **Features**       | **Methods**                        | **Test MAE** | **Test RMSE** |
|-------|---------------------|------------------------------------|--------------|---------------|
| 4     | Combined Text + Meta        | BERT Embedding + Meta → Linear MLP          | **47.210**     | **137.606**      |
| 6     | Combined Text + Meta        | BERT Embedding + Meta →  Fine-Tuned XGB       | **45.693**     | **140.799**      |
| 11    | Combined Text + Meta        | BERT (MaxPool) + Meta →  Fine-Tuned XGB       | **44.208**     | **137.736**      |
| 12    | Combined Text + Meta        | BERT (MaxPool) + Meta → 2-Layer MLP | **46.254**     | **140.097**      |

## Text & Numerical & Categorical Features only

| **#** | **Features**       | **Methods**                        | **Test MAE** | **Test RMSE** |
| 5     | Text + Meta        | TF-IDF + Meta →  Fine-Tuned XGB               | **43.788**     | **138.038**      |
| 7     | Text + Meta        | BERT (MaxPool) + Meta →  Fine-Tuned XGB       | **43.551**     | **136.251**      |
| 8     | Text + Meta        | BERT (MeanPool) + Meta →  Fine-Tuned XGB      | **43.775**     | **136.390**      |
| 9     | Text + Meta        | BERT (MaxPool) + Meta → 2-Layer MLP | **45.852**     | **137.473**      |
| 10     | Text + Meta        | BERT (MeanPool) + Meta → 2-Layer MLP | **46.319**     | **137.098**      |

## Convert all features to text features

| **#** | **Features**       | **Methods**                        | **Test MAE** | **Test RMSE** |
| 13    | Convert all features to text       | BERT Embedding → Linear       | **78.446**     | **176.159**      |

- **Model 1**: Used only text features via a BERT-based model and performed reasonably well.
- **Model 3**: Fine-tuned XGBRegressor on meta features delivered strong results for numerical data alone.
- **Model 5**: TF-IDF with meta features produced competitive results with fewer resources.
- **Model 6 & 7**: Pooling strategies with BERT embeddings demonstrated promising performance when combined with meta data.

# Feature Detail 
## Meta 
- Represents numerical and categorical attributes directly as structured metadata.

## Text Features 
- Text features are kept in their own separate columns for isolated analysis.

## Combined Text 
- When combining multiple text columns, individual text values are concatenated using a separator token, such as [SEP].
- 
Example:
> "We're in the US for 3 weeks [SEP] I’m an Aussie living in Uganda. [SEP] "

## Convert all features to text 

-Numerical and categorical features are transformed into a textual representation to unify feature processing.

> The space is separated from the living room by a screen. However it is as big as a bedroom and has a 'window' (actually it's a door) towards the balcony (east-facing). There are a desk, a chair, a lamp, a double size mattress & a portable wardrobe. You will have two flat mates in their 20s and a bunny pet to keep you company. Please NOTE that the living room are affected by NOISE and SMELL from cooking. [SEP] Unknown [SEP] This Airbnb offers amenities such as: hair_dryer, shampoo, heating, and smoking_allowed. It accommodates to 2 guests, with 2 bedrooms, 2 bathrooms, and 2 beds. The cleaning fee costs $11.0. Available for 0 days in the last 30 days and 0 days in the last year. 19 review.

# Best Performing Model
Pipeline: BERT (MaxPool) embeddings combined with meta features, followed by Fine-Tuned XGB.
MAE: 43.551
RMSE: 136.251

### Notes
- **BERT Embedding**: Feature vectors extracted from a pre-trained BERT model.
- **MaxPooling/MeanPooling**: Methods to aggregate BERT embeddings into fixed-length vectors.
- **Meta Data**: Includes numerical and categorical information processed through an MLP or concatenated with text features.

