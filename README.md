# Overview

This project focuses on developing a robust Multi-Modal Machine Learning model to predict Airbnb prices in Melbourne. The dataset comprises 275 diverse features, including property details, location attributes, and host information. The process began with Exploratory Data Analysis (EDA) to identify trends, correlations, and patterns, followed by Data Transformation, where comprehensive feature engineering, scaling, and encoding were applied to prepare the data.

To incorporate textual features like property descriptions, BERT embeddings with max pooling were employed to extract meaningful contextual representations, which were seamlessly integrated with numerical and categorical features to enhance model performance. Machine Learning models, such as XGBoost, were trained and fine-tuned using Optuna to achieve optimal results. The model's performance, evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on a testing dataset, demonstrated competitive results that outperformed benchmark scores from the author’s [GitHub repository](https://github.com/georgian-io/Multimodal-Toolkit). 

# Dataset

The Airbnb Prices Prediction Dataset can be found [here](https://github.com/georgian-io/Multimodal-Toolkit/tree/master/datasets/Melbourne_Airbnb_Open_Data). The dataset consists of 18,297 training samples, 2,288 validation samples, and 2,288 test samples.

# Exploratort Data Analysis(EDA)

This section presents a simple exploratory data analysis (EDA) to understand the some of the feature related to the pricing. Below are two visualizations that provide insights into the data:
  
  <table>
<tr>
    <td><img src="![image](https://github.com/user-attachments/assets/dc91f5b3-008b-4d82-a30c-46ab94778ec4)" alt="Image 1" width="400"/></td>
    <td><img src="![image](https://github.com/user-attachments/assets/06363623-341e-4bc5-8cb0-1dc670d6854a)" alt="Image 2" width="400"/></td>
</tr>
</table>

## Data Proprcessing
Feature Distribution Analysis: Analyzed the distribution of each feature to identify trends and detect anomalies.
Price Range Assessment: Checked the minimum and maximum values of the target variable (price) to understand its range and identify potential outliers.
Manual Feature Selection: Reviewed and manually selected key features based on domain knowledge and their correlation with the target variable.


# Data Transformation

# Machine Learning



bert-base-multilingual-cased

| **#** | **Features**       | **Methods**                        | **Test MAE** | **Test RMSE** |
|-------|---------------------|------------------------------------|--------------|---------------|
| 1     | Combined Text Only          | BERT Embedding → Linear                     | **77.353**     | **178.627**      |
| 2    | Meta Only          | 2-Layer MLP                      | **46.577**     | **136.621**      |
| 3     | Meta Only          | Fine-Tuned XGB                    | **44.115**     | **138.052**      |
| 4     | Combined Text + Meta        | BERT Embedding + Meta → Linear MLP          | **47.210**     | **137.606**      |
| 5     | Text + Meta        | TF-IDF + Meta →  Fine-Tuned XGB               | **43.788**     | **138.038**      |
| 6     | Combined Text + Meta        | BERT Embedding + Meta →  Fine-Tuned XGB       | **45.693**     | **140.799**      |
| 7     | Text + Meta        | BERT (MaxPool) + Meta →  Fine-Tuned XGB       | **43.551**     | **136.251**      |
| 8     | Text + Meta        | BERT (MeanPool) + Meta →  Fine-Tuned XGB      | **43.775**     | **136.390**      |
| 9     | Text + Meta        | BERT (MaxPool) + Meta → 2-Layer MLP | **45.852**     | **137.473**      |
| 10     | Text + Meta        | BERT (MeanPool) + Meta → 2-Layer MLP | **46.319**     | **137.098**      |
| 11    | Combined Text + Meta        | BERT (MaxPool) + Meta →  Fine-Tuned XGB       | **44.208**     | **137.736**      |
| 12    | Combined Text + Meta        | BERT (MaxPool) + Meta → 2-Layer MLP | **46.254**     | **140.097**      |
| 13    | Convert all features to text       | BERT Embedding → Linear       | **78.446**     | **176.159**      |

- **Model 1**: Used only text features via a BERT-based model and performed reasonably well.
- **Model 3**: Fine-tuned XGBRegressor on meta features delivered strong results for numerical data alone.
- **Model 5**: TF-IDF with meta features produced competitive results with fewer resources.
- **Model 6 & 7**: Pooling strategies with BERT embeddings demonstrated promising performance when combined with meta data.

---

### Notes
- **BERT Embedding**: Feature vectors extracted from a pre-trained BERT model.
- **MaxPooling/MeanPooling**: Methods to aggregate BERT embeddings into fixed-length vectors.
- **Meta Data**: Includes numerical and categorical information processed through an MLP or concatenated with text features.

