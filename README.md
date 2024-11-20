# Overview

This project focuses on developing a robust Multi-Modal Machine Learning model to accurately predict Airbnb prices in Melbourne. The dataset comprises 176 diverse features, including property details, location attributes, and host information. The project comprises three key phases: Exploratory Data Analysis (EDA) to identify trends, correlations, and patterns within the data, Data Transformation involving comprehensive feature engineering, scaling, and encoding to prepare the dataset, and Machine Learning, where models such as XGBoost were trained and fine-tuned using Optuna for optimal performance. To leverage textual features like property descriptions, BERT embeddings with max pooling were utilized to extract meaningful contextual representations, which were seamlessly integrated with numerical and categorical features for improved model result. The model was evaluated on a testing dataset using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), achieving competitive results that outperformed the benchmark results provided in the author’s GitHub repository [here](https://github.com/georgian-io/Multimodal-Toolkit). 



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

