# Airbnb-Price-Prediction-

bert-base-multilingual-cased

| **#** | **Features**       | **Methods**                        | **Test MAE** | **Test RMSE** |
|-------|---------------------|------------------------------------|--------------|---------------|
| 1     | Combined Text Only          | BERT Embedding → Linear                     | **78.502**     | **176.116**      |
| 2     | Meta Only          | 2-Layer MLP                       | **48.534**     | **136.277**      |
| 3     | Meta Only          | Fine-Tuned XGB                    | **44.115**     | **138.236**      |
| 4     | Combined Text + Meta        | BERT Embedding + Meta → Linear MLP          | **46.678**     | **140.983**      |
| 5     | Text + Meta        | TF-IDF + Meta →  Fine-Tuned XGB               | **44.469**     | **139.262**      |
| 6     | Combined Text + Meta        | BERT Embedding + Meta →  Fine-Tuned XGB       | **45.693**     | **140.799**      |
| 7     | Text + Meta        | BERT (MaxPool) + Meta →  Fine-Tuned XGB       | **43.551**     | **136.251**      |
| 8     | Text + Meta        | BERT (MeanPool) + Meta →  Fine-Tuned XGB      | **43.775**     | **136.390**      |
| 9     | Text + Meta        | BERT (MaxPool) + Meta → 2-Layer MLP | **47.033**     | **136.671**      |
| 10     | Text + Meta        | BERT (MeanPool) + Meta → 2-Layer MLP | **48.458**     | **137.182**      |
| 11    | Combined Text + Meta        | BERT (MaxPool) + Meta →  Fine-Tuned XGB       | **44.208**     | **137.736**      |
| 12    | Combined Text + Meta        | BERT (MaxPool) + Meta → 2-Layer MLP | **47.534**     | **139.301**      |
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

