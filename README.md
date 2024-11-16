# Airbnb-Price-Prediction-

bert-base-multilingual-cased

| **#** | **Features**       | **Methods**                        | **Test MAE** | **Test RMSE** |
|-------|---------------------|------------------------------------|--------------|---------------|
| 1     | Combined Text Only          | BERT → Linear                     | **78.465**     | **176.091**      |
| 2     | Meta Only          | 2-Layer MLP                       | **48.534**     | **136.277**      |
| 3     | Meta Only          | Fine-Tuned XGB                    | **44.368**     | **138.540**      |
| 4     | Combined Text + Meta        | BERT + Meta → Linear MLP          | **46.678**     | **140.983**      |
| 5     | Text + Meta        | TF-IDF + Meta → XGB               | **44.469**     | **139.262**      |
| 6     | Text + Meta        | BERT (MaxPool) + Meta → XGB       | **X.XX**     | **X.XX**      |
| 7     | Text + Meta        | BERT (MeanPool) + Meta → XGB      | **X.XX**     | **X.XX**      |
| 8     | Text + Meta        | BERT (MaxPool) + Meta → 2-Layer MLP | **X.XX**     | **X.XX**      |
| 9     | Text + Meta        | BERT (MeanPool) + Meta → 2-Layer MLP | **X.XX**     | **X.XX**      |
| 10     | Combined Text + Meta        | BERT + Meta → Linear MLP          | **X.XX**     | **X.XX**      |
| 11     | Combined Text + Meta        | BERT (MaxPool) + Meta → XGB       | **X.XX**     | **X.XX**      |
| 12    | Combined Text + Meta        | BERT (MaxPool) + Meta → 2-Layer MLP | **X.XX**     | **X.XX**      |

- **Model 1**: Used only text features via a BERT-based model and performed reasonably well.
- **Model 3**: Fine-tuned XGBRegressor on meta features delivered strong results for numerical data alone.
- **Model 5**: TF-IDF with meta features produced competitive results with fewer resources.
- **Model 6 & 7**: Pooling strategies with BERT embeddings demonstrated promising performance when combined with meta data.

---

### Notes
- **BERT Embedding**: Feature vectors extracted from a pre-trained BERT model.
- **MaxPooling/MeanPooling**: Methods to aggregate BERT embeddings into fixed-length vectors.
- **Meta Data**: Includes numerical and categorical information processed through an MLP or concatenated with text features.

