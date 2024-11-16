# Airbnb-Price-Prediction-

bert-base-multilingual-cased

# Model Comparison Results

| **Features**       | **Methods**                                                | **Test MAE** | **Test RMSE** |
|---------------------|------------------------------------------------------------|--------------|---------------|
| Text Only          | BERT Model + Linear Layer                                   | X.XX         | X.XX          |
| Meta Only          | 2-Layer MLP                                                | X.XX         | X.XX          |
| Meta Only          | Fine-tuned XGBRegressor                                    | X.XX         | X.XX          |
| Text + Meta Data   | Concat (BERT Model Embedding + MLP with Meta Data) + Linear Layers MLP | X.XX         | X.XX          |
| Text + Meta Data   | Concat (TF-IDF (max_features=100) + Meta Data) > Fine-Tuned XGBRegressor | X.XX         | X.XX          |
| Text + Meta Data   | Concat (MaxPooling(BERT Model Embedding) + Meta Data) > Fine-Tuned XGBRegressor | X.XX         | X.XX          |
| Text + Meta Data   | Concat (MeanPooling(BERT Model Embedding) + Meta Data) > Fine-Tuned XGBRegressor | X.XX         | X.XX          |
| Text + Meta Data   | Concat (MaxPooling(BERT Model Embedding) + Meta Data) > 2-Layer MLP | X.XX         | X.XX          |
| Text + Meta Data   | Concat (MeanPooling(BERT Model Embedding) + Meta Data) > 2-Layer MLP | X.XX         | X.XX          |
