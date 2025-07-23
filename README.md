DeBERTa-Based Sentiment Analysis on Amazon Reviews

This project demonstrates fine-tuning Microsoft's DeBERTa (Decoding-enhanced BERT with disentangled attention) on the Amazon Fine Food Reviews dataset to perform binary sentiment classification (positive or negative).

Project Overview
- Goal: Evaluate the performance of DeBERTa on sentiment analysis and compare it with traditional models.
- Dataset: Amazon Fine Food Reviews
- Model: microsoft/deberta-base
- Task: Binary classification (Positive = Score > 3, Negative = Score < 3)

Workflow Summary
1. Data Preprocessing
   - Filtered out neutral reviews (Score = 3)
   - Converted review scores into binary labels

2. Tokenization
   - Used DebertaTokenizer with truncation, padding, and max_length=256

3. Fine-Tuning
   - Used Trainer API from Hugging Face
   - Trained for 3 epochs on a sample of 20000 reviews
   - Evaluation metrics recorded: Accuracy, Precision, Recall, F1-Score

4. Evaluation
   - Assessed model performance on validation set
   - Measured training and inference times

Results (Sample of 20,000 reviews)
Accuracy: 0.9487
Precision: 0.9579
Recall: 0.9817
F1 Score: 0.9697
Training Time: 183.45 seconds
Testing Time: 6.27 seconds

File Structure
- DeBERTa_Sentiment_Analysis.ipynb – Full training and evaluation notebook
- Reviews.csv – Dataset (Amazon food reviews)
- README.md – This file

Requirements
pip install transformers torch scikit-learn pandas

Future Improvements
- Compare performance with BERT, RoBERTa, and DistilBERT
- Add robustness testing (emojis, typos, slang)
- Include zero-shot and few-shot evaluation (DeBERTa-V3)

Model Reference
He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. arXiv preprint arXiv:2006.03654.

License
This project is licensed under the MIT License — see the LICENSE file for details.
