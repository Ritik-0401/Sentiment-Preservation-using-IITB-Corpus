# Sentiment-Preservation-using-IITB-Corpus
Sentiment analysis and preservation between English and Hindi.

This project aims to analyze and preserve sentiment across English and Hindi languages using machine learning models. The core idea is to train a sentiment classification model on English text and then evaluate its ability to preserve sentiment when applied to Hindi translations of the same English text.

##  Project Process

### Data Loading and Analysis
The project utilizes the IITB Parallel Corpus for English-Hindi parallel sentences.

Corpus Files: IITB.en-hi.en, IITB.en-hi.hi for training, and dev.en, dev.hi, test.en, test.hi for development and testing.

Data Preparation: English and Hindi sentences are loaded into pandas DataFrames. Basic analysis includes checking dataset sizes (Training: ~1.66M, Dev: 520, Test: 2507 sentences) and sentence length statistics for both languages. Empty or invalid rows are removed to ensure data quality.

### Sentiment Label Assignment
To obtain sentiment labels for the English text, a pre-trained sentiment analysis model is employed.

Model Used: distilbert-base-uncased-finetuned-sst-2-english from the Hugging Face Transformers library.

Labeling Process: This model classifies English sentences into "POSITIVE" (mapped to 0) or "NEGATIVE" (mapped to 1) sentiments. A fallback for neutral sentiment (mapped to 2) is also included, though the primary focus appears to be on binary classification. Sentiment labels are assigned to the English sentences in the training, development, and test sets.

### Preprocessing and Tokenization
The text data is prepared for model input using a pre-trained tokenizer.

Tokenizer Used: xlm-roberta-base tokenizer. This choice is suitable for multilingual tasks as XLM-RoBERTa is a large multilingual language model.

Dataset Creation: A custom SentimentDataset class is defined to handle tokenization, truncation (to a maximum length of 128 tokens), padding, and conversion of text and labels into PyTorch tensors. This prepares the data for training a sequence classification model.

### Model Training
A pre-trained multilingual model is fine-tuned for sequence classification on the sentiment-labeled English data.

Model Architecture: xlm-roberta-base is loaded as AutoModelForSequenceClassification with 3 output labels (corresponding to positive, negative, and potentially neutral sentiments).

Training Configuration:

Device: The model is trained on the CPU (though GPU acceleration is supported if available).

Batch Size: per_device_train_batch_size and per_device_eval_batch_size are set to 16.

Epochs: The model is trained for 3 epochs.

Learning Rate: The learning rate is set to 2e-5.

Evaluation and Saving Strategy: Evaluation and model saving are performed at the end of each epoch (evaluation_strategy="epoch", save_strategy="epoch").

Best Model: The best model based on eval_loss is loaded at the end of training.

Mixed Precision Training: fp16=True is enabled for faster training if a compatible GPU is used.



## Algorithms and Models
Sentiment Classifier (English Labeling):

distilbert: A distilled version of BERT, pre-trained for sentiment analysis on English text. Used to generate initial sentiment labels for the English corpus.

Multilingual Sentiment Model (Fine-tuning):

xlm-roberta: A RoBERTa-based multilingual language model. This model is fine-tuned on the English sentiment-labeled data to learn sentiment representations that can potentially transfer to other languages (Hindi in this case) due to its multilingual pre-training.

## Evaluation Metrics
The project evaluates the model's performance on the English test set and its ability to preserve sentiment when translating to Hindi.

### English Test Set Evaluation
The following metrics are used to assess the model's performance on the English test set:

Precision: The proportion of correctly predicted positive/negative instances among all instances predicted as positive/negative.

Recall: The proportion of correctly predicted positive/negative instances among all actual positive/negative instances.

F1-score: The harmonic mean of precision and recall, providing a single metric that balances both.

Accuracy: The overall proportion of correctly classified instances.

Macro Average F1-score: The average of the F1-scores for each class, without considering class imbalance.

Weighted Average F1-score: The average of the F1-scores for each class, weighted by the number of true instances for each class.

Confusion Matrix: A table showing the number of true positive, true negative, false positive, and false negative predictions.

### Sentiment Preservation Accuracy (English vs Hindi)
This is a crucial metric for evaluating the cross-lingual aspect of the project.

It measures the percentage of instances where the sentiment predicted for the English text matches the sentiment predicted for its Hindi translation. This is calculated by comparing the sentiment_label of the English text with the predicted label for the Hindi text (hindi_labels).



## Results
### Model Training Performance
The training loss decreased significantly over epochs, indicating that the model was learning from the training data. However, the validation loss increased, suggesting potential overfitting to the training data.

![image](https://github.com/user-attachments/assets/d5a91119-f518-42ed-af02-417e67e796f6)

### English Test Set Evaluation
The model achieved an overall accuracy of 0.71 (71%) on the English test set. It performed better at predicting "negative" sentiment (higher precision and F1-score) compared to "positive" sentiment.

![image](https://github.com/user-attachments/assets/45e1a4d6-453a-46a2-9dc1-3a06be9ad467)

### Confusion Matrix
This matrix further illustrates the model's performance, showing the counts of correct and incorrect predictions for each class.

![image](https://github.com/user-attachments/assets/fcef6fdd-891d-4d57-8d96-e294f2f73a7b)


### Sentiment Preservation
Sentiment Preservation Accuracy: 63.26%
