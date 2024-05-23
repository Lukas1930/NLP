## NLP and Deep Learning Exam Project
### Comparing different model types' performance at NER with a Star Wars dataset

##### To run BiLSTM_CNN model:
Download the 6B Wikipedia GloVe embeddings from https://nlp.stanford.edu/projects/glove/ and put them in the embeddings folder
The code is configured to use the 100-dimensional ones, filename glove.6B.100d.txt
Run the train_and_test_bilstmcnn.py file using VS Code or some other IDE
To ensure the code runs properly, use Python 3.11.7 with the packages and versions as specified in BiLSTM_CNN_package_versions.txt

##### To run the Transformer models:
Run the transformer_models.py file using VS Code or some other IDE
To ensure the code runs properly, use Python 3.12.0 with the packages and versions as specified in Transformers_package_versions.txt

##### Credits:
Some code used by the BiLSTM_CNN model was inspired by this repo: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs/tree/master