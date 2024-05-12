import numpy as np
import tensorflow as tf
from seqeval.metrics import precision_score, recall_score, f1_score

def adjust_predicted_padding(original_labels, predicted_labels, padding_token='PADDING', replace_with='O'):
    adjusted_original_labels = []
    adjusted_predicted_labels = []

    for orig, pred in zip(original_labels, predicted_labels):
        orig_length = len(orig)

        # Trim predicted labels if longer than the original labels
        trimmed_pred = pred[:orig_length]

        # Extend predicted labels with 'O' if shorter than the original labels
        if len(trimmed_pred) < orig_length:
            trimmed_pred += [replace_with] * (orig_length - len(trimmed_pred))

        # Replace any instance of padding_token in the predicted labels with replace_with
        adjusted_pred = [label if label != padding_token else replace_with for label in trimmed_pred]

        adjusted_original_labels.append(orig)
        adjusted_predicted_labels.append(adjusted_pred)

    return adjusted_original_labels, adjusted_predicted_labels

def save_predictions_IOB2(sentences, pred_labels):
    with open('predictions.txt', 'w', encoding='utf-8') as f:
        for i in range(len(sentences)):
            f.write(f"\n# sneed\n")
            f.write("# text = " + " ".join([word for word in sentences[i]]) + "\n")

            for j in range(len(sentences[i])):
                f.write(f"{j+1}\t{sentences[i][j]}\t{pred_labels[i][j]}\t-\t-\n")

TRAIN = 'conll2003-ner/train.txt'
VALIDATE = 'starwars-data\StarWars_Full.conll'
#TRAIN = 'baseline-data/en_ewt-ud-train_CONV.iob2'
#VALIDATE = 'baseline-data/en_ewt-ud-dev_CONV.iob2'
EMBEDDINGS = 'embeddings/glove.6B.100d.txt'

from models.BiLSTM_CNN import preprocessing
from models.BiLSTM_CNN_2 import train

val_sentences, val_labels = preprocessing.get_lines(VALIDATE)
predictions = train.train_model(TRAIN, VALIDATE, EMBEDDINGS, 30)

idx2label = {1:'B-ORG',
            2:'O',
            3:'B-MISC',
            4:'B-PER',
            5:'B-LOC',
            6:'I-PER',
            7:'I-ORG',
            8:'I-LOC',
            9:'I-MISC',
            0:'PADDING'}

# Function to map indices to labels
def indices_to_labels(indices, mapping):
    return [[mapping[idx] for idx in sequence] for sequence in indices]

predicted_labels = indices_to_labels(predictions[0], idx2label)

val_labels, predicted_labels = adjust_predicted_padding(val_labels, predicted_labels)

def debug_compare_list_lengths(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same number of sublists.")
    
    for i in range(len(list1)):
        if len(list1[i]) != len(list2[i]):
            print(f"Mismatch at index {i}: List 1 has length {len(list1[i])}, List 2 has length {len(list2[i])}")

debug_compare_list_lengths(val_labels, predicted_labels)
save_predictions_IOB2(val_sentences, predicted_labels)

precision = precision_score(val_labels, predicted_labels)
recall = recall_score(val_labels, predicted_labels)
f1 = f1_score(val_labels, predicted_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)