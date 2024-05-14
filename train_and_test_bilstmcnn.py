from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

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
                if j == len(sentences[i])-1:
                    f.write("\n")

def save_predictions_CONLL(sentences, pred_labels):
    with open('Predictions_BiLSTM.conll', 'w', encoding='utf-8') as f:
        f.write("-DOCSTART- -X- -X- O\n\n")
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                f.write(f"{sentences[i][j]} -X- -X- {pred_labels[i][j]}\n")
                if j == len(sentences[i])-1:
                    f.write("\n")

TRAIN = 'conll2003-ner/train.txt'
VALIDATE = 'conll2003-ner/valid.txt'
STARWARS = 'starwars-data/StarWars_Full.conll'
#TRAIN = 'baseline-data/en_ewt-ud-train_CONV.iob2'
#VALIDATE = 'baseline-data/en_ewt-ud-dev_CONV.iob2'
EMBEDDINGS = 'embeddings/glove.6B.100d.txt'

from models.BiLSTM_CNN import preprocessing
from models.BiLSTM_CNN_2 import train

sw_sentences, sw_labels = preprocessing.get_lines(STARWARS)
_, test_sentences, _, test_labels = train_test_split(
    sw_sentences, sw_labels, test_size=0.2, random_state=42
)

predictions = train.train_model(TRAIN, VALIDATE, STARWARS, EMBEDDINGS, epochs=30, finetune=True)

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

test_labels, predicted_labels = adjust_predicted_padding(test_labels, predicted_labels)

def debug_compare_list_lengths(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same number of sublists.")
    
    for i in range(len(list1)):
        if len(list1[i]) != len(list2[i]):
            print(f"Mismatch at index {i}: List 1 has length {len(list1[i])}, List 2 has length {len(list2[i])}")

debug_compare_list_lengths(test_labels, predicted_labels)
save_predictions_CONLL(test_sentences, predicted_labels)

accuracy = accuracy_score(test_labels, predicted_labels) 
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)