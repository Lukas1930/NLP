# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm.auto import tqdm

def get_lines(file_path):
    """
    given the path to the dataset it returns it in format
    [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], ...]

    with the respective labels
    [['B-ORG\n', 'O\n', 'B-MISC\n', 'O\n', 'O\n', 'O\n', 'B-MISC\n', 'O\n', 'O\n'], ...]

    :param file_path: (str) path to the file
    :return: (list, list) list of lists where each element is a sentence and each element of
    the sublist is a word in that sentence and respective list of lists with a
    label for each word
    """

    with open(file_path, mode='rt', encoding="utf-8") as f:

        labels_sentence = []  # list of labels of a single sentence
        labels = []  # list of lists with all the sentences
        sentences = []
        sentence = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0 and len(labels_sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                    labels.append(labels_sentence)
                    labels_sentence = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            labels_sentence.append(splits[-1].rstrip('\n'))

    if len(sentence) > 0 and len(labels_sentence) > 0:
        sentences.append(sentence)
        labels.append(labels_sentence)

    return sentences, labels

def separate_special_characters_with_labels(sentences, sentence_labels):
    result_sentences = []
    result_sentence_labels = []
    
    for words, labels in zip(sentences, sentence_labels):
        result_words = []
        result_labels = []
        
        for word, label in zip(words, labels):
            # Check if the word contains '-' or '\''
            if '-' in word or '\'' in word or '.' in word or ',' in word or ':' in word or '/' in word or '$' in word or ')' in word or '(' in word or '=' in word or '*' in word or '+' in word or '#' in word or '&' in word:
                # Split the word by both '-' and '\'' and preserve the separators
                parts = []
                temp = [word]
                for separator in ('-', '\'', '.', ',', ':', '/', '$', ')', '(', '=', '*', '+', '#', '&'):
                    for split_word in temp:
                        expanded = split_word.split(separator)
                        parts.extend([piece if i == len(expanded) - 1 else piece + separator for i, piece in enumerate(expanded)])
                    temp = parts
                    parts = []
                
                # Add parts to the result, treating separators as separate tokens
                for i, part in enumerate(temp):
                    if part.endswith('-') or part.endswith('\'') or part.endswith('.') or part.endswith(',') or part.endswith(':') or part.endswith('/') or part.endswith('$') or part.endswith(')') or part.endswith('(') or part.endswith('=') or part.endswith('*') or part.endswith('+') or part.endswith('#') or part.endswith('&'):
                        part = part[:-1]  # Remove the separator at the end for processing
                        if part:
                            result_words.append(part)
                            result_labels.append(label if i == 0 else ('I' + label[1:] if label.startswith('B') else label))
                        result_words.append(temp[i][-1])  # Add the separator as a separate token
                        result_labels.append('O')  # Assuming 'O' as the label for separators
                    else:
                        if part:
                            result_words.append(part)
                            result_labels.append(label if i == 0 else ('I' + label[1:] if label.startswith('B') else label))
            else:
                result_words.append(word)
                result_labels.append(label)
        
        result_sentences.append(result_words)
        result_sentence_labels.append(result_labels)

    return result_sentences, result_sentence_labels

TEST = r"conll2003-ner\test.txt"
MODEL = "elastic/distilbert-base-uncased-finetuned-conll03-english"

### Initialise model

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL)

### Convert data into proper format

sentences, true_labels = get_lines(TEST)
sentences, true_labels = separate_special_characters_with_labels(sentences, true_labels)

### Make predictions

encodings = []
predicted_labels = []

for i in tqdm(range(len(sentences))):
    sentence = ' '.join(sentences[i])
    encoding = tokenizer(sentence, return_tensors="pt")
    encodings.append(encoding)
    with torch.no_grad():
        outputs = model(**encoding)

    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_labels.append([model.config.id2label[t.item()] for t in predictions[0]])

### Convert and detokenise predictions

condensed_labels = []
debug_sentences = []

for i in tqdm(range(len(predicted_labels))):
    words = []
    word_labels = []
    current_word = ""
    current_label = None

    for id, label in zip(encodings[i].input_ids.squeeze().tolist(), predicted_labels[i]):
        subtoken = tokenizer.decode([id])
        if subtoken.startswith("##"):
            current_word += subtoken[2:]  # Remove '##' and continue the current word
        else:
            if current_word:
                words.append(current_word)
                word_labels.append(current_label)
            current_word = subtoken
            current_label = label

    # Don't forget to add the last accumulated word if it's not a special token
    if current_word and current_word not in [tokenizer.cls_token, tokenizer.sep_token]:
        words.append(current_word)
        word_labels.append(current_label)

    # Exclude the special tokens and their labels
    final_words = [word for word, label in zip(words, word_labels) if word not in [tokenizer.cls_token, tokenizer.sep_token]]
    final_labels = [label for word, label in zip(words, word_labels) if word not in [tokenizer.cls_token, tokenizer.sep_token]]

    # Output the results
    #print(sentences[i])
    #print("Words:", final_words)
    #print("Labels:", final_labels)

    condensed_labels.append(final_labels)
    debug_sentences.append(final_words)

### Check if the predictions are the same length as the true labels

def compare_nested_list_lengths(list1, list2):
    if len(list1) != len(list2):
        print("Warning: The outer lists do not have the same length.")
    
    for index, (sublist1, sublist2) in enumerate(zip(list1, list2)):
        if len(sublist1) != len(sublist2):
            print(f"Length mismatch at index {index}: Length of list1 is {len(sublist1)}, Length of list2 is {len(sublist2)}.")
            print(sentences[index])
            print(debug_sentences[index])

compare_nested_list_lengths(true_labels, condensed_labels)

### Calculate precision, recall, and F1 score

precision = precision_score(true_labels, condensed_labels)
recall = recall_score(true_labels, condensed_labels)
f1 = f1_score(true_labels, condensed_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)