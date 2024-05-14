from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, EarlyStoppingCallback, TrainerCallback
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

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
            if '-' in word or '\'' in word or '.' in word or (MODEL != "geckos/deberta-base-fine-tuned-ner" and ',' in word) or ':' in word or '/' in word or '$' in word or ')' in word or '(' in word or '=' in word or '*' in word or '+' in word or '#' in word or '&' in word:
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

def save_predictions_CONLL(sentences, pred_labels):
    with open(f'Predictions_BERT.conll', 'w', encoding='utf-8') as f:
        f.write("-DOCSTART- -X- -X- O\n\n")
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                f.write(f"{sentences[i][j]} -X- -X- {pred_labels[i][j]}\n")
                if j == len(sentences[i])-1:
                    f.write("\n")

TEST = r"starwars-data\StarWars_Full.conll"
MODEL = "dslim/bert-base-NER"
FINETUNE = True

### Initialise model

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL)

# Convert data into proper format
sentences, true_labels = get_lines(TEST)
sentences, true_labels = separate_special_characters_with_labels(sentences, true_labels)

# Split the data into training and test sets (remove if we ever need to test on conll again or wrap in an if statement)
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences, true_labels, test_size=0.2, random_state=42
)

### Finetune model
if FINETUNE:
    # Create a custom dataset
    class TokenClassificationDataset(Dataset):
        def __init__(self, sentences, labels, label2id, tokenizer):
            self.sentences = sentences
            self.labels = labels
            self.label2id = label2id
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            sentence = self.sentences[idx]
            labels = [self.label2id[label] for label in self.labels[idx]]

            tokenized_inputs = self.tokenizer(sentence, truncation=True, is_split_into_words=True)
            word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective words.

            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])  # Label only the first token of each word
                else:
                    label_ids.append(-100)  # Subsequent tokens of a word
                previous_word_idx = word_idx
            
            tokenized_inputs['labels'] = label_ids
            return tokenized_inputs

    class LossCurveCallback(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])

    # Split the data into training and evaluation sets
    train_sentences, eval_sentences, train_labels, eval_labels = train_test_split(
        train_sentences, train_labels, test_size=0.2, random_state=42
    )

    # Convert into the Dataset class
    train_dataset = TokenClassificationDataset(train_sentences, train_labels, model.config.label2id, tokenizer)
    eval_dataset = TokenClassificationDataset(eval_sentences, eval_labels, model.config.label2id, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=30,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    loss_curve_callback = LossCurveCallback()

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), loss_curve_callback] 
    )

    trainer.train()

    train_losses = loss_curve_callback.train_losses
    eval_losses = loss_curve_callback.eval_losses

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(eval_losses) + 1), eval_losses, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
### Make predictions

encodings = []
predicted_labels = []

# Get the device model is currently on
device = next(model.parameters()).device

for i in tqdm(range(len(test_sentences))):
    sentence = ' '.join(test_sentences[i])
    encoding = tokenizer(sentence, return_tensors="pt")
    # Move encoding to the same device as model
    encoding = {k: v.to(device) for k, v in encoding.items()}
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

    tokens = encodings[i]["input_ids"].squeeze().tolist() if isinstance(encodings[i], dict) else encodings[i].input_ids.squeeze().tolist()

    for id, label in zip(tokens, predicted_labels[i]):
        subtoken = tokenizer.decode([id])

        if MODEL == 'geckos/deberta-base-fine-tuned-ner':
            if subtoken.startswith(" "):
                if current_word:
                    words.append(current_word)
                    word_labels.append(current_label)
                current_word = subtoken.strip()  # Remove leading space for DeBERTa
                current_label = label
            else:
                if subtoken != tokenizer.sep_token:
                    current_word += subtoken  # Continue building the word for DeBERTa
        else: #BERTlike
            if subtoken.startswith("##"):
                current_word += subtoken[2:]  # Remove '##' for BERT and continue the current word
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
    #print(test_sentences[i])
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
            print(test_sentences[index])
            print(debug_sentences[index])

if MODEL == "geckos/deberta-base-fine-tuned-ner":
    new_test_sentences = []
    new_test_labels = []
    
    # Iterate over each sentence and its corresponding labels
    for sentence, labels in zip(test_sentences, test_labels):
        filtered_sentence = []
        filtered_labels = []
        
        # Iterate over each token and label in the sentence
        for token, label in zip(sentence, labels):
            if token not in [',', '.', '!']:  # Filter out specific unwanted tokens
                filtered_sentence.append(token)
                filtered_labels.append(label)
        
        # Append the filtered sentence and labels to the new lists
        new_test_sentences.append(filtered_sentence)
        new_test_labels.append(filtered_labels)

    # Replace the old lists with the new filtered lists
    test_sentences = new_test_sentences
    test_labels = new_test_labels

compare_nested_list_lengths(test_labels, condensed_labels)

### Calculate precision, recall, and F1 score

precision = precision_score(test_labels, condensed_labels)
recall = recall_score(test_labels, condensed_labels)
f1 = f1_score(test_labels, condensed_labels)

save_predictions_CONLL(test_sentences, condensed_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)