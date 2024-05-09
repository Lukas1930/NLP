# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

text = "Luke Skywalker and Darth Vader are central characters in Star Wars."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
predicted_tokens_classes = [model.config.id2label[t.item()] for t in predictions[0]]
print(predicted_tokens_classes)