import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer from the exported folder
tokenizer = AutoTokenizer.from_pretrained("./distilbert_onnx")

# Create ONNX Runtime session using DirectML GPU
session = ort.InferenceSession(
    "./distilbert_onnx/model.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)

def classify(text: str):
    # Convert input text into token IDs
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    ort_inputs = {k: v for k, v in inputs.items()}

    # Run inference
    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    # Softmax to convert logits → probabilities
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    neg, pos = probabilities[0]

    return {
        "text": text,
        "positive": float(pos),
        "negative": float(neg),
        "label": "POSITIVE" if pos > neg else "NEGATIVE"
    }

# Test Examples
print(classify("I love this. This is great."))
print(classify("This is awful. I hate it."))