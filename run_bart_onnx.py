import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer from ONNX directory
tokenizer = AutoTokenizer.from_pretrained("bart_mnli_onnx")

# Load ONNX model session with AMD GPU acceleration
session = ort.InferenceSession(
    "bart_mnli_onnx/model.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)

def classify_zero_shot(text, candidate_labels):
    hypotheses = [f"This text is about {label}." for label in candidate_labels]

    encodings = tokenizer(
        [text] * len(candidate_labels),
        hypotheses,
        return_tensors="np",
        padding=True,
        truncation=True
    )

    ort_inputs = {k: v for k, v in encodings.items()}
    logits = session.run(None, ort_inputs)[0]  # shape: [num_labels, 3]

    # Softmax over MNLI classes
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    # Extract probability of ENTAILMENT (index = 2)
    entailment = probs[:, 2]

    # SECOND softmax across labels → match Hugging Face pipeline output
    final_scores = np.exp(entailment) / np.exp(entailment).sum()

    best_label = candidate_labels[np.argmax(final_scores)]
    return best_label, final_scores

text = "I should go to the gym more often."
labels = ["health", "finance", "relationships"]

best, scores = classify_zero_shot(text, labels)

print("Text:", text)
print("Labels:", labels)
print("Scores:", scores)
print("Prediction:", best)