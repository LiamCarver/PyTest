from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Export & optimize the model for ONNX + DirectML
model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    export=True,                     # <- triggers ONNX export
    provider="DmlExecutionProvider"  # <- GPU backend (AMD / Intel / DirectML)
)

export_path = "./distilbert_onnx"
model.save_pretrained(export_path)
tokenizer.save_pretrained(export_path)

print(f"✅ Export complete. Files saved to: {export_path}")