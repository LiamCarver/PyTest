import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_device():
    """
    Select the GPU device provided by DirectML.
    """
    return torch_directml.device()


def load_model_and_tokenizer(model_name, device):
    """
    Load the tokenizer and model into memory, and move the model to the GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    return model, tokenizer


def preprocess_text(text, tokenizer, device):
    """
    Convert raw text into tokenized tensors suitable for the model.
    Sends tensors to the GPU.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    return inputs


def predict_sentiment(model, inputs):
    """
    Run the model on the input tensors and return the predicted label (0 or 1).
    """
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = logits.argmax(dim=1).item()
    return prediction


def interpret_prediction(prediction):
    """
    Convert the model’s numeric prediction into a human-readable label.
    """
    return "✅ Positive" if prediction == 1 else "❌ Negative"


def main():
    # Step 1: Select GPU
    device = get_device()

    # Step 2: Load model + tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Step 3: Text to analyze
    text = "I did not enjoy learning how to run models locally!"

    # Step 4: Convert text to tensors
    inputs = preprocess_text(text, tokenizer, device)

    # Step 5: Run model and decode output
    prediction = predict_sentiment(model, inputs)
    result = interpret_prediction(prediction)

    # Step 6: Display result
    print(f"Text: {text}")
    print(f"Sentiment: {result}")


if __name__ == "__main__":
    main()
