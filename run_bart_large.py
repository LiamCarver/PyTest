from transformers import pipeline

clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

result = clf(
    "I should go to the gym more often.",
    candidate_labels=["health", "finance", "relationships"]
)

print(result)