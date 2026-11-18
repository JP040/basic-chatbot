from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for subsequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

print("Chatbot ready! (type 'exit' to quit)\n")

history_string = "\n".join(conversation_history)

input_text = input("> ")

prompt = history_string + f"\nUser: {input_text}\nBot:"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)
