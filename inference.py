from llama_cpp import Llama
import time

def run_gguf():
	llm = Llama.from_pretrained(
		repo_id = "unsloth/gemma-3-270m-it-GGUF",
		filename = "gemma-3-270m-it-Q8_0.gguf",
	)
 
	t1 = time.time()
	output = llm(
		"How are you bro",
		max_tokens = 30,
		echo = True
	)
	t2 = time.time()
	print(f"\n\nAI({(t2-t1)*1000:.2f}ms): ", output['choices'][0]['text'],  "\n\n")


# Transformers
def run_transformers():
	from transformers import pipeline

	pipe = pipeline("text-generation", model = "google/gemma-3-270m")
	print("Model loaded")

	t1 = time.time()
	result = pipe("how are you")
	t2 = time.time()
	print(f"{(t2-t1)*1000:.2f} ms: {result[0]['generated_text']}")



run_gguf()