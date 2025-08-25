from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# add the path to my llm
model_path = "./SmolLM2-1.7B-Instruct"


# load tokenizer and model. Also added torch_dtype="auto", which saves memory (makes it faster) by compressing words sizes to smaller ones.
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")

# Creating a text-generaiton pipeline (basically adding a toolkit that already transforms and simplifies communication with ai)
chatbot = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens=60
)

# Creating a function to turn on the AI for you to interact with
while True:
    user_input = input("Your prompt: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = chatbot(user_input)[0]["generated_text"]
    print(f'AI:\n{response}')