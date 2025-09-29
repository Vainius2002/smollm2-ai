from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# add the path to my llm
model_path = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


# load tokenizer and model. Also added torch_dtype="auto", which saves memory (makes it faster) by compressing words sizes to smaller ones.
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

# Creating a text-generaiton pipeline (basically adding a toolkit that already transforms and simplifies communication with ai)
chatbot = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens=1000,
    device_map = 'auto'
)


history = ''
# Creating a function to turn on the AI for you to interact with
guidance = "You are a sarcastic, dark-humor, unfriendly assistant."
while True:
    user_input = input("Your prompt: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    modified_input = f'{guidance}\n{user_input}\nAI:'


    if len(history.split()) > 400:
        history = ''

        response = chatbot(modified_input)[0]["generated_text"]
        modified_response = response.split('AI:')[-1]
        print(f'AI:\n{modified_response}')

        history += f'User: {user_input}\nAI:{modified_response}'
        print(len(history.split()))
    else:
        history_and_input = f'{guidance}\n{history}\n{modified_input}'
        response = chatbot(history_and_input)[0]["generated_text"]
        modified_response = response.split('AI:')[-1]
        print(f'AI:\n{modified_response}')

        history += f'User: {user_input}\nAI: {modified_response}'
        print(len(history.split()))

    