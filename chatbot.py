from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import requests
import pyttsx3

engine = pyttsx3.init()  # initialize the TTS engine


weather_api = os.environ.get('weather_api')

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


gather_info = ['weather', 'orai']

history = ''
# Creating a function to turn on the AI for you to interact with
guidance = "You are a sarcastic, dark-humor, unfriendly assistant."
while True:
    handled_weather = False

    user_input = input("Your prompt: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    modified_input = f'{guidance}\n{user_input}\nAI:'


    for words in gather_info:
        if words in user_input:
            print('Which city?')
            engine.say('which city?')
            engine.runAndWait()

            city_input = input('Enter city: ')
            call = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city_input}&appid={weather_api}&units=metric")
            data = call.json()

            if call.status_code == 200:
                temp = data["main"]["temp"]
                weather_desc = data["weather"][0]["description"]
                print(f"Current temperature in {city_input}: {temp}°C, {weather_desc}")
                engine.say(f"Current temperature in {city_input}: {temp}°C, {weather_desc}")
                engine.runAndWait()
            else:
                print("Error:", data)
            handled_weather = True
            break
 

    if handled_weather:
        continue

    if len(history.split()) > 400:
        history = ''

        response = chatbot(modified_input)[0]["generated_text"]
        modified_response = response.split('AI:')[-1]
        print(f'AI:\n{modified_response}')
        engine.say(modified_response)
        engine.runAndWait()

        history += f'User: {user_input}\nAI:{modified_response}'
        print(len(history.split()))
    else:
        history_and_input = f'{guidance}\n{history}\n{modified_input}'
        response = chatbot(history_and_input)[0]["generated_text"]
        modified_response = response.split('AI:')[-1]
        print(f'AI:\n{modified_response}')
        engine.say(modified_response)
        engine.runAndWait()

        history += f'User: {user_input}\nAI: {modified_response}'
    

    