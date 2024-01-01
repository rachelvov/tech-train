import openai

def call_open_ai(system_message, text, engine="gpt-35-turbo", examples=[]):
    message_text = [{"role": "system", "content": system_message}]
    for example in examples:
        message_text.append({"role": "user", "content": example[0]})
        message_text.append({"role": "assistant", "content": example[1]})
    message_text.append({"role": "user", "content": text})
    completion = openai.ChatCompletion.create(
        engine=engine,
        messages=message_text,
        temperature=0,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    if "content" in completion["choices"][0]["message"]:
        return completion["choices"][0]["message"]["content"]
    else:
        print(completion)
        return "none"
