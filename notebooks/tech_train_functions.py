import time
import openai

import json
import os

import requests


class TA4HPredictor:
    def __init__(self):
        self.endpoint = os.getenv("TA4H_ENDPOINT")

    def predict_ta4h(self, texts):
        """
        call TA4H API to get entities and relations
        This assumes a ta4h synchronous endpoint. This currently only exists on the onprem container,
        at {base_url}/text/analytics/v3.1/entities/health.

        :param text:
        :return: list of entities and relations
       """
        payload = json.dumps({
            "documents": [{"id": idx, "text": text} for idx, text in enumerate(texts)],
            "output_format": "ayalon"
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", self.endpoint, headers=headers, data=payload)
        response = json.loads(response.text)
        entities = [r["entities"] for r in response['documents']]
        return entities
    

def call_open_ai(system_message, text, engine="gpt-35-turbo", examples=[], client=None, temprature=0):
    message_text = [{"role": "system", "content": system_message}]
    for example in examples:
        message_text.append({"role": "user", "content": example[0]})
        message_text.append({"role": "assistant", "content": example[1]})
    message_text.append({"role": "user", "content": text})
    try:
        response = call_client_or_completion(system_message, text, message_text, engine, client, temprature)
    except openai.error.RateLimitError as e:
        # wait 10 seconds and try again
        print("Rate limit exceeded. Waiting 10 seconds and trying again...")
        time.sleep(10)
        response = call_client_or_completion(system_message, text, message_text, engine, client, temprature)
    return response


def call_client_or_completion(system_message, text, message_text, engine="gpt-35-turbo", client=None, temprature=0):
    """
    Call the openAI client or the openAI completion API
    :param system_message: The system message
    :param text: The user message
    :param message_text: The entire conversation
    :param engine: The openAI engine
    :param client: The openAI client
    :param temprature: The openAI temprature
    :return: The openAI response
    """
    
    if client==None: #use azure openAI
        completion = openai.ChatCompletion.create(
            engine=engine,
            messages=message_text,
            temperature=temprature,
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
    else: #use openAI studio
        if engine=="gpt-35-turbo":
            model="gpt-3.5-turbo"
        elif engine=="gpt4":
            model="gpt-4"
        else:
             model=engine
        completion = client.chat.completions.create(
        model=model,
        messages=message_text,
        temperature=temprature,
        max_tokens=800,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        
        return completion.choices[0].message.content

def is_similar(gold_answer, gold_answer_idx, answer):
        if len(answer) == 1:
            return 1 if gold_answer_idx  == answer else 0
        gold_answer = gold_answer.lower()
        answer = answer.lower()
        return 1 if gold_answer in  answer or answer in gold_answer else 0

def create_example_CoT(system_message_explainer, question, options, answer, client=None):   
    question = "Question: "+question+" "+ str(options)+"."
    explained_answer = call_open_ai(system_message_explainer, question+" Correct answer: "+ answer, client=client)    
    return question, explained_answer
