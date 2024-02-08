import time
import openai

import json
import os

import requests



import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import (
        TextAnalyticsClient,
        AnalyzeHealthcareEntitiesAction,
        RecognizePiiEntitiesAction,
    )



def sample_analyze_healthcare_action(endpoint, key, documents) -> None:
    



    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )


    poller = text_analytics_client.begin_analyze_actions(
        documents,
        display_name="Sample Text Analysis",
        actions=[
            AnalyzeHealthcareEntitiesAction(),
            RecognizePiiEntitiesAction(domain_filter="phi"),
        ],
    )

    document_results = poller.result()
    for doc, action_results in zip(documents, document_results):
        print(f"\nDocument text: {doc}")
        for result in action_results:
            if result.kind == "Healthcare":
                print("...Results of Analyze Healthcare Entities Action:")
                for entity in result.entities:
                    print(f"Entity: {entity.text}")
                    print(f"...Normalized Text: {entity.normalized_text}")
                    print(f"...Category: {entity.category}")
                    print(f"...Subcategory: {entity.subcategory}")
                    print(f"...Offset: {entity.offset}")
                    print(f"...Confidence score: {entity.confidence_score}")
                    if entity.data_sources is not None:
                        print("...Data Sources:")
                        for data_source in entity.data_sources:
                            print(f"......Entity ID: {data_source.entity_id}")
                            print(f"......Name: {data_source.name}")
                    if entity.assertion is not None:
                        print("...Assertion:")
                        print(f"......Conditionality: {entity.assertion.conditionality}")
                        print(f"......Certainty: {entity.assertion.certainty}")
                        print(f"......Association: {entity.assertion.association}")
                for relation in result.entity_relations:
                    print(f"Relation of type: {relation.relation_type} has the following roles")
                    for role in relation.roles:
                        print(f"...Role '{role.name}' with entity '{role.entity.text}'")

            elif result.kind == "PiiEntityRecognition":
                print("Results of Recognize PII Entities action:")
                for pii_entity in result.entities:
                    print(f"......Entity: {pii_entity.text}")
                    print(f".........Category: {pii_entity.category}")
                    print(f".........Confidence Score: {pii_entity.confidence_score}")

            elif result.is_error is True:
                print(f"...Is an error with code '{result.error.code}' and message '{result.error.message}'")

            print("------------------------------------------")

    

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
