from flask import Flask, request, jsonify
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import os
import random


# For publishing image
import json
import requests
import time

# For sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# For caption generation from image
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


app = Flask(__name__)

HUGGINGFACEHUB_API_TOKEN = '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'
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

model_id = "tiiuae/falcon-7b-instruct"
conv_model = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id=model_id,
                            model_kwargs={"temperature": 0.8, "max_new_tokens":200})

template = """You are a helpful caption generator that should only generate Instagram post captions. 
            User will give you a prompt to generate caption for a specific type of post. You should only give caption according to that prompt.
            The caption should be detailed and well defined.

            Instructions for returned data:
            1. Suggestions should be in the following format:
            "Suggestion 1: suggestion text"
            "Suggestion 2: suggestion text"
            "Suggestion 3: suggestion text"
            "Suggestion 4: suggestion text"
            No other format is acceptable. 
            2. Each caption should only be of one to two lines only.
            3. Dont include any tags like <p></p> or // in the result.
            4. Only include text and emojis if applicable. Nothing else.
            
            Please provide 3 or 4 different suggestions for the given prompt:
            {query}
            """

prompt = PromptTemplate(template=template, input_variables=['query'])
conv_chain = LLMChain(llm=conv_model, prompt=prompt, verbose=True)

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    data = request.json
    query = data['query']
    result = conv_chain.run(query)
    print("Debug - Full Output:", result)  

    # Split the result based on the query
    parts = result.split(query)

    # Initialize an empty array to store suggestions
    suggestions = []

    # Take the part after the query and split it into individual suggestions
    if len(parts) > 1:
        suggestions_part = parts[1].strip().replace('\nUser', '')
        if suggestions_part:
            suggestions = [s.strip('"').strip() for s in suggestions_part.split("\n") if s.strip()]

        print("Debug - Generated Suggestions:", suggestions)
        return jsonify(suggestions=suggestions)
    
    else:
        return jsonify(suggestions=null, error="No caption found")

def generate_random_coordinates():
    return random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)

@app.route('/publish-image', methods=['POST'])
def publish_image():
    data = request.json
    ig_user_id = "17841463533012506"
    access_token = 'EAAPoBWhu8NwBO006GWrVYuoln2erJT6NZCDo51HfURtpXZBJvsf7BbUzHFMAsMwCyrZCm7Ws85putjIC2PIwLSY7QRieRGJSFdL0CGUlwHkNSM2gUys7L1optLYcPHAfIZBSOSuZAqI6omfIsJa5Oq8zqm0VFKr1D6gwVFF6pgYVVRGtkS0xZCjtQQseuHCFD1'
    post_url = 'https://graph.facebook.com/v19.0/{}/media'.format(ig_user_id)

    user_tags = []
    for username in data.get('user_tags', []):
        x, y = generate_random_coordinates()
        user_tags.append({'username': username, 'x': x, 'y': y})

    payload = {
        'image_url': data['image_url'],
        'caption': data['caption']+' #bakery #cafe',
        'user_tags':  json.dumps(user_tags),
        'access_token':access_token
    }
    r = requests.post(post_url, data = payload)
    print(r.text)
    print("Media Uploaded Successfully")
    results = json.loads(r.text)

    if 'id' in results:
        creation_id = results['id']
        second_url = 'https://graph.facebook.com/v19.0/{}/media_publish'.format(ig_user_id)
        second_payload = {
            'creation_id':creation_id,
            'access_token':access_token
        }
        r = requests.post(second_url, data = second_payload)
        print(r.text)
        print('Image published to Instagram')
        return jsonify(message="Image published to Instagram")
    else:
        print('Image posting not possible.')
        return jsonify(error="Image posting not possible.")

@app.route('/calculate-sentiment', methods=['POST'])
def calculate_sentiment():
    data = request.json
    analyzer = SentimentIntensityAnalyzer()

    # Analyze sentiment
    sentiment_score = analyzer.polarity_scores(data["caption"])

    # Interpret the sentiment score
    if sentiment_score['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    print("Overall Sentiment:", sentiment)
    return jsonify(sentiment=sentiment)

@app.route('/generate-caption-from-image', methods=['POST'])
def generate_caption_from_image():
    data = request.json
    # image_url = "de86f8c0c93196549706ba9cd60e815d.jpg"
    image_url = "pexels-peter-de-vink-288978-975012.jpg"
    # image_url = "03e4be0ad8c3d09136dfb0de5d0f7440.jpg"
    url = 'https://7b01-34-28-60-193.ngrok-free.app/caption'
    files = {'image': open(image_url, 'rb')}
    response = requests.post(url, files=files)
    # return jsonify(caption=response.caption)
    print(response.json())
    return response.json()

if __name__ == '__main__':
    app.run(debug=True)
