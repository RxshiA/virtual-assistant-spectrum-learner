import speech_recognition as sr
import librosa
import numpy as np
import os
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from elevenlabs import generate, stream
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydub import AudioSegment
import uvicorn
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Path to save uploaded audio
AUDIO_FILE_PATH = "temp_audio.wav"

# API Response Model
class ResponseModel(BaseModel):
    text: str
    emotion: str
    openai_response: str

# Load the Hugging Face pipeline for speech emotion recognition
pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# Load API keys
pc = Pinecone(api_key="pcsk_fPUZL_RZ2eZPsVKzXzMyGmr9zoAMnTXyDLrAuxaypEHDwcvxQ18hxYzoaGRtSupepvuwJ")
client = OpenAI(api_key="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")
elevenlabs_api_key = "sk_b56ab61e77b0fe28eeab140f6a81543957e01dc71362d3ed"

# Define Pinecone index name
INDEX_NAME = "asd-therapy-interactions"

index = pc.Index(INDEX_NAME)

# Path to save audio for analysis
AUDIO_FILE_PATH = "temp_audio.wav"

def extract_features(audio_path, sr=22050):
    """
    Extract audio features using librosa.
    Returns a feature vector with MFCCs and other audio characteristics.
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

    # Combine features into a single array
    features = np.hstack([mfccs, chroma, mel])
    return features

# Function to generate OpenAI embeddings
def generate_embedding(text: str) -> list:
    return client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding

def similarity_search(query_embedding: list, top_k: int = 5):
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]

# Function to combine all data for OpenAI query
def formulate_openai_query(text: str, emotion: str, context: list):
    # Format context from Pinecone results
    context_str = "\n".join([
        f"Child: {match['metadata']['text']}"  # Assuming metadata contains the question text
        for match in context if match['metadata']['role'] == 'child'
    ])
    
    response_str = "\n".join([
        f"Assistant: {match['metadata']['text']}"  # Assuming metadata contains the response text
        for match in context if match['metadata']['role'] == 'assistant'
    ])
    
    # Formulate query for OpenAI
    query = (
        f"The child said: \"{text}\" with emotion detected as {emotion}.\n\n"
        f"Relevant past interactions:\n{context_str}\n\n"
        f"Assistant responses:\n{response_str}\n\n"
        f"Provide a response that is supportive and suitable for a child with ASD."
    )
    print("Generated OpenAI Query:")
    print(query)
    
    return query

# Function to get response from OpenAI using chat models
def get_response_from_openai(prompt: str):
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal:spectrum-learner:AarQpVNF",  # Chat model
        messages=[
            {"role": "system", "content": "You are a voice assistant of a kid with autistic spectrum disorders and your purpose is to help the kid to understand emotions and improve social interaction skills."},  # System instructions
            {"role": "user", "content": prompt}  # User input
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def detect_emotion(audio_path):
    """
    Detect emotion from an audio file using a pre-trained Hugging Face model.
    """
    try:
        # Use the pipeline to classify the audio file
        print(f"Analyzing emotions in: {audio_path}")
        results = pipe(audio_path)
        
        # Extract the top result (highest confidence emotion)
        emotion = results[0]['label']
        confidence = results[0]['score']
        
        print(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")
        return emotion
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "Unknown"

def generate_audio(text: str):
        """Generate audio response using ElevenLabs."""
        audio_stream = generate(
            api_key=elevenlabs_api_key,
            text=text,
            voice="Brian",
            stream=True
        )
        stream(audio_stream)

# Endpoint to process audio file
@app.post("/process-audio", response_model=ResponseModel)
async def process_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Save uploaded audio file
        with open(AUDIO_FILE_PATH, "wb") as f:
            f.write(await file.read())
        
        # Detect emotion
        emotion = detect_emotion(AUDIO_FILE_PATH)

        # Convert speech to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE_PATH) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Generate embedding and perform similarity search
        query_embedding = generate_embedding(text)
        similar_interactions = similarity_search(query_embedding)

        # Formulate OpenAI query
        prompt = formulate_openai_query(text, emotion, similar_interactions)

        # Get OpenAI response
        openai_response = get_response_from_openai(prompt)
        
        # Stream response as audio using ElevenLabs
        background_tasks.add_task(generate_audio, openai_response)

        # Return response
        return ResponseModel(
            text=text,
            emotion=emotion,
            openai_response=openai_response
        )

    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Clean up temporary audio file
        if os.path.exists(AUDIO_FILE_PATH):
            os.remove(AUDIO_FILE_PATH)

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
