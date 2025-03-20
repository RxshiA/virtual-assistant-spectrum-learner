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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from pydub import AudioSegment
from pydub.utils import make_chunks
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Define the response structure expected by the frontend
class ResponseModel(BaseModel):
    main_response: str
    follow_up_questions: List[str]
    
class AudioRequest(BaseModel):
    audioUrl: str  # Base64-encoded audio data
    config: dict
    savedFilePath: str

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to save uploaded audio
AUDIO_FILE_PATH = "temp_audio.wav"
AUDIO_SAVE_DIR = "audio_files"  # Directory to save original audio files

# Create the directory if it doesn't exist
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)

# Load API keys
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_api_key = (os.getenv("ELEVENLABS_API_KEY"))
google_speech_to_text_api_key = os.getenv("GOOGLE_SPEECH_TO_TEXT_API_KEY")

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

import os
import base64
import requests
from pathlib import Path
from datetime import datetime

def speech_to_text(audio_url, config):
    """
    Convert speech to text using Google Cloud Speech-to-Text API.

    Args:
        audio_url (str): Base64-encoded audio data.
        config (dict): Configuration for the audio (e.g., encoding, sample rate, language code).

    Returns:
        str: Transcribed text from the audio.
    """
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path(__file__).parent / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Create audio recordings directory if it doesn't exist
        audio_dir = uploads_dir / "audio_recordings"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename with timestamp and .wav extension
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"recording_{timestamp}.wav"
        file_path = audio_dir / file_name

        # Convert base64 string to binary and save to file
        audio_data = base64.b64decode(audio_url)
        with open(file_path, 'wb') as audio_file:
            audio_file.write(audio_data)

        print(f"Audio file saved at: {file_path}")

        # Prepare the request to Google Speech-to-Text API
        api_url = "https://speech.googleapis.com/v1/speech:recognize"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-goog-api-key": os.getenv("GOOGLE_SPEECH_TO_TEXT_API_KEY")
        }
        payload = {
            "audio": {
                "content": audio_url
            },
            "config": config
        }

        # Make the request to Google Speech-to-Text API
        response = requests.post(api_url, json=payload, headers=headers)
        speech_results = response.json()
        print("Speech Results: ", speech_results)

        # Extract the transcribed text
        if "results" in speech_results and len(speech_results["results"]) > 0:
            transcribed_text = speech_results["results"][0]["alternatives"][0]["transcript"]
        else:
            transcribed_text = "No transcription available."

        return transcribed_text

    except Exception as err:
        print(f"Error converting speech to text: {err}")
        return f"Error: {str(err)}"

    except Exception as err:
        print(f"Error converting speech to text: {err}")
        return {"error": str(err)}

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
def formulate_openai_query(text: str, context: list):
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
        f"The child said: \"{text}\".\n\n"
        f"Relevant past interactions:\n{context_str}\n\n"
        f"Assistant responses:\n{response_str}\n\n"
        f"Provide a response that is supportive and suitable for a child with ASD."
    )
    print("Generated OpenAI Query:")
    print(query)
    
    return query

# Function to get response from OpenAI using chat models
def get_response_from_openai(prompt: str):
    # Modify the system instruction to include generating follow-up questions
    system_instruction = (
        "You are a voice assistant for a child with autistic spectrum disorders. "
    "Your purpose is to help the child understand emotions and improve social interaction skills. "
    "When responding, always include one or two simple follow-up questions that the child could ask to continue the conversation. "
    "Make sure the questions are phrased in the first person (e.g., 'Whom should I talk to?') and are easy to understand and relevant to the context. "
    "Format your response as follows:\n\n"
    "Response: <Your main response>\n"
    "Follow-up Questions: <Question 1>|<Question 2>"
    )

    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal:spectrum-learner:AarQpVNF",  # Chat model
        messages=[
            {"role": "system", "content": system_instruction},  # Updated system instructions
            {"role": "user", "content": prompt}  # User input
        ],
        max_tokens=200,
        temperature=0.7
    )

    # Extract the response text
    response_text = response.choices[0].message.content.strip()

    # Split the response into main response and follow-up questions
    if "Follow-up Questions:" in response_text:
        main_response, follow_up_questions = response_text.split("Follow-up Questions:")
        main_response = main_response.replace("Response:", "").strip()
        follow_up_questions = follow_up_questions.strip().split("|")
    else:
        main_response = response_text
        follow_up_questions = []

    return {
        "main_response": main_response,
        "follow_up_questions": follow_up_questions
    }
# def detect_emotion(audio_path):
#     """
#     Detect emotion from an audio file using a pre-trained Hugging Face model.
#     """
#     try:
#         # Use the pipeline to classify the audio file
#         print(f"Analyzing emotions in: {audio_path}")
#         results = pipe(audio_path)
        
#         # Extract the top result (highest confidence emotion)
#         emotion = results[0]['label']
#         confidence = results[0]['score']
        
#         print(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")
#         return emotion
#     except Exception as e:
#         print(f"Error in emotion detection: {e}")
#         return "Unknown"

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
async def process_audio(request: AudioRequest):
    try:
        # Decode base64 audio data (assume raw base64 without prefix)
        audio_data = base64.b64decode(request.audioUrl)

        # Save the audio file
        with open(AUDIO_FILE_PATH, "wb") as f:
            f.write(audio_data)
        print(f"Audio file saved to: {AUDIO_FILE_PATH}")

        # Convert audio to WAV format using pydub (if necessary)
        audio = AudioSegment.from_file(AUDIO_FILE_PATH)
        audio.export(AUDIO_FILE_PATH, format="wav")
        print("Audio file converted to WAV format")

        # Convert speech to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE_PATH) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"Recognized Text: {text}")
        
        # text = speech_to_text(request.audioUrl, request.config)
        # print(f"Speech-to-Text Result: {text}")
            
        # Generate embedding and perform similarity search
        query_embedding = generate_embedding(text)
        similar_interactions = similarity_search(query_embedding)

        # Formulate OpenAI query
        prompt = formulate_openai_query(text,similar_interactions)

        # Get OpenAI response
        openai_response = get_response_from_openai(prompt)

        # Return response
        return ResponseModel(
            main_response=openai_response["main_response"],
            follow_up_questions=openai_response["follow_up_questions"]
        )

    except Exception as e:
        print(f"Error in processing audio: {e}")
        return ResponseModel(
            main_response=f"Error: {str(e)}",
            follow_up_questions=[]
        )

    finally:
        # Clean up temporary audio file
        if os.path.exists(AUDIO_FILE_PATH):
            os.remove(AUDIO_FILE_PATH)
# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
