from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import uuid
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_fPUZL_RZ2eZPsVKzXzMyGmr9zoAMnTXyDLrAuxaypEHDwcvxQ18hxYzoaGRtSupepvuwJ")
client = OpenAI(api_key="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")

# Define your Pinecone index name
INDEX_NAME = "asd-therapy-interactions"

# Create or connect to Pinecone index
if INDEX_NAME not in pc.list_indexes():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine", spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ))
index = pc.Index(INDEX_NAME)

# Function to load interactions from JSON
def load_interactions(file_path: str) -> list:
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []


# Function to save interactions to JSON
def save_interactions(file_path: str, data: list):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


# Function to add a new interaction to JSON
def add_interaction(child_statement: str, therapist_response: str):
    interactions = load_interactions(INTERACTIONS_FILE)
    interaction_id = str(uuid.uuid4())  # Unique session ID
    new_interaction = {
        "id": interaction_id,
        "child_statement": child_statement,
        "therapist_response": therapist_response
    }
    interactions.append(new_interaction)
    save_interactions(INTERACTIONS_FILE, interactions)
    print(f"New interaction added with ID: {interaction_id}")
    return interaction_id


# Function to generate OpenAI embeddings
def generate_embedding(text: str) -> list:
    return client.embeddings.create(input = [text], model="text-embedding-ada-002").data[0].embedding


# Function to insert interaction data into Pinecone
def insert_interaction_to_pinecone(json_file_path: str):
# Load the JSON file
    with open(json_file_path, "r") as file:
        interactions = json.load(file)

    vectors = []
    # Iterate through each interaction in the JSON
    for idx, interaction in enumerate(interactions):
        question = interaction["question"]
        response = interaction["response"]

        # Generate embeddings
        question_embedding = generate_embedding(question)
        response_embedding = generate_embedding(response)

         # Create vector for the question
        vectors.append({
            "id": f"question_{idx}",
            "values": question_embedding,
            "metadata": {
                "role": "child",
                "text": interaction["question"],
                "type": "question"
            }
        })

        # Create vector for the response
        vectors.append({
            "id": f"response_{idx}",
            "values": response_embedding,
            "metadata": {
                "role": "assistant",
                "text": interaction["response"],
                "type": "response"
            }
        })

    # Upsert all vectors into Pinecone in batches
    index.upsert(vectors=vectors)
    print(f"Inserted {len(interactions)} interactions into Pinecone")


# Example Usage
if __name__ == "__main__":

    json_file_path = "./data/interactions-data.json"  # Replace with your JSON file path
    insert_interaction_to_pinecone(json_file_path)
