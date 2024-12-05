
import matplotlib.pyplot as plt


query_text = "I am feeling really excited today!"

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

# Mock Pinecone similarity results
similarity_results = [
    {"id": "interaction_1", "score": 0.85, "metadata": {"interaction": "I feel happy about my day"}},
    {"id": "interaction_2", "score": 0.78, "metadata": {"interaction": "I feel so joyful"}},
    {"id": "interaction_3", "score": 0.72, "metadata": {"interaction": "I am thrilled with what happened!"}},
]

# Extract data for the chart
interactions = [result["metadata"]["interaction"] for result in similarity_results]
scores = [result["score"] for result in similarity_results]


# Data from the similarity search
interactions = ["I feel happy about my day", "I feel so joyful", "I am thrilled with what happened!"]
scores = [0.85, 0.78, 0.72]

# Plot a horizontal bar chart
plt.figure(figsize=(8, 5))
plt.barh(interactions, scores, color='lightgreen')
plt.title("Pinecone Similarity Search Results")
plt.xlabel("Similarity Score")
plt.ylabel("Interactions")
plt.xlim(0, 1)  # Scores range from 0 to 1

# Add score annotations
for i, v in enumerate(scores):
    plt.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=10)

# Show the chart
plt.tight_layout()
plt.show()