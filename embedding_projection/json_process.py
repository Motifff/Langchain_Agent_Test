import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from langchain_community.embeddings import OllamaEmbeddings

# Load the JSON data
with open('embedding_projection/vis_data.json', 'r') as file:
    data = json.loads(file.read())

# Function to generate embeddings using OllamaEmbeddings
def get_embedding_vector(model: str, text: str):
    """Get the embedding vector for a given text."""
    embeddings_model = OllamaEmbeddings(model=model)
    # get embedding vector
    o_vec = embeddings_model.embed_query(text)
    return o_vec

# Function to generate embeddings for a list of texts
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        embedding = get_embedding_vector(model="qwen2", text=text)
        embeddings.append(embedding)
    return embeddings

# Collect all text data that needs embeddings
texts_to_embed = []

for round_data in data['runtime']:
    if round_data['topic']['vector'] is None:
        texts_to_embed.append(round_data['topic']['text'])
    for proposal in round_data['proposals']:
        if proposal['vector'] is None:
            texts_to_embed.append(proposal['proposal'])

# Generate embeddings
embeddings = generate_embeddings(texts_to_embed)

# Convert embeddings to numpy array for PCA
all_vectors_np = np.array(embeddings)

# Standardize the vectors
scaler = StandardScaler()
all_vectors_scaled = scaler.fit_transform(all_vectors_np)

# Apply PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(all_vectors_scaled)

print(principal_components)

# Assign embeddings back to the respective places in the data
embedding_index = 0
for round_data in data['runtime']:
    if round_data['topic']['vector'] is None:
        round_data['topic']['vector'] = embeddings[embedding_index]
        round_data['topic']['pca_vector'] = principal_components[embedding_index].tolist()
        embedding_index += 1
    for proposal in round_data['proposals']:
        if proposal['vector'] is None:
            proposal['vector'] = embeddings[embedding_index]
            proposal['pca_vector'] = principal_components[embedding_index].tolist()
            embedding_index += 1

# Save the updated JSON data back to the file
with open('embedding_projection/vis_data_2.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Embeddings generated and saved to vis_data_2.json")