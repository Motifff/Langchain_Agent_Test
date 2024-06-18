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

# Assign embeddings back to the respective places in the data
embedding_index = 0
for round_data in data['runtime']:
    if round_data['topic']['vector'] is None:
        round_data['topic']['vector'] = embeddings[embedding_index]
        embedding_index += 1
    for proposal in round_data['proposals']:
        if proposal['vector'] is None:
            proposal['vector'] = embeddings[embedding_index]
            embedding_index += 1

# Save the updated JSON data back to the file
with open('embedding_projection/vis_data_1.json', 'w') as file:
    json.dump(data, file, indent=4)

# Prepare data for PCA visualization
all_vectors = []
labels = []
colors = []
paragraphs = []

round_colors = ['#000000', '#0000FF', '#00FF00', '#00FFFF', '#FF0000']
alpha = 0.5

for i, round_data in enumerate(data['runtime']):
    topic_vector = round_data['topic']['vector']
    all_vectors.append(topic_vector)
    labels.append(f"Round {round_data['round_count']} Topic")
    colors.append(round_colors[i])
    paragraphs.append(round_data['topic']['text'])

    for proposal in round_data['proposals']:
        proposal_vector = proposal['vector']
        all_vectors.append(proposal_vector)
        labels.append(f"Round {round_data['round_count']} Proposal")
        colors.append(round_colors[i])
        paragraphs.append(proposal['proposal'])

# Convert all_vectors to numpy array for PCA
all_vectors_np = np.array(all_vectors)

# Standardize the vectors
scaler = StandardScaler()
all_vectors_scaled = scaler.fit_transform(all_vectors_np)

# Apply PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(all_vectors_scaled)

# Plot using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot
scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=colors, alpha=0.5)

# Function to update annotation on hover
def update_annot(ind):
    pos = scatter._offsets3d
    x, y, z = pos[0][ind["ind"][0]], pos[1][ind["ind"][0]], pos[2][ind["ind"][0]]
    annot.xy = (x, y)
    annot.set_position((x, y, z))
    text = "\n".join([paragraphs[n] for n in ind["ind"]])
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

# Add annotation for hover effect
annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))


# Connect the topics and proposals
for i, round_data in enumerate(data['runtime']):
    for idx, proposal in enumerate(round_data['proposals']):
        ax.plot([principal_components[i*4, 0], principal_components[i*4+idx+1, 0]],
                [principal_components[i*4, 1], principal_components[i*4+idx+1, 1]],
                [principal_components[i*4, 2], principal_components[i*4+idx+1, 2]],
                color=round_colors[i],linestyle='--', alpha=0.2)

# Connect topics from consecutive rounds
for i in range(len(data['runtime']) - 1):
    current_topic_index = labels.index(f"Round {data['runtime'][i]['round_count']} Topic")
    next_topic_index = labels.index(f"Round {data['runtime'][i+1]['round_count']} Topic")
    print("current_topic:"+str(current_topic_index)+"&next_topic:"+str(next_topic_index))
    ax.plot([principal_components[current_topic_index, 0], principal_components[next_topic_index, 0]],
            [principal_components[current_topic_index, 1], principal_components[next_topic_index, 1]],
            [principal_components[current_topic_index, 2], principal_components[next_topic_index, 2]],
            color=round_colors[i])

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('3D PCA Visualization of Topics and Proposals')
annot.set_visible(False)
fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()