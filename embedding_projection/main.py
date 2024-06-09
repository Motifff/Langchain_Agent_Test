import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from langchain_community.embeddings import OllamaEmbeddings
  # Placeholder for your embedding function

# Step 1: Load your text file
with open('embedding_projection/similarity.txt', 'r') as file:
    paragraphs = file.read().split('\n\n')  # Adjust split as needed

# Step 2: Obtain embeddings for each paragraph
def get_embedding_vector(model: str, text: str):
    """Get the embedding vector for a given text."""
    embeddings_model = OllamaEmbeddings(model=model)
    # get embedding vector
    o_vec = embeddings_model.embed_query(text)
    return o_vec

embeddings = np.array([get_embedding_vector("qwen2", paragraph) for paragraph in paragraphs])

# Step 3: Apply PCA
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(embeddings)

# Step 4: Create an interactive 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])

# Function to update annotation on hover
def update_annot(ind):
    pos = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = pos
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
annot.set_visible(False)
fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
