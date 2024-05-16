import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

USER_NAME = "Pam"  # The name you want to use when interviewing the agent.

# Local Llama3 
LLM = ChatOllama(
    model="llama3",
    #keep_alive=-1, # keep the model loaded indefinitely
    temperature=0,
    max_new_tokens=512)

from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

import faiss
import numpy as np

def score_normalizer(val: float) -> float:
    ret = 1.0 - 1.0 / (1.0 + np.exp(val))
    print("val: "+str(float(val))+"_"+"ret: "+str(ret))
    return ret


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OllamaEmbeddings(model="llama3")

    # Automatically determine the size of the embeddings
    test_embedding = embeddings_model.embed_query("test query")
    embedding_size = len(test_embedding)
    
    # Initialize the vectorstore as empty
    index = faiss.IndexFlatL2(embedding_size)
    
    # Initialize FAISS vector store
    vectorstore = FAISS(
        embeddings_model,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=score_normalizer,
        normalize_L2=True
    )
    
    # Create and return the retriever
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, 
        other_score_keys=["importance"], 
        k=15
    )

tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
)

tommie = GenerativeAgent(
    name="Tommie",
    age=25,
    traits="anxious, likes design, talkative",  # You can add more persistent traits here
    status="looking for a job",  # When connected to a virtual world, we can have the characters update their status
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=tommies_memory,
)

# The current "Summary" of a character can't be made because the agent hasn't made
# any observations yet.
print(tommie.get_summary())

# We can add memories directly to the memory object
tommie_observations = [
    "Tommie remembers his dog, Bruno, from when he was a kid",
    "Tommie feels tired from driving so far",
    "Tommie sees the new home",
    "The new neighbors have a cat",
    "The road is noisy at night",
    "Tommie is hungry",
    "Tommie tries to get some rest.",
]
for observation in tommie_observations:
    tommie.memory.add_memory(observation)

# Now that Tommie has 'memories', their self-summary is more descriptive, though still rudimentary.
# We will see how this summary updates after more observations to create a more rich description.
print(tommie.get_summary(force_refresh=True))

def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    print(new_message)
    return agent.generate_dialogue_response(new_message)[1]

interview_agent(tommie, "What are you looking forward to doing today?")