import logging
logging.basicConfig(level=logging.ERROR)

from datetime import datetime, timedelta
from typing import List

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from termcolor import colored

USER_NAME = "Pam"  # The name you want to use when interviewing the agent.

# Local Llama3 
LLM = ChatOllama(
    model="llama3",
    keep_alive=-1, # keep the model loaded indefinitely
    temperature=0,
    max_new_tokens=512)

from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

import math

import faiss


import math

import faiss


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def relevance_score_fn(score):
    # Convert the raw distance score to a relevance score between 0 and 1
    # Assuming scores are negative distances, higher relevance should be closer to 0
    min_score = -30000  # Set this based on the observed range of your scores
    max_score = 0
    normalized_score = (score - min_score) / (max_score - min_score)
    return min(1, max(0, normalized_score))


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
        embedding_function=embeddings_model.embed_query,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
        relevance_score_fn=relevance_score_fn,
    )
    
    # Create and return the retriever
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, 
        other_score_keys=["importance"], 
        k=15
    )
    
    return retriever

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

interview_agent(tommie, "What do you like to do?")