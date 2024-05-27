import os
import numpy as np
import faiss
import random

from typing import List, Dict, Any, Tuple



from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from termcolor import colored
from GA._init_ import GenerativeAgent, GenerativeAgentMemory

USER_NAME = "User"  # The name you want to use when interviewing the agent.

def score_normalizer(val: float) -> float:
    # This function normalizes the scores to be between 0 and 1
    return 1.0 - 1.0 / (1.0 + np.exp(val))

def create_new_memory_retriever(model: str):
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OllamaEmbeddings(model=model)

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
        normalize_L2=True # Normalize the embeddings
    )
    
    # Create and return the retriever
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, 
        other_score_keys=["importance"], 
        k=15
    )

class OneAgent:
    def __init__(self, info: Dict, model: str, temperature: float, max_new_tokens: int):
        self.info = info
        self.llm = ChatOllama(
            model=model,
            keep_alive=-1,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        self.memory = GenerativeAgentMemory(
            llm=self.llm,
            memory_retriever=create_new_memory_retriever(model),
            verbose=False,
            reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
        )
        self.agent = GenerativeAgent(
            name=self.info["name"],
            age=self.info["age"],
            traits=self.info["traits"],  # You can add more persistent traits here
            status=self.info["status"],  # When connected to a virtual world, we can have the characters update their status
            llm=self.llm,
            memory=self.memory,
        )

    def init_memory(self):
        for i, observation in enumerate(self.info["initial_memory"]):
            _, reaction = self.agent.generate_reaction(observation)
            print(colored(observation, "green"), reaction)
        print("*" * 40)
        print(
            colored(
                f"After {i+1} inital observations, summary is:\n{self.agent.get_summary(force_refresh=True)}",
                "blue",
            )
        )
        print("*" * 40)

    # reserved method for external user to interview
    def interview_agent(self,message: str) -> str:
        """Help the notebook user interact with the agent."""
        new_message = f"{USER_NAME} says {message}"
        ans =  self.agent.generate_dialogue_response(new_message)[1]
        print(colored(f"{self.agent.name} says: {ans}", "green"))
        return ans

    # response to the environment we provided
    def env_response(self,env_observations: str) -> str:
        for i, observation in enumerate(env_observations):
            _, reaction = self.agent.generate_reaction(observation)
            print(colored(observation, "green"), reaction)
        print("*" * 40)
        print(
            colored(
                f"After {i+1} environmental observations, summary is:\n{self.agent.get_summary(force_refresh=True)}",
                "blue",
            )
        )
        print("*" * 40)

    def memory_based_propose(self, message:str):
        return self.agent.propose()
    
    # based on the proposals, agent come up with a voting decision
    def vote(self, proposals:List[str]) -> int:
        ans = self.agent.vote_decision(proposals)
        print(ans)
        return ans
