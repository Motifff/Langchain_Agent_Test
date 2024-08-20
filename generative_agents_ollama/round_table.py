"""
File Hierachy of vis_data.json
{
    "original_topic"
    "agents": [
        {
            "name"
            "age"
            "traits"
            "status"
            "initial_memory"
        }
    ,...]
    "total_round":
    "runtime":[
        "round_count"
        "topic":{
            "text"
            "vector"
            "pca_vector"
        }
        "proposals":[
            {
                "proposal"
                "vector"
                "pca_vector"
            },...
        ]
        "vote":{
            "process": [],
            "win": 
        }
    ]
}
"""

import os, json ,fcntl, time
import socket
import threading

from termcolor import colored
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

from utils import OneAgent
from GA._init_ import GenerativeAgent

import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# This is a workaround for a known issue with the transformers library.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

file_path = "/Users/motif/Documents/Projs/code/Langchain_Agent_Test/generative_agents_ollama/data.json"

ifRun = False

class DesignerRoundTableChat:
    def __init__(self, agents: List[OneAgent], topic: str, jsonFile: Dict, directinal_text, emb_model: str = "phi3"):
        self.agents = agents
        self.topic = topic
        self.init_topic = topic
        self.round_count = 0
        self.data_round = 0
        self.proposal_won = ""
        self.design_proposals = []
        self.votes = []
        self.extreme_words = directinal_text
        self.scaler = StandardScaler()  # This should be fit on your extreme vectors if possible
        self.llm = ChatOllama(
            model="phi3",
            keep_alive=-1,
            temperature=0.2,
            max_new_tokens=4096
        )
        self.embeddings_model = OllamaEmbeddings(model=emb_model)

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        """Create a chain with the same settings as the agent."""

        return LLMChain(
            llm=self.llm, prompt=prompt
        )

    def generate_design_proposals(self):
        for each in self.agents:
            # Simulating design proposal by each agent
            proposal = each.agent.propose(self.topic)
            print("\n"+each.agent.name+"\n\n"+colored(str(proposal), "green"))
            self.design_proposals.append(proposal)

    def interview_agent(agent: GenerativeAgent, message: str):
        """user could use interview to understand current states of an agent"""
        new_message = input("You are interviewing" + "Enter your message: ")
        if new_message:
            print(colored(agent.generate_dialogue_response(new_message)[1], "blue"))
        else:
            print(colored("SKIP", "red"))      

    def conduct_voting(self):
        # Each agent has two votes to distribute among proposals
        for agent in self.agents:
            votes = agent.vote(self.design_proposals)
            self.votes.append(votes)       

    def count_votes(self):
        # find out the most appeared number in the self.votes, if there is two, then return 999
        # Count the occurrences of each number
        vote_counts = Counter(self.votes)
        # Find the highest occurrence count
        max_count = max(vote_counts.values())
        # Find all numbers that have the highest occurrence count
        most_common_numbers = [num for num, count in vote_counts.items() if count == max_count]
        # If there is a tie, return 999
        if len(most_common_numbers) > 1:
            return 999
        else:
            return most_common_numbers[0]

    def generate_new_round_topic(self) -> str:
        """from current topic and every agent's proposal, generate a new topic for next round"""
        prompt = PromptTemplate.from_template(
            "You are a round table holder and overall topic is {overall_topic}"+
            "You have just finished a round of discussion"+
            "Based on the proposals from the agents, they are {current_round_proposals},"+
            "And the winning proposal is {winning_proposal}"+
            "you need to generate a new topic for the next round of discussion based on given information"+
            "The new topic should be related to the current topic but with a new focus."+
            "The response should be within 3-5 sentences."
        )
        # convert proposals to string
        proposals = "\n".join(self.design_proposals)
        kwargs: Dict[str, Any] = dict(
            current_round_proposals=proposals,
            winning_proposal=self.proposal_won,
            current_topic=self.topic,
            overall_topic=self.init_topic,
        )
        # Send the prompt to ChatOllama and get the response
        response = self.chain(prompt).run(**kwargs).strip()

        print(colored(f"New topic for the next round is: {response}", "red"))
        self.topic = response 
        return response

    def generate_extreme_vectors(self):
        # ask the llm to generate six extreme vectors for self.topic and return in a json list format
        prompt = PromptTemplate.from_template(
            "You are a round table holder and overall topic is {overall_topic}"+
            "You have just finished a round of discussion"+
            "Based on the proposals from the agents, they are {current_round_proposals},"+
            "And the winning proposal is {winning_proposal}"+
            "you need to generate a new topic for the next round of discussion based on given information"+
            "The new topic should be related to the current topic but with a new focus."+
            "The response should be within 3-5 sentences."
        )
        kwargs: Dict[str, Any] = dict(
            current_topic=self.topic
        )
        # Send the prompt to ChatOllama and get the response
        response = self.chain(prompt).run(**kwargs).strip()
        # parse the json text to get the extreme vectors's list
        self.extreme_vectors = json.loads(response)

    def get_embedding_vector(self, text: str):
        """Get the embedding vector for a given text and also return its PCA-reduced version."""
        # Get the embedding vector for the text and self.extreme_text
        o_vec = self.embeddings_model.embed_query(text)
        extreme_vectors = [self.embeddings_model.embed_query(extreme_text) for extreme_text in self.extreme_words]
        
        # Combine the original vector with the extreme vectors
        all_vectors = np.vstack([extreme_vectors, o_vec])
        
        # Scale the combined vectors
        scaled_vectors = self.scaler.fit_transform(all_vectors)
        
        # Perform PCA to reduce the dimensionality to 3
        pca = PCA(n_components=3)
        reduced_vectors = pca.fit_transform(scaled_vectors)
        
        # Extract the PCA-reduced vector for the input text
        p_vec = reduced_vectors[-1]
        
        return [o_vec, p_vec.tolist()]
    
    # First we should init the initial memory for each agent
    def init_memory(self):
        for each in self.agents:
            each.init_memory()

    def run_round(self,now: Optional[datetime] = None):        
        # Generate design proposals, the round should be like this:
        # 1. Generate design proposals based on given topic(only in first round)
        # 2. Each agent generates a design proposal and presents it to the group
        # 3. Interview each agent to understand their design proposals(Optional for user)
        # 4. Count voting results and decide the winning proposal, if there is a tie, redo 2 and 3
        # 5. Announce the winning proposal and add them to each agent's memory
        # Step should -1 if successfully finish a round, if steps is not 0, continue this process
        global data
        self.data_round = data["total_round"]
        while 1:          
            if self.round_count < self.data_round and ifRun:
                self.generate_design_proposals()
                for each in self.agents:
                    print(colored(f"skip interview process", "blue"))
                self.conduct_voting()
                result = self.count_votes()
                if result == 999:
                    print(colored("There is a tie, redo the round", "red"))
                    continue
                else:
                    self.proposal_won = self.design_proposals[result-1]
                    print(colored(f"The winning proposal is: {self.proposal_won}", "green"))
                    for each in self.agents:
                        for index, one in enumerate(self.design_proposals):
                            each.agent.memory.save_context(
                                {},
                                { 
                                    each.agent.memory.add_memory_key: f"{self.agents[index].agent.name} proposed {one}",
                                    each.agent.memory.now_key: now,
                                }
                            )
                        each.agent.memory.save_context(
                            {},
                            {
                                each.agent.memory.add_memory_key: f"The winning proposal is {self.design_proposals[result-1]}",
                                each.agent.memory.now_key: now,
                            },
                        )
                        #print(colored(str(each.memory.memory_retriever.memory_stream), "red"))
                    # dump data to json file
                    data["runtime"].append({
                        "round_count": self.round_count,
                        "topic": {
                            "text": self.topic,
                            "vector": self.get_embedding_vector(self.topic)[0],
                            "pca_vector": self.get_embedding_vector(self.topic)[1],
                        },
                        "proposals": [
                            {
                                "proposal": each,
                                "vector": self.get_embedding_vector(each)[0],
                                "pca_vector": self.get_embedding_vector(each)[1],
                            } for each in self.design_proposals
                        ],
                        "vote": {
                            "process": self.votes,
                            "win": self.proposal_won
                        }
                    })
                    with open(file_path,"w") as file:
                        #fcntl.flock(file, fcntl.LOCK_EX)
                        json.dump(data, file, indent=4)
                        #fcntl.flock(file, fcntl.LOCK_UN)
                        file.close()
                    self.round_count += 1
                    print(colored(f"Round is finished", "green"))
                    self.generate_new_round_topic()
                    #clean this round's data
                    self.design_proposals = []
                    self.votes = []
                    self.proposal_won = ""
            else:
                # system delay for 30s
                time.sleep(30)
                print(colored("###System is waiting for signal to the next round###", "blue"))
                # update data from json file
                with open(file_path,"r") as file:
                    #fcntl.flock(file, fcntl.LOCK_SH)
                    data = json.load(file)
                    #fcntl.flock(file, fcntl.LOCK_UN)
                file.close()
                self.steps = data["total_round"]
                print(colored(f"Total round is: {self.steps}", "green"))

def listen_udp():
    global ifRun
    udp_ip = "localhost"
    udp_port = 6666

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))

    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        message = data.decode('utf-8')
        if message == "start":
            ifRun = True
            print("ifRun set to True")

listener_thread = threading.Thread(target=listen_udp)
listener_thread.daemon = True  # This allows the thread to exit when the main program exits
listener_thread.start()

# initial main function
if __name__ == "__main__":
    # read from json file
    global data
    with open(file_path,"r") as file:
        #fcntl.flock(file, fcntl.LOCK_SH)
        data = json.load(file)
        #fcntl.flock(file, fcntl.LOCK_UN)
        file.close()
    # read topic from json file in [original_topic]
    topic = data["original_topic"]
    agents = []

    # agents are generated from the data[agents] file, which have name, age, traits, status, initial_memory, but model data is not included
    for each in data["agents"]:
        agent = OneAgent(each,"phi3",0.1,512)
        agents.append(agent)

    agents_general_memory = [
        
    ]

    directional_text = [
        "Focus on objective facts and data, analyze existing information.",
        "Represents emotions and intuition, expresses personal feelings and emotions.",
        "Used for critical thinking, identify potential problems and risks.",
        "Symbolizes optimism, looking for positive aspects and opportunities in problems.",
        "Represents creative thinking, encourages new ideas and solutions.",
        "Responsible for organizing and controlling the thinking process, ensuring that thinking is carried out in an orderly manner.",
    ]

    round_table_chat = DesignerRoundTableChat(agents, topic, data, directional_text)
    #round_table_chat.init_memory()
    round_table_chat.run_round()
        