import os, random

from termcolor import colored
from typing import List, Dict, Any, Tuple
from collections import Counter

from utils import OneAgent
from GA._init_ import GenerativeAgent, GenerativeAgentMemory

# This is a workaround for a known issue with the transformers library.
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DesignerRoundTableChat:
    def __init__(self, agents: List[OneAgent], topic: str):
        self.agents = agents
        self.topic = topic
        self.design_proposals = []
        self.votes = []

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


    def run_round(self,steps):
        # Firstly, we should initialize the memory of each agent
        
        # Generate design proposals, the round should be like this:
        # 1. Generate design proposals based on given topic(only in first round)
        # 2. Each agent generates a design proposal and presents it to the group
        # 3. Interview each agent to understand their design proposals(Optional for user)
        # 4. Count voting results and decide the winning proposal, if there is a tie, redo 2 and 3
        # 5. Announce the winning proposal and add them to each agent's memory
        # Step should -1 if successfully finish a round, if steps is not 0, continue this process
        while steps != 0:
            self.generate_design_proposals()
            for each in self.agents:
                print(colored(f"skip interview process", "blue"))
            self.conduct_voting()
            result = self.count_votes()
            if result == 999:
                print(colored("There is a tie, redo the round", "red"))
                continue
            else:
                print(colored(f"The winning proposal is: {self.design_proposals[result]}", "green"))
                for each in self.agents:
                    each.memory.add_memory(self.design_proposals[result])
                steps -= 1
                #clean this round's data
                self.design_proposals = []
                self.votes = []
        

# Example
topic = "Design a new logo for our company"

# Define the agents with self, info: Dict, model: str, temperature: float, max_new_tokens: int

agents = [
    OneAgent({"name": "Tommie", "age": 25, "traits": "creative and friendly", "status": "working on a robotic project","initial_memory": ""}, "phi3", 0.2, 4096),
    OneAgent({"name": "Sally", "age": 30, "traits": "logical and detail-oriented", "status": "dive into books about bio-design","initial_memory": ""}, "phi3", 0.2, 4096),
    OneAgent({"name": "Bob", "age": 35, "traits": "analytical and introverted", "status": "doing excercise","initial_memory": ""}, "phi3", 0.2, 4096),
]

god_agent = OneAgent({"name": "God", "age": 9999, "traits": "all-knowing", "status": "omnipotent","initial_memory": ""}, "phi3", 0.2, 4096)

agents_general_memory = [
    
]

round_table_chat = DesignerRoundTableChat(agents, topic)
round_table_chat.run_round(5)
