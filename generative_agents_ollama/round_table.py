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
topic = "Design and development of future cities"

# Define the agents with self, info: Dict, model: str, temperature: float, max_new_tokens: int

agents = [
    OneAgent({"name": "Alex", "age": 25, "traits":  "innovative, analytical and urban planner", "status": "working on a robotic project","initial_memory": 
            "Alex has worked on several smart city projects across different continents."
            "Alex believes in the potential of technology to solve urban challenges."
            "Alex had a mentor who emphasized the importance of community involvement in urban planning."
            "Alex recently attended a conference on sustainable city development."
            "Alex enjoys reading about the latest advancements in renewable energy."
            "Alex notes the increasing integration of IoT devices in urban infrastructure."
            "Alex observes a trend towards mixed-use developments in major cities."
            "Alex finds that public opinion is often divided on the implementation of autonomous vehicles."
            "Alex sees the growing importance of green spaces in urban areas for residents' well-being."
            "Alex is concerned about the digital divide and its impact on equitable access to smart city benefits."},
            "phi3", 0.2, 4096),
    OneAgent({"name": "Jordan", "age": 30, "traits": "curious" "critical" "environmental scientist", "status": "dive into books about bio-design","initial_memory": 
            "Jordan has been researching the impact of urbanization on local ecosystems."
            "Jordan is passionate about reducing carbon footprints in city planning."
            "Jordan worked on a project that successfully integrated green roofs in a metropolitan area."
            "Jordan often collaborates with urban planners and architects to promote sustainable practices."
            "Jordan recently published a paper on the benefits of urban biodiversity."
            "Jordan notices the rise of eco-friendly building materials in construction."
            "Jordan is intrigued by the potential of vertical farming in urban settings."
            "Jordan is concerned about the pollution levels in rapidly growing cities."
            "Jordan observes that public transportation systems are key to reducing urban emissions."
            "Jordan finds that cities with robust recycling programs have lower waste management costs."}, 
            "phi3", 0.2, 4096),
    OneAgent({"name": "Bob", "age": 35, "traits": "empathetic, community-oriented and sociologist", "status": "doing design project","initial_memory": 
            "Taylor has conducted extensive research on the social impact of urban development."
            "Taylor advocates for inclusive city planning that considers diverse community needs."
            "Taylor participated in a community-led urban renewal project."
            "Taylor is interested in how urban environments affect mental health."
            "Taylor recently attended a seminar on the future of work in smart cities."
            "Taylor observes that gentrification often leads to displacement of long-term residents."
            "Taylor sees a trend towards community-driven development projects."
            "Taylor notes the importance of affordable housing in maintaining social equity."
            "Taylor finds that well-designed public spaces can foster social cohesion."
            "Taylor is concerned about the social implications of widespread surveillance in smart cities."}, "phi3", 0.2, 4096),
]

god_agent = OneAgent({"name": "God", "age": 59, "traits": "all-knowing", "status": "omnipotent","initial_memory": ""}, "phi3", 0.2, 4096)

agents_general_memory = [
    "The agents recently collaborated on a project to design a new smart district in a major city."
    "They attended a workshop on integrating renewable energy sources into urban infrastructure."
    "The agents frequently discuss the balance between technological advancements and social equity."
    "They are all aware of a recent government initiative to promote green transportation."
    "The agents have diverse perspectives but share a common goal of creating sustainable and livable cities for all residents."
]

round_table_chat = DesignerRoundTableChat(agents, topic)
round_table_chat.run_round(10)
