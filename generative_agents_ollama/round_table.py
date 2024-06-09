import os, random

from termcolor import colored
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

from utils import OneAgent
from GA._init_ import GenerativeAgent

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# This is a workaround for a known issue with the transformers library.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DesignerRoundTableChat:
    def __init__(self, agents: List[OneAgent], topic: str):
        self.agents = agents
        self.topic = topic
        self.init_topic = topic
        self.proposal_won = ""
        self.design_proposals = []
        self.votes = []
        self.llm = ChatOllama(
            model="phi3",
            keep_alive=-1,
            temperature=0.2,
            max_new_tokens=4096
        )

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

    def get_embedding_vector(model: str, text: str):
        """Get the embedding vector for a given text."""
        embeddings_model = OllamaEmbeddings(model="qwen2")
        # get embedding vector
        o_vec = embeddings_model.embed_query(text)
        
        return 
    
    # First we should init the initial memory for each agent
    def init_memory(self):
        for each in self.agents:
            each.init_memory()

    def run_round(self,steps,now: Optional[datetime] = None):        
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
                steps -= 1
                print(colored(f"Round is finished", "green"))
                self.generate_new_round_topic()
                #clean this round's data
                self.design_proposals = []
                self.votes = []
                self.proposal_won = ""
        

# Example
topic = "Design and development of future cities"

# Define the agents with self, info: Dict, model: str, temperature: float, max_new_tokens: int

agents = [
    OneAgent({"name": "Alex", "age": 25, "traits": "innovative, analytical and urban planner", "status": "working on a robotic project","initial_memory": [
                    "Alex has worked on several smart city projects across different continents.",
                    "Alex believes in the potential of technology to solve urban challenges.",
                    "Alex had a mentor who emphasized the importance of community involvement in urban planning.",
                    "Alex recently attended a conference on sustainable city development.",
                    "Alex enjoys reading about the latest advancements in renewable energy.",
                    "Alex notes the increasing integration of IoT devices in urban infrastructure.",
                    "Alex observes a trend towards mixed-use developments in major cities.",
                    "Alex finds that public opinion is often divided on the implementation of autonomous vehicles.",
                    "Alex sees the growing importance of green spaces in urban areas for residents' well-being.",
                    "Alex is concerned about the digital divide and its impact on equitable access to smart city benefits."
                ]
              }, "qwen2", 0.1, 512),
    OneAgent({"name": "Sally", "age": 30, "traits": "curious,critical,environmental scientist", "status": "dive into books about bio-design","initial_memory": [
                    "Sally has been researching the impact of urbanization on local ecosystems.",
                    "Sally is passionate about reducing carbon footprints in city planning.",
                    "Sally worked on a project that successfully integrated green roofs in a metropolitan area.",
                    "Sally often collaborates with urban planners and architects to promote sustainable practices.",
                    "Sally recently published a paper on the benefits of urban biodiversity.",
                    "Sally notices the rise of eco-friendly building materials in construction.",
                    "Sally is intrigued by the potential of vertical farming in urban settings.",
                    "Sally is concerned about the pollution levels in rapidly growing cities.",
                    "Sally observes that public transportation systems are key to reducing urban emissions.",
                    "Sally finds that cities with robust recycling programs have lower waste management costs.",
                ]
                }, "qwen2", 0.1, 512),
    OneAgent({"name": "Taylor", "age": 35, "traits": "analytical and introverted", "status": "have great passion of graphical design","initial_memory": [
                    "Taylor has conducted extensive research on the social impact of urban development."
                    "Taylor advocates for inclusive city planning that considers diverse community needs."
                    "Taylor participated in a community-led urban renewal project."
                    "Taylor is interested in how urban environments affect mental health."
                    "Taylor recently attended a seminar on the future of work in smart cities."
                    "Taylor observes that gentrification often leads to displacement of long-term residents."
                    "Taylor sees a trend towards community-driven development projects."
                    "Taylor notes the importance of affordable housing in maintaining social equity."
                    "Taylor finds that well-designed public spaces can foster social cohesion."
                    "Taylor is concerned about the social implications of widespread surveillance in smart cities."
                ]
                }, "qwen2", 0.1, 512),
]

agents_general_memory = [
    
]

round_table_chat = DesignerRoundTableChat(agents, topic)
#round_table_chat.init_memory()
round_table_chat.run_round(5)
