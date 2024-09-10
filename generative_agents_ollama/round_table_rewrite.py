import os
import json
import asyncio
import numpy as np
from enum import Enum
from termcolor import colored
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

from utils import OneAgent
from GA._init_ import GenerativeAgent

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# This is a workaround for a known issue with the transformers library.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class RoundState(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    CHATINROUND = 3
    ADDMEMORY = 4

class UDPSignalListener:
    def __init__(self, ip="localhost", port=6666):
        self.ip = ip
        self.port = port
        self.state = RoundState.WAITING
        self.last_message = None

    async def listen(self):
        transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
            lambda: self,
            local_addr=(self.ip, self.port)
        )
        print(f"Listening on {self.ip}:{self.port}")

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        try:
            message = json.loads(data.decode())
            if isinstance(message, dict) and 'command' in message:
                if message['command'] == 'start':
                    self.state = RoundState.RUNNING
                    print("running")
                elif message['command'] == 'stop':
                    self.state = RoundState.FINISHED
                elif message['command'] == 'chat_in_round':
                    self.state = RoundState.CHATINROUND
                    self.last_message = message
                elif message['command'] == 'add_memory':
                    self.state = RoundState.ADDMEMORY
                    self.last_message = message
                print(f"Received signal: {message['command']}")
            else:
                print(f"Received invalid message format: {message}")
        except json.JSONDecodeError:
            print(f"Received invalid JSON data: {data.decode()}")
        except Exception as e:
            print(f"Error processing message: {str(e)}")

    def connection_lost(self, exc):
        print("Connection closed")
    

class DesignerRoundTableChat:
    def __init__(self, agents: List[OneAgent], topic: str, jsonFile: Dict,file_path, directional_text, emb_model: str = "phi3"):
        self.agents = agents
        self.topic = topic
        self.init_topic = topic
        self.round_count = 0
        self.data_round = 0
        self.proposal_won = ""
        self.design_proposals = []
        self.votes = []
        self.extreme_words = directional_text
        self.scaler = StandardScaler()
        self.llm = ChatOllama(
            model="phi3",
            keep_alive=-1,
            temperature=0.2,
            max_new_tokens=4096
        )
        self.embeddings_model = OllamaEmbeddings(model=emb_model)
        self.file_path = file_path
        self.data = jsonFile

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt)

    def get_embedding_vector(self, text: str):
        o_vec = self.embeddings_model.embed_query(text)
        extreme_vectors = [self.embeddings_model.embed_query(extreme_text) for extreme_text in self.extreme_words]
        all_vectors = np.vstack([extreme_vectors, o_vec])
        scaled_vectors = self.scaler.fit_transform(all_vectors)
        pca = PCA(n_components=3)
        reduced_vectors = pca.fit_transform(scaled_vectors)
        p_vec = reduced_vectors[-1]
        return [o_vec, p_vec.tolist()]
    
    def init_memory(self):
        for each in self.agents:
            each.init_memory()

    def reset_round_data(self):
        self.design_proposals = []
        self.votes = []
        self.proposal_won = ""

    async def write_json_file(self):
        try:
            async with asyncio.Lock():
                with open(self.file_path, "w") as file:
                    json.dump(self.data, file, indent=4)
            print(colored("Successfully wrote data to JSON file", "green"))
        except Exception as e:
            print(colored(f"Error writing to JSON file: {str(e)}", "red"))

    async def update_data_from_json(self):
        try:
            async with asyncio.Lock():
                with open(self.file_path, "r") as file:
                    self.data = json.load(file)
            self.data_round = self.data["total_round"]
            print(colored(f"Total round is: {self.data_round}", "green"))
        except Exception as e:
            print(colored(f"Error reading JSON file: {str(e)}", "red"))

    async def save_new_proposal(self, new_proposal):
        # Implement logic to save the new proposal separately
        # This is a placeholder and should be modified based on your requirements
        new_proposal_data = {
            "round_count": self.round_count,
            "proposal": {
                "text": new_proposal,
                "vector": self.get_embedding_vector(new_proposal)[0],
                "pca_vector": self.get_embedding_vector(new_proposal)[1],
            }
        }
        
        if "chat_in_round_proposals" not in self.data:
            self.data["chat_in_round_proposals"] = []
        
        self.data["chat_in_round_proposals"].append(new_proposal_data)
        await self.write_json_file()

    async def save_round_results(self, exclude_new_proposal=False):
        round_data = {
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
                } for each in (self.design_proposals[:-1] if exclude_new_proposal else self.design_proposals)
            ],
            "vote": {
                "process": self.votes,
                "win": self.proposal_won
            }
        }
        self.data["runtime"].append(round_data)
        await self.write_json_file()

    async def handle_waiting_state(self):
        print(colored("###System is waiting for signal to the next round###", "blue"))
        await self.update_data_from_json()
        await asyncio.sleep(10)

    async def update_agent_memories(self):
        for agent in self.agents:
            # Add memories for proposals from known agents
            for index, proposal in enumerate(self.design_proposals[:len(self.agents)]):
                agent.agent.memory.save_context(
                    {},
                    { 
                        agent.agent.memory.add_memory_key: f"{self.agents[index].agent.name} proposed {proposal}",
                        agent.agent.memory.now_key: datetime.now(),
                    }
                )
            
            # If there's an additional proposal (user-added), add it as a separate memory
            if len(self.design_proposals) > len(self.agents):
                user_proposal = self.design_proposals[-1]
                agent.agent.memory.save_context(
                    {},
                    {
                        agent.agent.memory.add_memory_key: f"An additional proposal was made: {user_proposal}",
                        agent.agent.memory.now_key: datetime.now(),
                    }
                )
            
            # Add memory for the winning proposal
            agent.agent.memory.save_context(
                {},
                {
                    agent.agent.memory.add_memory_key: f"The winning proposal is {self.proposal_won}",
                    agent.agent.memory.now_key: datetime.now(),
                },
            )

    def generate_design_proposals(self):
        for each in self.agents:
            proposal = each.agent.propose(self.topic)
            print("\n"+each.agent.name+"\n\n"+colored(str(proposal), "green"))
            self.design_proposals.append(proposal)

    def conduct_voting(self):
        for agent in self.agents:
            votes = agent.vote(self.design_proposals)
            self.votes.append(votes)       

    def count_votes(self):
        vote_counts = Counter(self.votes)
        max_count = max(vote_counts.values())
        most_common_numbers = [num for num, count in vote_counts.items() if count == max_count]
        return 999 if len(most_common_numbers) > 1 else most_common_numbers[0]

    def generate_new_round_topic(self) -> str:
        prompt = PromptTemplate.from_template(
            "You are a round table holder and overall topic is {overall_topic}"+
            "You have just finished a round of discussion"+
            "Based on the proposals from the agents, they are {current_round_proposals},"+
            "And the winning proposal is {winning_proposal}"+
            "you need to generate a new topic for the next round of discussion based on given information"+
            "The new topic should be related to the current topic but with a new focus."+
            "The response should be within 3-5 sentences."
        )
        proposals = "\n".join(self.design_proposals)
        kwargs: Dict[str, Any] = dict(
            current_round_proposals=proposals,
            winning_proposal=self.proposal_won,
            current_topic=self.topic,
            overall_topic=self.init_topic,
        )
        response = self.chain(prompt).run(**kwargs).strip()
        print(colored(f"New topic for the next round is: {response}", "red"))
        self.topic = response 
        return response

    async def process_voting_results(self):
        result = self.count_votes()
        if result == 999:
            print(colored("There is a tie, redo the round", "red"))
            return False
        self.proposal_won = self.design_proposals[result-1]
        print(colored(f"The winning proposal is: {self.proposal_won}", "green"))
        await self.update_agent_memories()
        return True

    async def run_rounds(self):
        signal_listener = UDPSignalListener()
        asyncio.create_task(signal_listener.listen())

        while True:
            print(signal_listener.state)
            if signal_listener.state == RoundState.FINISHED:
                break
            elif signal_listener.state == RoundState.RUNNING:
                print("running")
                await self.run_single_round()
                print(colored(f"Current state: Round {self.round_count + 1}, Topic: {self.topic}", "cyan"))
                print(colored(f"Number of proposals: {len(self.design_proposals)}", "cyan"))
                print(colored(f"Number of votes: {len(self.votes)}", "cyan"))
                if self.round_count >= self.data_round:
                    signal_listener.state = RoundState.WAITING
            elif signal_listener.state == RoundState.CHATINROUND:
                await self.handle_chat_in_round()
                signal_listener.state = RoundState.WAITING
            elif signal_listener.state == RoundState.ADDMEMORY:
                await self.handle_add_memory()
                signal_listener.state = RoundState.WAITING
            elif signal_listener.state == RoundState.WAITING:
                await self.handle_waiting_state()
            else:
                print(colored(f"Unhandled state: {signal_listener.state}", "red"))
            
            await asyncio.sleep(1)  # Small delay to prevent busy-waiting

    async def run_single_round(self):
        if self.round_count < self.data_round:
            self.generate_design_proposals()
            self.conduct_voting()
            if not await self.process_voting_results():
                return  # Redo the round if there's a tie
            await self.save_round_results()
            self.round_count += 1
            print(colored(f"Round {self.round_count} is finished", "green"))
            self.generate_new_round_topic()
            self.reset_round_data()
        else:
            print(colored("All rounds completed", "green"))

    async def handle_chat_in_round(self):
        print(colored("Handling chat in round...", "yellow"))
        
        if not signal_listener.last_message or 'content' not in signal_listener.last_message:
            print(colored("Invalid or missing message for chat_in_round", "red"))
            return

        new_proposal = signal_listener.last_message['content']
        signal_listener.last_message = None  # Clear the message after using it
        
        # Add the new proposal to the vote pool
        self.design_proposals.append(new_proposal)
        
        # Conduct voting including the new proposal
        self.conduct_voting()
        
        if not await self.process_voting_results():
            return  # Redo the round if there's a tie
        
        # Save round results, excluding the new proposal from the main round data
        await self.save_round_results(exclude_new_proposal=True)
        
        # Save the new proposal separately
        await self.save_new_proposal(new_proposal)
        
        self.round_count += 1
        print(colored(f"Chat in round {self.round_count} is finished", "green"))
        self.generate_new_round_topic()
        self.reset_round_data()

    async def handle_add_memory(self):
        print(colored("Handling add memory...", "yellow"))
        
        if not signal_listener.last_message or 'content' not in signal_listener.last_message:
            print(colored("Invalid or missing message for add_memory", "red"))
            return

        try:
            agent_number = int(signal_listener.last_message['content']['agent_number'])
            if agent_number < 0 or agent_number >= len(self.agents):
                raise ValueError("Invalid agent number")

            content = signal_listener.last_message['content']['memory']
            signal_listener.last_message = None  # Clear the message after using it
            
            # Add memory to the specified agent
            agent = self.agents[agent_number]
            agent.agent.memory.save_context(
                {},
                {
                    agent.agent.memory.add_memory_key: content,
                    agent.agent.memory.now_key: datetime.now(),
                }
            )
            
            print(colored(f"Added memory to agent {agent_number + 1}: {content}", "green"))
        
        except ValueError as e:
            print(colored(f"Error adding memory: {str(e)}", "red"))

if __name__ == "__main__":
    path = "/Users/motif/Documents/Projs/code/Langchain_Agent_Test/generative_agents_ollama/data.json"
    
    with open(path, "r") as file:
        data = json.load(file)

    topic = data["original_topic"]
    agents = []

    for each in data["agents"]:
        agent = OneAgent(each, "phi3", 0.1, 512)
        agents.append(agent)

    directional_text = [
        "Focus on objective facts and data, analyze existing information.",
        "Represents emotions and intuition, expresses personal feelings and emotions.",
        "Used for critical thinking, identify potential problems and risks.",
        "Symbolizes optimism, looking for positive aspects and opportunities in problems.",
        "Represents creative thinking, encourages new ideas and solutions.",
        "Responsible for organizing and controlling the thinking process, ensuring that thinking is carried out in an orderly manner.",
    ]

    round_table_chat = DesignerRoundTableChat(agents, topic, data, path,directional_text)
    asyncio.run(round_table_chat.run_rounds())