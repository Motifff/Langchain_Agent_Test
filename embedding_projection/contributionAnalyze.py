import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON data
with open('embedding_projection/data.json') as f:
    data = json.load(f)

# Extract rounds and agents
rounds = data['runtime']
agents = data['agents']

# Prepare to store distances for each agent and topic across rounds
agent_names = [agent['name'] for agent in agents]
distances = {name: [] for name in agent_names}
topic_distances = []

# Calculate distances for each round
for round_data in rounds:
    current_topic_vector = np.array(round_data['topic']['vector'])
    
    # Calculate topic distance for the current round
    topic_distance = np.linalg.norm(current_topic_vector)
    topic_distances.append(topic_distance)
    
    for agent in agents:
        # Assuming each agent has a 'proposal' field with a vector
        # Here we will create a dummy proposal vector for demonstration
        # Replace this with the actual proposal vector if available
        agent_proposal_vector = np.random.rand(len(current_topic_vector))  # Dummy proposal vector
        distance = np.linalg.norm(current_topic_vector - agent_proposal_vector)
        distances[agent['name']].append(distance)

# Visualization
plt.figure(figsize=(12, 6))

# Plot each agent's distance trend
for agent_name in agent_names:
    plt.plot(distances[agent_name], marker='o', label=f'Agent: {agent_name}')

# Plot the topic distance trend
plt.plot(topic_distances, marker='x', color='black', linestyle='--', label='Topic Distance')

plt.title('Distance of Each Agent\'s Proposal to Current Topic Across Rounds')
plt.xlabel('Rounds')
plt.ylabel('Distance to Current Topic')
plt.xticks(ticks=range(len(rounds)), labels=[f'Round {i}' for i in range(len(rounds))])
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()