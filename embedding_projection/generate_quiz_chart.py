import matplotlib.pyplot as plt
import numpy as np

# Extracted summary data from the image
data = np.array([
    [0, 0, 1, 2, 3],   # Manage information from stakeholders
    [0, 0, 1, 1, 4],   # Identify and organize key insights
    [0, 0, 0, 3, 3],   # Switch perspectives effortlessly
    [0, 0, 1, 4, 1],   # Summarize stakeholder viewpoints
    [0, 0, 1, 3, 2],   # Identify conflicts between stakeholders
    [0, 0, 0, 2, 4],   # Synthesize information for compromise
    [0, 0, 0, 3, 3],   # Locate expert input easily
    [0, 0, 1, 1, 4],   # Update expert's perspective easily
    [0, 0, 1, 2, 3],   # Assess impact of new information
    [0, 0, 0, 1, 5],   # Identify emerging trends
    [0, 0, 1, 2, 3],   # Understand evolving stakeholder opinions
    [0, 1, 1, 1, 3],   # Reduce information overload
    [0, 0, 1, 2, 3],   # Minimize cognitive strain
    [0, 0, 0, 2, 4],   # Synthesize insights effectively
    [0, 0, 1, 1, 4],   # Confidence in analysis accuracy
    [0, 0, 1, 1, 4]    # Preference for future use
])

# Labels for each row (corresponding to each question)
labels = [
    "Manage information from stakeholders",
    "Identify and organize key insights",
    "Switch perspectives effortlessly",
    "Summarize stakeholder viewpoints",
    "Identify conflicts between stakeholders",
    "Synthesize information for compromise",
    "Locate expert input easily",
    "Update expert's perspective easily",
    "Assess impact of new information",
    "Identify emerging trends",
    "Understand evolving stakeholder opinions",
    "Reduce information overload",
    "Minimize cognitive strain",
    "Synthesize insights effectively",
    "Confidence in analysis accuracy",
    "Preference for future use"
]

# Colors for each category (Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree)
colors = ['#d9b38c', '#df9173', '#e2cdad', '#b5e6ab', '#83c79b']  # Custom colors for each category

# Category labels
category_labels = ['1 - Strongly Disagree', '2 - Disagree', '3 - Neutral', '4 - Agree', '5 - Strongly Agree']

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize x offset (used to stack the bars horizontally)
x_offset = np.zeros(len(labels))

# Plot the stacked bar chart
for i in range(data.shape[1]):
    ax.barh(labels, data[:, i], left=x_offset, color=colors[i], label=category_labels[i])
    x_offset += data[:, i]

# Add the legend
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Hierarchy View")  # Changed legend position

# Add gridlines for better readability
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Invert y-axis to match the original image order
ax.invert_yaxis()

# Display the plot
plt.tight_layout()
plt.show()
