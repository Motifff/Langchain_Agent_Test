import json

# Step 1: Read the JSON file
with open('embedding_projection/data_vectorremove.json', 'r') as file:
    data = json.load(file)

# Step 2: Extract the relevant data
agents = data['agents']
rounds = data['runtime']

# Step 3: Generate LaTeX content
latex_content = r"""
\documentclass{article}
\usepackage{array}
\usepackage[toc,page]{appendix}
\begin{document}

\begin{appendix}
\section*{Eight Rounds of Data}

\begin{tabular}{| m{2cm} | m{3cm} | m{5cm} | m{5cm} |}
\hline
\textbf{Round} & \textbf{Topic} & \textbf{Proposals} & \textbf{Votes} \\
\hline
"""

for i, round_data in enumerate(rounds):
    latex_content += f"{i+1} & {round_data['topic']['text']} & "
    
    # Combine proposals in a three-stack cell
    proposals_cell = ""
    for j, agent in enumerate(agents):
        proposal = round_data['proposals'][j].get("proposal", 'No proposal provided')
        proposals_cell += f"\\textbf{{Agent {agent['name']}}}: {proposal} \\\\ \n"
    
    # Combine votes in a three-stack cell
    votes_cell = ""
    for j, agent in enumerate(agents):
        vote = round_data['vote']['process'][j]
        votes_cell += f"\\textbf{{Agent {agent['name']}}}: {vote} \\\\ \n"
    
    latex_content += f"{proposals_cell} & {votes_cell} \\\\\n\\hline\n"

latex_content += r"""
\end{tabular}
\end{appendix}
\end{document}
"""

# Step 4: Write to a LaTeX file
with open('embedding_projection/Tex/output.tex', 'w') as file:
    file.write(latex_content)

print("LaTeX file generated successfully.")