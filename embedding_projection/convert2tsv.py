import json
import csv

# Load the JSON data from the file
input_file = 'embedding_projection/data.json'
output_vectors_file = 'embedding_projection/vectors.tsv'
output_metadata_file = 'embedding_projection/metadata.tsv'

with open(input_file, 'r') as json_file:
    data = json.load(json_file)

# Prepare to write to vectors TSV
with open(output_vectors_file, 'w', newline='') as vectors_file:
    tsv_writer = csv.writer(vectors_file, delimiter='\t')

    # Write vector data for topics
    for round_data in data['runtime']:
        # Write topic vector
        topic_vector = round_data['topic']['vector']
        tsv_writer.writerow(topic_vector)

        # Write proposal vectors
        for proposal in round_data.get('proposals', []):
            proposal_vector = proposal['vector']
            tsv_writer.writerow(proposal_vector)

# Prepare to write to metadata TSV
with open(output_metadata_file, 'w', newline='') as metadata_file:
    tsv_writer = csv.writer(metadata_file, delimiter='\t')

    # Write header
    tsv_writer.writerow(['Text'])  # Change made here

    # Write round data for topics and proposals
    for round_data in data['runtime']:
        # Write topic metadata
        topic_text = round_data['topic']['text']
        compressed_text = round_data['topic'].get('compressed_text', 'NA')  # Change made here
        tsv_writer.writerow([topic_text])  # Change made here

        # Write proposal metadata
        for proposal in round_data.get('proposals', []):
            proposal_text = proposal['proposal']
            proposal_compressed_text = proposal.get('compressed_text', 'NA')  # Change made here
            tsv_writer.writerow([proposal_text])  # Change made here

print(f'TSV files have been created:\n- {output_vectors_file}\n- {output_metadata_file}')