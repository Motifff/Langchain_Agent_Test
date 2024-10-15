import json

# Load the JSON data from the file
with open('embedding_projection/data_vectorremove.json', 'r') as file:
    data = json.load(file)

# Function to recursively remove 'vector' keys
def remove_vectors(obj):
    if isinstance(obj, dict):
        # Remove 'vector' key if it exists
        obj.pop('vector', None)
        # Recursively call for all values
        for value in obj.values():
            remove_vectors(value)
    elif isinstance(obj, list):
        # Recursively call for all items in the list
        for item in obj:
            remove_vectors(item)

# Remove all 'vector' keys from the data
remove_vectors(data)

# Save the modified JSON back to the file
with open('embedding_projection/data_vectorremove.json', 'w') as file:
    json.dump(data, file, indent=4)