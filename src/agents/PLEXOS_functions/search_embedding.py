# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:20:47 2024

@author: ENTSOE
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Sample DataFrame creation


# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find the closest description match
def find_closest_name(prompt, df, search_column):
    # Compute the embedding for the prompt
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    
    # Compute embeddings for all descriptions in the DataFrame
    descriptions = df['Description'].tolist()
    description_embeddings = model.encode(descriptions, convert_to_tensor=True)
    
    # Find the closest description
    closest_index = util.semantic_search(prompt_embedding, description_embeddings, top_k=1)[0][0]['corpus_id']
    closest_name = df.iloc[closest_index][search_column]
    return closest_name

# def find_multiple_values(prompt, df, search_column, threshold=0.3):
#     words = [word.strip() for word in prompt.split(",")]  # Improved tokenization
    
#     # Flatten carriers and aliases into a single lookup dictionary
#     carrier_dict = {}
#     for index, row in df.iterrows():
#         carrier_dict[row[search_column]] = row[search_column]
#         for alias in row['Alias 1']:
#             carrier_dict[alias] = row[search_column]
#         for alias in row['Alias 2']:
#             carrier_dict[alias] = row[search_column]

#     # Keyword matching
#     keyword_matches = [carrier_dict[word] for word in words if word in carrier_dict]
    
#     # Prepare for embedding matching
#     carriers = list(set(carrier_dict.values()))  # Unique list of primary carrier names
#     carrier_embeddings = model.encode(carriers, convert_to_tensor=True)
    
#     # Embedding matching
#     word_embeddings = model.encode(words, convert_to_tensor=True)
#     embedding_matches = []
#     for word, word_embedding in zip(words, word_embeddings):
#         distances = util.cos_sim(word_embedding, carrier_embeddings)
#         max_similarity = max(distances[0])
#         if max_similarity > threshold:
#             matching_carrier = carriers[distances[0].argmax().item()]
#             embedding_matches.append(matching_carrier)

#     # Combine results and remove duplicates
#     combined_results = set(keyword_matches + embedding_matches)
#     return list(combined_results)

def find_multiple_values(prompt, df, categories, threshold=0.5):
    # Improved tokenization that handles commas and the word "and"
    import re
    words = re.split(r',\s*|\sand\s', prompt)

    # Flatten categories into a single lookup dictionary
    term_dict = {}
    for index, row in df.iterrows():
        for category in categories:
            term_dict[row[category]] = row[category]
            if 'Alias 1' in row and category in row['Alias 1']:
                for alias in row['Alias 1'][category]:
                    term_dict[alias] = row[category]
            if 'Alias 2' in row and category in row['Alias 2']:
                for alias in row['Alias 2'][category]:
                    term_dict[alias] = row[category]

    # Keyword matching
    keyword_matches = [term_dict[word] for word in words if word in term_dict]
    
    # Prepare for embedding matching
    terms = list(set(term_dict.values()))  # Unique list of primary category names
    term_embeddings = model.encode(terms, convert_to_tensor=True)
    
    # Embedding matching
    word_embeddings = model.encode(words, convert_to_tensor=True)
    embedding_matches = []
    for word, word_embedding in zip(words, word_embeddings):
        distances = util.cos_sim(word_embedding, term_embeddings)
        max_similarity = max(distances[0])
        if max_similarity > threshold:
            matching_term = terms[distances[0].argmax().item()]
            embedding_matches.append(matching_term)

    # Combine results and remove duplicates
    combined_results = set(keyword_matches + embedding_matches)
    return list(combined_results)

# Example usage
if __name__ == '__main__':
    prompt = "Run a model for Belgium, Netherlands, Portugal and Germany for the electricity system"
    # df = pd.read_csv(r'categorisation\function_map.csv')
    # data = {
    #     'Category':[ 'Electricity',   'Hydrogen',     'E-fuel',            'Methane', 'Transport',                    'Heat'],
    #     'Alias 1':       [['Power'],        ['H2'],         ['RFNBO'],           ['CH4'],   ['EVs'],                        ['Hybrid Heat Pumps']], 
    #     'Alias 2':       [[''],             [''],   ['Synthetic Fuels'], [''],      ['Electric Vehicles'],          ['Residential Heating'] ]}
    
    # df = pd.DataFrame(data)
    
    # result = find_multiple_values(prompt, df, 'Category')
    # print(result)
