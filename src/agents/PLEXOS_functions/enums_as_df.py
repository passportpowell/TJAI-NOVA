# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:33:40 2024

@author: ENTSOE
"""

import pandas as pd
import os

def parse_enums_to_dataframe_v2():
    file_path = os.path.join(os.path.dirname(__file__), 'query_enums.txt')  # Dynamically set the path relative to the script's location
    data = {'Enum': [], 'Name': [], 'ID': []}
    current_enum = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Check if the line is likely an enum header (no '=' present)
            if '=' not in line:
                current_enum = line
            else:
                # Process enum value lines
                parts = line.split(' = ')
                if len(parts) == 2:
                    key, value = parts
                    data['Enum'].append(current_enum)
                    data['Name'].append(key)
                    data['ID'].append(int(value))
                else:
                    continue
    return pd.DataFrame(data)

def extract_df(df, column, column_val, property_name, property_value):
    new_df = df[df['Enum'] == column_val]
    return new_df[new_df[property_name] == property_value]
     
def main():
    df_v2 = parse_enums_to_dataframe_v2()
    df_v2['Property ID'] = df_v2['Name'].apply(lambda x: x.split('_')[1] if '_' in x else None)
    unique_enums = df_v2['Enum'].unique()
    return df_v2, unique_enums

def property_enum_id(self, collection_name, property_value, parent_class_name):
    # column = 'Enum'
    collection_name = collection_name.replace(' ','')
    property_value = property_value.replace(' ','')
    property_value = property_value.replace('&','')
    column_val = f'{parent_class_name}{collection_name}Enum'
    # property_name = 'Name'
    # property_value = 'BuildCost'
    x = extract_df(enum_list, 'Enum', column_val, 'Name', property_value)['ID'].sum()
    # print(column_val, property_value, x)

def find_enum(parent_class_name, collection_name, property_name):
    """
    Searches for an enum code based on the parent class name, collection name, and property name.
    
    Args:
        parent_class_name (str): The name of the parent class.
        collection_name (str): The name of the collection.
        property_name (str): The name of the property.
        enum_list (pd.DataFrame): A DataFrame containing the enums information.
        
    Returns:
        int: The ID of the found enum or None if not found.
    """
    # Define the search terms
    search_terms = [
        f'{parent_class_name}{collection_name}Enum',
        f'{parent_class_name}Out{collection_name}Enum'
    ]
    
    # if property_name == 'Energy Curtailed':
    #     property_name = 'CapacityCurtailed'

    for search_term in search_terms:
        # print('search term', search_term)
        # Filter the DataFrame based on the search term and property name
        property_name = property_name.replace(' ', '')
        filtered_df = enum_list[(enum_list['Enum'] == search_term) & (enum_list['Name'] == property_name)]
        
        if not filtered_df.empty:
            # If a match is found, return the ID
            property_id = filtered_df['ID'].iloc[0]
            return property_id
    
    # Return None if no match is found
    return None
    
# x = extract_df(df_v2, 'ID', 20)    
    
enum_list, unique_enums = main()

if __name__ == '__main__':
    parent_class_name = 'System'
    collection_name = 'GasNodes'
    column = 'Enum'
    property_name = 'Price'
    enum_id = find_enum(parent_class_name, collection_name, property_name)
    ex = enum_list[enum_list['Name'] == 'Price']
    print(enum_id)
    

