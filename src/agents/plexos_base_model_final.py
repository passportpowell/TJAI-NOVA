# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:32:26 2024

@author: ENTSOE
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import clr
print("clr module imported successfully")
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure imports from local directories
# sys.path.append('utils')
# sys.path.append('functions\\LLMs')
# sys.path.append('functions\Input')
# sys.path.append('C:\\TeraJoule\\AI Assistants\\Emil - AI Engineer')

sys.path.append(os.path.join(os.path.dirname(__file__), 'PLEXOS_functions'))

# from stt import stt
#from PLEXOS_functions import open_ai_calls as oaic
# import lm_studio as lms


# otis test import
import os
import sys

# Add the parent directory of this file to the path (so imports work when run via main.py or directly)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PLEXOS_functions import open_ai_calls as oaic
from PLEXOS_functions import plexos_database_core_methods as pdcm
from PLEXOS_functions import plexos_build_functions_final as pbf


##### otis end

from PLEXOS_functions import plexos_build_functions_final as pbf
from PLEXOS_functions import plexos_database_core_methods as pdcm

from PLEXOS_functions.search_embedding import find_multiple_values as fmv
from PLEXOS_functions.loading_bar import printProgressBar
from PLEXOS_functions import prompt_templates as pt



def ai_call(prompt: str, context: str, model: str = 'gpt-4.1-mini') -> str:
    """
    Wrapper function to handle different model calls for LLMs.

    Parameters
    ----------
    prompt : str
        The textual prompt to send to the model.
    context : str
        Additional context to guide the model.
    model : str, optional
        The LLM model to use, by default 'gpt-4.1-mini'.

    Returns
    -------
    str
        Model response as a string.
    """
    if model == 'gpt-4.1-mini':
        response = oaic.run_open_ai_ns(prompt, context)
    elif model == 'mistral':
        import mistral_llm as mllm
        response = mllm.mistral_call(prompt)
    elif model == 'phi_3':
        import phi_3 as p3
        response = p3.phi3_call(prompt)
    elif model == 'lm_studio':
        response = lms.run_lm_studio_call(prompt)
    else:
        raise ValueError(f"Unsupported model: {model}")
    return response


def extract_country_codes(data: dict) -> list:
    """
    Recursively parses a nested dictionary to extract all 2-letter country codes.

    Parameters
    ----------
    data : dict
        Nested JSON/dict structure containing potential country codes.

    Returns
    -------
    list
        List of all extracted 2-letter codes.
    """
    country_codes = []

    def parse_dict(d):
        for key, value in d.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        parse_dict(item)
            elif isinstance(value, dict):
                parse_dict(value)
            elif isinstance(value, str) and len(value) == 2:
                country_codes.append(value)

    parse_dict(data)
    return country_codes


def extract_countries_with_retries(oai_prompt: str,
                                   context: str,
                                   model: str = 'gpt-4.1-mini',
                                   max_retries: int = 3) -> list:
    """
    Repeatedly calls LLM to extract a list of countries (2-letter codes),
    retrying if none found.

    Parameters
    ----------
    oai_prompt : str
        Prompt containing instructions to extract country codes.
    context : str
        Additional context for the LLM.
    model : str, optional
        Model to call, by default 'gpt-4.1-mini'.
    max_retries : int, optional
        Maximum number of retries, by default 3.

    Returns
    -------
    list
        List of extracted 2-letter country codes.
    """
    for attempt in range(max_retries):
        response_str = ai_call(oai_prompt, context, model=model)
        try:
            countries_json = json.loads(response_str)
            country_list = extract_country_codes(countries_json)
            if country_list:
                return country_list
            else:
                print(f"Attempt {attempt + 1}/{max_retries}: No countries found, retrying...")
        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1}/{max_retries}: JSON decode error, retrying...")

    print("Countries couldn't be found after all retries.")
    return []


### Otis Addition

# In src/agents/plexos_base_model_final.py, replace the extract_countries_with_retries function:

def extract_countries_with_retries(oai_prompt: str,
                                   context: str,
                                   model: str = 'gpt-4.1-mini',
                                   max_retries: int = 3) -> list:
    """
    Use LLM intelligence to extract only the countries actually mentioned in the prompt.
    """
    
    print(f"🧠 Using LLM to intelligently extract countries from: '{oai_prompt}'")
    
    # Use LLM with clear, intelligent instructions
    extraction_prompt = f"""
You are analyzing this text to find which countries are specifically mentioned:

Text: "{oai_prompt}"

TASK: Extract ONLY the countries that are explicitly mentioned in the text.

INSTRUCTIONS:
- Look for country names like "Spain", "Greece", "Denmark", "France", etc.
- Do NOT include countries that are not mentioned
- If the text says "Spain, Greece and Denmark" then only return those 3 countries
- Return the result as a JSON array of 2-letter ISO country codes

EXAMPLES:
Text: "Build a model for Spain, Greece and Denmark" 
Response: ["ES", "GR", "DK"]

Text: "Create a wind model for France"
Response: ["FR"]

Text: "Generate models for Germany and Italy"
Response: ["DE", "IT"]

Now analyze the given text and return ONLY the countries actually mentioned:
"""

    for attempt in range(max_retries):
        try:
            print(f"🔍 LLM extraction attempt {attempt + 1}/{max_retries}")
            
            # Call LLM for intelligent extraction
            response_str = ai_call(extraction_prompt, context, model=model)
            print(f"🧠 LLM response: {response_str}")
            
            # Try to parse JSON response
            try:
                import json
                countries_json = json.loads(response_str)
                
                if isinstance(countries_json, list):
                    # Validate and clean the country codes
                    valid_countries = []
                    for code in countries_json:
                        if isinstance(code, str) and len(code) == 2:
                            valid_countries.append(code.upper())
                    
                    if valid_countries:
                        print(f"✅ LLM successfully extracted countries: {valid_countries}")
                        return valid_countries
                        
            except json.JSONDecodeError:
                print("🔄 JSON parsing failed, trying regex extraction...")
                
                # Fallback: extract 2-letter codes from response
                import re
                country_codes = re.findall(r'\b[A-Z]{2}\b', response_str.upper())
                
                if country_codes:
                    # Remove duplicates while preserving order
                    unique_codes = []
                    for code in country_codes:
                        if code not in unique_codes:
                            unique_codes.append(code)
                    
                    print(f"✅ Extracted via regex fallback: {unique_codes}")
                    return unique_codes
                    
        except Exception as e:
            print(f"❌ LLM extraction attempt {attempt + 1} failed: {str(e)}")
            
    # Final fallback: simple keyword matching for critical countries only
    print("🔄 LLM extraction failed, using simple keyword fallback...")
    
    import re
    prompt_lower = oai_prompt.lower()
    fallback_countries = []
    
    # Only check for the most common countries with word boundaries
    basic_checks = [
        (r'\bspain\b', "ES"),
        (r'\bgreece\b', "GR"), 
        (r'\bdenmark\b', "DK"),
        (r'\bfrance\b', "FR"),
        (r'\bgermany\b', "DE"),
        (r'\bitaly\b', "IT"),
        (r'\buk\b', "GB"),
        (r'\bunited kingdom\b', "GB")
    ]
    
    for pattern, code in basic_checks:
        if re.search(pattern, prompt_lower) and code not in fallback_countries:
            fallback_countries.append(code)
            print(f"🔍 Fallback found: {pattern} -> {code}")
    
    if fallback_countries:
        print(f"✅ Fallback extraction successful: {fallback_countries}")
        return fallback_countries
    
    print("❌ All extraction methods failed")
    return []


def extract_countries_with_retries(oai_prompt: str,
                                   context: str,
                                   model: str = 'gpt-4.1-mini',
                                   max_retries: int = 3) -> list:
    """
    Use LLM intelligence to extract only the countries actually mentioned in the prompt.
    """
    
    # FIXED: Much shorter print statement
    # Extract just the core request from the verbose prompt
    if "Here is the request:" in oai_prompt:
        core_request = oai_prompt.split("Here is the request:")[1].split(".")[0].strip()
        print(f"🧠 Extracting countries from: '{core_request}'")
    else:
        # Fallback: show first 50 characters
        print(f"🧠 Extracting countries from: '{oai_prompt[:50]}...'")
    
    # Use LLM with intelligent context understanding
    extraction_prompt = f"""
You are analyzing this text to find which countries are specifically mentioned:

Text: "{oai_prompt}"

TASK: Extract ONLY the countries that are explicitly mentioned in the text.

INSTRUCTIONS:
- Look for country names like "Spain", "Greece", "Denmark", "France", etc.
- Do NOT include countries that are not mentioned
- If the text says "Spain, Greece and Denmark" then only return those 3 countries
- Return the result as a JSON array of 2-letter ISO country codes

EXAMPLES:
Text: "Build a model for Spain, Greece and Denmark" 
Response: ["ES", "GR", "DK"]

Text: "Create a wind model for France"
Response: ["FR"]

Text: "Generate models for Germany and Italy"
Response: ["DE", "IT"]

Now analyze the given text and return ONLY the countries actually mentioned:
"""

    for attempt in range(max_retries):
        try:
            # FIXED: Shorter progress message
            print(f"🔍 Attempt {attempt + 1}/{max_retries}")
            
            # Call LLM for intelligent extraction
            response_str = ai_call(extraction_prompt, context, model=model)
            
            # FIXED: Only show first 50 chars of response, not the full thing
            print(f"🧠 Response: {response_str[:50]}{'...' if len(response_str) > 50 else ''}")
            
            # Try to parse JSON response
            try:
                import json
                countries_json = json.loads(response_str)
                
                if isinstance(countries_json, list):
                    # Validate and clean the country codes
                    valid_countries = []
                    for code in countries_json:
                        if isinstance(code, str) and len(code) == 2:
                            valid_countries.append(code.upper())
                    
                    if valid_countries:
                        print(f"✅ Extracted countries: {valid_countries}")
                        return valid_countries
                        
            except json.JSONDecodeError:
                print("🔄 JSON failed, using regex...")
                
                # Fallback: extract 2-letter codes from response
                import re
                country_codes = re.findall(r'\b[A-Z]{2}\b', response_str.upper())
                
                if country_codes:
                    # Remove duplicates while preserving order
                    unique_codes = []
                    for code in country_codes:
                        if code not in unique_codes:
                            unique_codes.append(code)
                    
                    print(f"✅ Extracted via regex: {unique_codes}")
                    return unique_codes
                    
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)[:50]}...")
            
    # Final fallback: simple keyword matching for critical countries only
    print("🔄 Using keyword fallback...")
    
    import re
    prompt_lower = oai_prompt.lower()
    fallback_countries = []
    
    # Only check for the most common countries with word boundaries
    basic_checks = [
        (r'\bspain\b', "ES"),
        (r'\bgreece\b', "GR"), 
        (r'\bdenmark\b', "DK"),
        (r'\bfrance\b', "FR"),
        (r'\bgermany\b', "DE"),
        (r'\bitaly\b', "IT"),
        (r'\buk\b', "GB"),
        (r'\bunited kingdom\b', "GB")
    ]
    
    for pattern, code in basic_checks:
        if re.search(pattern, prompt_lower) and code not in fallback_countries:
            fallback_countries.append(code)
    
    if fallback_countries:
        print(f"✅ Found: {fallback_countries}")
        return fallback_countries
    
    print("❌ No countries found")
    return []


def extract_countries_with_retries(oai_prompt: str,
                                   context: str,
                                   model: str = 'gpt-4.1-mini',
                                   max_retries: int = 3) -> list:
    """
    Use LLM intelligence to extract only the countries actually mentioned in the prompt.
    """
    
    # FIXED: Much shorter print statement
    if "Here is the request:" in oai_prompt:
        core_request = oai_prompt.split("Here is the request:")[1].split(".")[0].strip()
        print(f"🧠 Extracting countries from: '{core_request}'")
    else:
        print(f"🧠 Extracting countries from: '{oai_prompt[:50]}...'")
    
    # Use LLM with intelligent context understanding
    extraction_prompt = f"""
You are analyzing this text to find which countries are specifically mentioned:

Text: "{oai_prompt}"

TASK: Extract ONLY the countries that are explicitly mentioned in the text.

INSTRUCTIONS:
- Look for country names like "Spain", "Greece", "Denmark", "France", etc.
- Do NOT include countries that are not mentioned
- If the text says "Spain, Greece and Denmark" then only return those 3 countries
- Return the result as a JSON array of 2-letter ISO country codes

EXAMPLES:
Text: "Build a model for Spain, Greece and Denmark" 
Response: ["ES", "GR", "DK"]

Text: "Create a wind model for France"
Response: ["FR"]

Text: "Generate models for Germany and Italy"
Response: ["DE", "IT"]

Now analyze the given text and return ONLY the countries actually mentioned:
"""

    for attempt in range(max_retries):
        try:
            print(f"🔍 Attempt {attempt + 1}/{max_retries}")
            
            # Call LLM for intelligent extraction
            response_str = ai_call(extraction_prompt, context, model=model)
            print(f"🧠 Response: {response_str[:50]}{'...' if len(response_str) > 50 else ''}")
            
            # Try to parse JSON response
            try:
                import json
                countries_json = json.loads(response_str)
                
                if isinstance(countries_json, list):
                    # Validate and clean the country codes
                    valid_countries = []
                    for code in countries_json:
                        if isinstance(code, str) and len(code) == 2:
                            valid_countries.append(code.upper())
                    
                    if valid_countries:
                        print(f"✅ Extracted countries: {valid_countries}")
                        return valid_countries
                        
            except json.JSONDecodeError:
                print("🔄 JSON failed, using regex...")
                
                # Fallback: extract 2-letter codes from response
                import re
                country_codes = re.findall(r'\b[A-Z]{2}\b', response_str.upper())
                
                if country_codes:
                    # Remove duplicates while preserving order
                    unique_codes = []
                    for code in country_codes:
                        if code not in unique_codes:
                            unique_codes.append(code)
                    
                    print(f"✅ Extracted via regex: {unique_codes}")
                    return unique_codes
                    
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)[:50]}...")
    
    # FIXED: If LLM fails completely, return empty list instead of hardcoded fallback
    # Let the calling code handle the empty list appropriately
    print("❌ All LLM extraction attempts failed")
    return []



# Also enhance the extract_country_codes function:
def extract_country_codes(data: dict) -> list:
    """
    Enhanced country code extraction from nested data.
    """
    country_codes = []

    def parse_dict(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            parse_dict(item)
                        elif isinstance(item, str) and len(item) == 2 and item.isupper():
                            country_codes.append(item)
                elif isinstance(value, dict):
                    parse_dict(value)
                elif isinstance(value, str) and len(value) == 2 and value.isupper():
                    country_codes.append(value)
        elif isinstance(d, list):
            for item in d:
                if isinstance(item, str) and len(item) == 2 and item.isupper():
                    country_codes.append(item)
                elif isinstance(item, dict):
                    parse_dict(item)

    parse_dict(data)
    return list(set(country_codes))  # Remove duplicates


### Otis End




def filter_dataset(df: pd.DataFrame, filter_set: list, filter_columns: str) -> pd.DataFrame:
    """
    Filters a DataFrame based on a given set of values in a specified column.
    Keeps rows only if the column’s value is in the filter_set.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter.
    filter_set : list
        List of acceptable values.
    filter_columns : str
        Column name to filter on.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    filter_set_with_hyphen = set(filter_set) | {'-'}
    filtered = df[df[filter_columns].isin(filter_set_with_hyphen)]
    if 'Default' in df.columns:
        filtered = filtered[filtered['Default'] == 'Yes']
    return filtered


def create_sample_dataframe() -> pd.DataFrame:
    """
    Creates a sample DataFrame for demonstration.

    Returns
    -------
    pd.DataFrame
    """
    data = {
        'Category_ID': list(range(1, 16)),
        'Class_Category': ['Electricity'] * 15,
        'Class': ['Generator'] * 15,
        'Object_Category': [
            'Solar Thermal Expansion', 'Other RES', 'Onshore Wind Expansion',
            'Hard coal', 'Rooftop Solar Tertiary', 'Onshore Wind', 'Heavy oil',
            'Rooftop Tertiary Solar Expansion', 'Pump Storage - closed loop',
            'Solar PV Expansion', 'DSR Industry', 'Offshore Wind Radial',
            'Solar PV', 'RoR and Pondage', 'Bio Fuels'
        ],
        'Default': [
            'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'
        ],
        'Type': [
            'Expansion', 'Dispatch', 'Expansion', 'Dispatch', 'Dispatch',
            'Dispatch', 'Dispatch', 'Expansion', 'Dispatch', 'Dispatch',
            'Dispatch', 'Dispatch', 'Dispatch', 'Dispatch', 'Dispatch'
        ],
        'Property_Type': [
            'Renewable Generator Expansion', 'Renewable Generator',
            'Renewable Generator Expansion', 'Thermal', 'Renewable Generator',
            'Renewable Generator', 'Thermal', 'Renewable Generator Expansion',
            'PS Closed', 'Renewable Generator Expansion', 'Offer Price',
            'Renewable Generator', 'Renewable Generator', 'ROR and Pondage',
            'Thermal'
        ],
        'Description': ['-'] * 15,
        'Remove': ['-'] * 15,
        'Default_Structure': ['-'] * 15,
        'Abbreviation': ['-'] * 15
    }
    return pd.DataFrame(data)


def check_file_exists(prompt: str,
                      filename: str,
                      modelname: str,
                      replace: bool = False,
                      model: str = 'gpt-4.1-mini') -> str:
    """
    Checks if a file exists on disk. If `replace` is False and file exists,
    calls the LLM for a new name.

    Parameters
    ----------
    prompt : str
        Prompt to send to the LLM if file name is taken.
    filename : str
        Desired filename.
    modelname : str
        Name of the model or resource being saved.
    replace : bool, optional
        Whether to overwrite existing file, by default False.
    model : str, optional
        LLM to call, by default 'gpt-4.1-mini'.

    Returns
    -------
    str
        Final filename (possibly modified if name was taken).
    """
    if replace:
        return filename
    else:
        if os.path.exists(filename):
            print("File with the name exists, finding a new name...")
            context = "We need a new file name for the PLEXOS model"
            new_name = ai_call(f"{prompt}, {modelname} is already taken", context, model)
            print(f"Proposed new file name: {new_name}")
            return os.path.join(os.path.dirname(__file__), "PLEXOS_models", new_name)
        return filename


def initiate_file(high_level_prompt: str, model: str = 'gpt-4.1-mini') -> str:
    """
    Creates a .xml filename for the PLEXOS model. If a file with that name
    exists, requests a new one from LLM.

    Parameters
    ----------
    high_level_prompt : str
        User instructions or context describing the model scope.
    model : str, optional
        LLM to use, by default 'gpt-4.1-mini'.

    Returns
    -------
    str
        Valid path to the newly created or selected .xml file.
    """
    context = (
        "You are building a PLEXOS model for a client. The model will be used "
        "to simulate energy systems and optimize energy resources. We are "
        "initiating the environment."
    )
    sub_prompt = (
        "Please give a title for my PLEXOS file name. Respond ONLY with a filename "
        "ending in .xml (e.g., new_model.xml). No extra text or punctuation."
    )

    prompt = f"{high_level_prompt}\n{sub_prompt}"
    model_name = ai_call(prompt, context, model=model)
    filename = os.path.join(os.path.dirname(__file__), "PLEXOS_models", model_name)
    print(f"Proposed file name: {model_name}")
    filename = check_file_exists(prompt, filename, model_name)
    return filename


def process_dataframes(df1: pd.DataFrame,
                       df2: pd.DataFrame,
                       main_column: str) -> pd.Series:
    """
    Concatenates two DataFrames, cleans up the specified column by removing
    periods and spaces, and returns the cleaned column as a Series.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame to concatenate.
    df2 : pd.DataFrame
        Second DataFrame to concatenate.
    main_column : str
        The column to clean and return.

    Returns
    -------
    pd.Series
        Series of cleaned column values.
    """
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df[main_column] = (
        combined_df[main_column]
        .str.replace('.', '', regex=False)
        .str.replace(' ', '', regex=False)
    )
    return combined_df[main_column]


def extract_property_value(property_list: pd.DataFrame, idx: int) -> str:
    """
    Retrieves the default property value from the property list DataFrame.

    Parameters
    ----------
    property_list : pd.DataFrame
        A DataFrame containing property definitions.
    idx : int
        Row index from which to extract the 'Value_default'.

    Returns
    -------
    str
        The property value.
    """
    return property_list.loc[idx, 'Value_default']


def split_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits the 'Collection' column into two new columns: 'Parent_Class_Name' and
    'Child_Class_Name'. Removes spaces and periods for consistency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain 'Collection' column.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with new columns for parent and child class names.
    """
    df[['Parent_Class_Name', 'Child_Class_Name']] = df['Collection'].str.split('.', expand=True)
    df['Parent_Class_Name'] = df['Parent_Class_Name'].str.replace(' ', '')
    df['Child_Class_Name'] = df['Child_Class_Name'].str.replace(' ', '')
    df['Collection'] = df['Collection'].str.replace(' ', '').str.replace('.', '')
    return df


def filter_data(plexos_prompt_sheet: dict,
                high_level_prompt: str,
                model: str = 'gpt-4.1-mini') -> tuple:
    """
    Filters out the relevant categories, objects, and properties from the
    PLEXOS Model Builder xlsx based on user instructions.

    Parameters
    ----------
    plexos_prompt_sheet : dict of DataFrames
        Dictionary of DataFrames loaded from an Excel workbook, containing
        carrier_class_cat_sheet, carrier_classes_sheet, etc.
    high_level_prompt : str
        The user instruction describing the model scope.
    model : str, optional
        LLM to use for extracting categories and countries, by default 'gpt-4.1-mini'.

    Returns
    -------
    tuple
        filtered_object_cats, filtered_objects, carrier_property_cat_sheet,
        filtered_setting_object_cats, filtered_setting_objects,
        setting_membership_sheet, filtered_object_memberships
    """
    context = (
        "You are building a PLEXOS model for a client. The model will be used to "
        "simulate energy systems and optimize energy resources. We are initiating the environment."
    )

    # Read base data
    node_map = pd.read_csv(os.path.join(os.path.dirname(__file__), "PLEXOS_inputs", "node_mapping.csv"))
    reduced_map = node_map.drop(['Longitude', 'Latitude', 'PLEXOS_Region', 'Node'], axis=1).drop_duplicates().reset_index(drop=True)

    # carrier_class_cat_sheet = plexos_prompt_sheet['carrier_class_cat_sheet'].fillna('-')
    # carrier_classes_sheet = plexos_prompt_sheet['carrier_classes_sheet'].fillna('-')
    # carrier_object_cat_sheet = plexos_prompt_sheet['carrier_object_cat_sheet'].fillna('-')
    # carrier_object_sheet = plexos_prompt_sheet['carrier_object_sheet'].fillna('-')
    # carrier_property_cat_sheet = plexos_prompt_sheet['carrier_property_cat_sheet']
    energy_alias_sheet = plexos_prompt_sheet['energy_alias_sheet'].fillna('-')
    classes_sheet = plexos_prompt_sheet['classes_sheet'].fillna('-')
    energy_carrier_sheet = plexos_prompt_sheet['energy_carrier_sheet'].fillna('-')
    object_sheet = plexos_prompt_sheet['object_sheet'].fillna('-')
    membership_sheet = plexos_prompt_sheet['membership_sheet'].fillna('-')
    property_sheet = plexos_prompt_sheet['property_sheet']

    membership_sheet = membership_sheet.applymap(
                        lambda x: x.strip() if isinstance(x, str) else x
                        )

    # object_membership_sheet = plexos_prompt_sheet['object_membership_sheet'].fillna('-')
    # setting_membership_sheet = plexos_prompt_sheet['setting_membership_sheet'].fillna('-')

    # setting_object_cat_sheet = plexos_prompt_sheet['setting_object_cat_sheet'].fillna('-')
    # settings_objects_sheet = plexos_prompt_sheet['settings_objects'].fillna('-')

    # 1) Extract categories from high level prompt
    pps_json = energy_alias_sheet.to_json(orient='index')
    categories_embed = fmv(high_level_prompt, energy_alias_sheet, ['Category'], threshold=0.3)
    check_prompt = f"{high_level_prompt}.\nEmbedding guess: {categories_embed}"
    oai_prompt = pt.plexos_prompt_sheet_categorization(high_level_prompt, pps_json, 'Category')

    categories_str = ai_call(oai_prompt, context, model=model)
    cat_json = json.loads(categories_str)
    filter_values = list(cat_json.values()) + ['-']
    print(filter_values)

    # Filter data using extracted categories
    filtered_classes = filter_dataset(classes_sheet, filter_values, 'Class_Category')
    filtered_object_cats = filter_dataset(energy_carrier_sheet, filter_values, 'Class_Category')
    # filtered_object_cats = filtered_object_cats[filtered_object_cats['Default'] == 'Yes']

    # filtered_setting_object_cats = filter_dataset(setting_object_cat_sheet, filter_values, 'Class_Category')
    # filtered_setting_object_cats = filtered_setting_object_cats[filtered_setting_object_cats['Default'] == 'Yes']

    filtered_objects = filter_dataset(object_sheet,
                                      list(set(filtered_object_cats['Class'])),
                                      'Class')
    filtered_objects = filter_dataset(filtered_objects,
                                      list(set(filtered_object_cats['Object_Category'])),
                                      'Category')

    # filtered_setting_objects = filter_dataset(settings_objects_sheet,
    #                                          list(set(filtered_setting_object_cats['Object_Category'])),
    #                                          'Category')

    # 2) Extract countries
    map_json = reduced_map.to_json(orient='index')
    countries_embed = fmv(high_level_prompt, reduced_map, ['Country'], threshold=0.5)
    country_prompt = pt.extract_countries(countries_embed, high_level_prompt)
    final_prompt = pt.plexos_prompt_sheet_categorization(country_prompt, map_json, 'Regions and Countries')

    country_list = extract_countries_with_retries(final_prompt, context, model=model)
    print("Embedding-based guess for countries:", countries_embed)
    print("Extracted countries from LLM:", country_list)

    # Classes and Categories that are default to be added
    default_class_cats = energy_carrier_sheet[
        energy_carrier_sheet['Class'].isin([
            'Fuel', 'Emission', 'Horizon', 'Diagnostic', 'DataFile',
            'LTPlan', 'Model', 'MTSchedule', 'PASA', 'Performance',
            'Report', 'Scenario', 'Stochastic', 'STSchedule', 'Variable'
        ])
    ][['Class', 'Object_Category']].reset_index(drop=True)

    # Filter objects by the countries found
    filtered_objects = filtered_objects[
        (filtered_objects['Country 1'].str[:2].isin(country_list)) |
        (filtered_objects['Country 2'].str[:2].isin(country_list)) |
        (filtered_objects['Class'].isin(default_class_cats['Class'].unique())) |
        (filtered_objects['Category'].isin(['-']))
    ].set_index('Name')

    # Membership filtering
    membership_sheet = split_point(membership_sheet)
    filtered_object_memberships = membership_sheet[
        membership_sheet['Parent Name'].str[:2].isin(country_list) |
        (
            (membership_sheet['Parent Name'] == 'System') &
            (membership_sheet['Child Name'].str[:2].isin(country_list))
        ) |
        membership_sheet['Parent Category'].isin(default_class_cats['Object_Category'].unique())
    ]

    # Action dictionary to designate integer
    action_dict = {'=': 0, '×': 1, '÷': 2, '+': 3, '-': 4, '^': 5, '?': 6}

    # Properties filtering
    filtered_property_sheet = property_sheet[
        ((property_sheet['Parent_Name'] == 'System') & 
        (property_sheet['Child_Name'].str[:2].isin(country_list))) |
        
        ((property_sheet['Parent_Name'] != 'System') & 
        (property_sheet['Parent_Name'].str[:2].isin(country_list)))
    ]
    filtered_property_sheet = filtered_property_sheet.reset_index(drop=True)

    # Apply mapping to a column (e.g., 'Action') in the DataFrame
    filtered_property_sheet['Action'] = filtered_property_sheet['Action'].map(
        lambda x: action_dict.get(str(x).strip()) if str(x).strip().lower() not in ['nan', ''] else None
    )

    filtered_property_sheet['Value'] = pd.to_numeric(
        filtered_property_sheet['Value'], errors='coerce'
    )

    # # Creating Collection_Name column in Property Sheet
    # # Step 1: Create a lookup table with unique Parent_Name → Category
    # lookup_df = (
    #     filtered_objects[['Category']]
    #     .reset_index()
    #     .drop_duplicates(subset='Name')
    #     .rename(columns={'Name': 'Parent_Name', 'Category': 'Parent_Category'})
    # )

    # # Step 2: Merge into property sheet
    # filtered_property_sheet = filtered_property_sheet.merge(lookup_df, on='Parent_Name', how='left')

    # # Step 3: Remove rows where Parent_Category is missing (i.e., no match in filtered_objects)
    # filtered_property_sheet = filtered_property_sheet[
    #     (filtered_property_sheet['Parent_Name'] == 'System') |  # keep 'System'
    #     (filtered_property_sheet['Parent_Category'].notna())     # keep only matched rows
    # ]

    # # Step 4: Assign Collection_Name
    # filtered_property_sheet['Collection_Name'] = np.where(
    #     filtered_property_sheet['Parent_Name'] == 'System',
    #     'System.' + filtered_property_sheet['Collection'],
    #     filtered_property_sheet['Parent_Category'] + '.' + filtered_property_sheet['Collection']
    # )

    # Filter by valid categories
    valid_categories = list(set(filtered_object_cats['Object_Category'])) + ['-']
    filtered_objects = filtered_objects[filtered_objects['Category'].isin(valid_categories)].reset_index()

    dict_filters = {
        'filtered_object_cats': filtered_object_cats,
        'filtered_objects': filtered_objects,
        'filtered_properties': filtered_property_sheet,
        'filtered_memberships': filtered_object_memberships
    }

    return dict_filters


def _common_build_steps(db,dict_filters):
    """
    Helper function that executes the common steps for building:
      - Creating categories 
      - Creating objects
      - Creating memberships
      - Creating properties

    Parameters
    ----------
    db : object
        PLEXOS database connection.
    filtered_setting_object_cats : pd.DataFrame
    settings_objects : pd.DataFrame
    filtered_object_cats : pd.DataFrame
    filtered_objects : pd.DataFrame
    carrier_property_cat_sheet : pd.DataFrame
    setting_membership_sheet : pd.DataFrame
    filtered_object_memberships : pd.DataFrame

    Returns
    -------
    tuple
        (class_list, collection_list) for final model closure steps.
    """
    missing_objects = {}
    missing_memberships = {}
    missing_properties = {}

    # filtered_setting_object_cats = dict_filters['filtered_setting_object_cats']
    filtered_object_cats = dict_filters['filtered_object_cats']
    filtered_objects = dict_filters['filtered_objects']
    filtered_property_sheet = dict_filters['filtered_properties']
    filtered_object_memberships = dict_filters['filtered_memberships']

    # filtered_setting_objects = dict_filters['filtered_setting_objects']
    # settings_membership_sheet = dict_filters['setting_membership_sheet']

    # print("----------")
    # print("Filtered Dictionary: ")
    # print(dict_filters['filtered_setting_object_cats'])
    # print("----------")
    # # 1) Create Setting Categories
    # for idx, row in filtered_setting_object_cats.iterrows():
    #     catname = row['Object_Category']
    #     class_id_name = row['Class'].replace(' ','')
    #     objclassid = pbf.extract_enum(db, class_id_name, 'Class')
    #     # print("Object Class: ", objclassid)
    #     pdcm.add_category(db, objclassid, catname)
    #     printProgressBar(idx + 1, len(filtered_setting_object_cats), 'Creating Setting Categories', catname)

    # 1) Create Categories
    for idx, row in filtered_object_cats.iterrows():
        catname = row['Object_Category']
        class_id_name = row['Class'].replace(' ','')
        objclassid = pbf.extract_enum(db, class_id_name, 'Class')
        pdcm.add_category(db, objclassid, catname)
        printProgressBar(idx + 1, len(filtered_object_cats), 'Creating Categories', catname)

    # 3) Create Setting 
    # For convenience, set a MultiIndex on filtered_setting_object_cats
    # so we can match them for property creation
    # if not filtered_setting_object_cats.index.names == ['Object_Category', 'Class']:
    #     filtered_setting_object_cats.set_index(['Object_Category', 'Class'], inplace=True, drop=True)

    # for i in filtered_setting_objects.index:
    #     object_name = str(filtered_setting_objects.loc[i, 'Name']) if not isinstance(filtered_setting_objects.loc[i, 'Name'], str) else filtered_setting_objects.loc[i, 'Name']
    #     pbf.add_object_to_plexos(db, filtered_setting_objects, i, missing_objects)
    #     printProgressBar(i + 1, len(filtered_setting_objects), 'Creating Setting Objects', str(i))

    # 2) Create Objects
    # For convenience, set a MultiIndex on filtered_object_cats
    # so we can match them for property creation
    if not filtered_object_cats.index.names == ['Object_Category', 'Class']:
        filtered_object_cats.set_index(['Object_Category', 'Class'], inplace=True, drop=True)

    for i in filtered_objects.index:
        object_name = str(filtered_objects.loc[i, 'Name']) if not isinstance(filtered_objects.loc[i, 'Name'], str) else filtered_objects.loc[i, 'Name']
        pbf.add_object_to_plexos(db, filtered_objects, i, missing_objects)
        printProgressBar(i + 1, len(filtered_objects), 'Creating Objects', str(i))

    # # 5) Create Setting Memberships
    # for i in filtered_setting_objects.index:
    #     object_name = str(filtered_setting_objects.loc[i, 'Name']) if not isinstance(filtered_setting_objects.loc[i, 'Name'], str) else filtered_setting_objects.loc[i, 'Name']
    #     pbf.create_memberships(db, object_name, settings_membership_sheet, missing_memberships)
    #     printProgressBar(i + 1, len(filtered_setting_objects), 'Creating Setting Memberships', str(i))

    # 3) Create Memberships
    for i in filtered_objects.index:
        object_name = str(filtered_objects.loc[i, 'Name']) if not isinstance(filtered_objects.loc[i, 'Name'], str) else filtered_objects.loc[i, 'Name']
        pbf.create_memberships(db, object_name, filtered_object_memberships, missing_memberships)
        printProgressBar(i + 1, len(filtered_objects), 'Creating Memberships', str(i))

    # 4) Ad Properties
    # for i in filtered_objects.index:
        # category = filtered_objects.loc[i, 'Category']
        # pbf.add_property_to_plexos(db, filtered_object_cats, category, objclassname, carrier_property_cat_sheet, object_name, filtered_object_memberships, missing_properties)
        # objclassname = filtered_objects.loc[i, 'Class']
        # object_name = str(filtered_objects.loc[i, 'Name']) if not isinstance(filtered_objects.loc[i, 'Name'], str) else filtered_objects.loc[i, 'Name']

    # 4) Ad Properties
    for i, row in filtered_property_sheet.iterrows():
        parent_name = str(row['Parent_Name']).strip()
        child_name = str(row['Child_Name']).strip()

        collection_name = str(row['Collection_Name'])
        collection_name = collection_name.replace('.','').replace(' ','').strip()

        property_name = str(row['Property']).strip()
        value = row['Value']

        scenario = row['Scenario'] if 'Scenario' in filtered_property_sheet.columns and pd.notna(row.Scenario) else None

        kwargs = pbf.clean_kwargs(
            datafile = row['Data_File'],
            band_id = row['Band'],
            date_from = row['Date_From'],
            date_to = row['Date_To'],
            variable = row['Expression'],
            pattern = row['Timeslice'],
            action = row['Action']
        )

        pbf.add_property(
            db = db,
            collection_name = collection_name,
            parent_object_name = parent_name,
            child_name = child_name,
            property_name = property_name,
            value = value,
            scenario = scenario,
            missing_properties = missing_properties,
            **kwargs
        )
        printProgressBar(i + 1, len(filtered_property_sheet), 'Creating Properties', str(i))

    #  Build a final class and collection list for PLEXOS “close” operations
    # class_names = process_dataframes(filtered_setting_objects, filtered_objects, 'Class').drop_duplicates()
    # collection_names = process_dataframes(settings_membership_sheet,
    #                                       filtered_object_memberships,
    #                                       'Collection').drop_duplicates()

    # print(missing_memberships)

    return missing_objects, missing_memberships, missing_properties #class_names, collection_names, 


def process_base_model_task(db,
                            plexos_prompt_sheet: dict,
                            high_level_prompt: str,
                            model: str = 'gpt-4.1-mini') -> None:
    """
    Main pipeline to build a "base model" in PLEXOS using data from
    `plexos_prompt_sheet` and user instructions in `high_level_prompt`.

    Parameters
    ----------
    db : object
        PLEXOS database object or connection.
    plexos_prompt_sheet : dict of DataFrames
        Dictionary containing multiple sheets from the PLEXOS Model Builder file.
    high_level_prompt : str
        User instructions describing what to build.
    model : str, optional
        LLM model to use for country/category extraction, by default 'gpt-4.1-mini'.
    """
    # Filter the data according to the user prompt
    dict_filters = filter_data(plexos_prompt_sheet, high_level_prompt, model=model)

    # class_list, collection_list, 
    mis_objs, mis_mems, mis_props = _common_build_steps(db, dict_filters)

    # Finally, close the database object
    # pbf.close(db, collection_list, class_list, run_classes=True, run_collections=True)
    try:
        pdcm.close_db(db)
        print('Model Build Completed')
    except Exception as e:
        print(f'Error closing database: {e}')

    print(f"Missing Objects: {len(mis_objs)}, Missing Memberships: {len(mis_mems)}, Missing Properties: {len(mis_props)}")


def add_components(db,
                   plexos_prompt_sheet: dict,
                   high_level_prompt: str,
                   model: str = 'gpt-4.1-mini') -> None:
    """
    Adds new components (categories, objects, memberships, properties)
    to an already-loaded base PLEXOS model, derived from user instructions.

    Parameters
    ----------
    db : object
        The existing PLEXOS database object or connection.
    plexos_prompt_sheet : dict of DataFrames
        Dictionary containing data from the PLEXOS Model Builder workbook.
    high_level_prompt : str
        The user instructions describing which components to add.
    model : str, optional
        LLM for category/country extraction, by default 'gpt-4.1-mini'.
    """
    (filtered_object_cats,
     filtered_objects,
     carrier_property_cat_sheet,
     filtered_setting_object_cats,
     settings_objects,
     setting_membership_sheet,
     filtered_object_memberships) = filter_data(plexos_prompt_sheet,
                                                high_level_prompt,
                                                model=model)

    # Only build objects, categories, memberships, & properties
    class_list, collection_list = _common_build_steps(
        db,
        filtered_setting_object_cats,
        settings_objects,
        filtered_object_cats,
        filtered_objects,
        carrier_property_cat_sheet,
        setting_membership_sheet,
        filtered_object_memberships
    )

    # Don’t close if we want to keep adding more, but can close if needed:
    pbf.close(db, collection_list, class_list, run_classes=True, run_collections=True)


if __name__ == '__main__':
    # Example usage
    # user_prompt = stt()  # or 
    # user_prompt = input('Enter Your Prompt: ')
    user_prompt = 'build a solar model for croatia.'

    plexos_prompt_sheet = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "PLEXOS_inputs", "PLEXOS_Model_Builder_v2.xlsx"),
        sheet_name=None
    )

    # Create the request for task categorization
    context = "You are an assistant helping to build an energy model through the PLEXOS API."
    formatted_prompt = pt.plexos_prompt_sheet_builder(user_prompt)


    import os
    print(f"API Key env var set: {'OPENAI_API_KEY' in os.environ}")
    print(f"API Key available: {bool(oaic.API_KEY)}")
    print(f"API Key first 5 chars: {oaic.API_KEY[:5] if oaic.API_KEY else 'None'}")
    print(f"Using model: {'gpt-4.1-nano'}")  # Or whatever model is being used
    print(f"Function model parameter: {'gpt-4.1-nano'}")

    # Evaluate tasks from the user’s prompt
    #response = oaic.run_open_ai_ns(formatted_prompt, context)
    response = oaic.run_open_ai_ns(
        formatted_prompt, 
        context,
        model="gpt-4.1-nano",
        # Uncomment the next line if you want to try a different key
        # api_key=oaic.API_KEY  # Add this parameter if run_open_ai_ns supports it
    )



    if response is None:
        print("Error: OpenAI API returned None.")
        task_list = []  # Or provide a default value
    else:
        task_list = json.loads(response)
    # task_list = json.loads(oaic.run_open_ai_ns(formatted_prompt, context))



    print("Parsed tasks from prompt:", task_list)

    # Base model creation
    if "Base Model Task" in task_list:
        base_task = task_list["Base Model Task"]
        filename = initiate_file(base_task, model='gpt-4.1-mini')
        db = pbf.load_plexos_xml(source_file=filename, blank=True)

        # Build the base model
        process_base_model_task(db, plexos_prompt_sheet, base_task, model='gpt-4.1-mini')
    
    # Other tasks could be handled here (e.g., "Add Components", "Modify Parameters", etc.)
