# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:56:03 2024

@author: ENTSOE
"""
import json 
from datetime import datetime, timedelta

def plexos_prompt_sheet_builder(user_prompt):
    formatted_prompt = f"""Please create a task list from the prompt, categorized into 'Base Model Task' and 'Modifications'.
    Base model tasks: High level build of a system typically based on a data. Will be used if a model is being build from scratch, which can be determined from the prompt.
    ONLY add task to the category if it is clear that a new model needs to be built. Typicall the word build will be present in the prompt.
    Modification: These are modification of a model. this could be additional of classes, categories, child object, memberships, properties attributes.
    
    A prompt can include the combination of base model tasks and modification, only base model task or only modification
    
    Example:
    Prompt: Build an EU electricity and hydrogen model but exclude Iceland. Increase the CO2 price by 10% and remove the ramping constraint from gas generators in Germany.
    Tasks:
        Base Model Task: Build an EU electricity and hydrogen model but exclude Iceland.
        Modifications:
            1. Increase the CO2 price by 10%.
            2. Remove the ramping constraint from gas generators in Germany.
    
    Do not include any reasoning in your response. Only return the tasks in the following as a Dictionary:
    
    {{
      "Base Model Task": "Build an EU electricity and hydrogen model but exclude Iceland.", 
      "Modifications": [
          "Increase the CO2 price by 10%.", 
          "Remove the ramping constraint from gas generators in Germany."
      ]
    }}
    
    IT IS CRITICAL the prompt meets the template guidelines before it is returned! NO EXTRA TEXT!
    
    # Your prompt to be processed
    Prompt: {user_prompt}"""
    return formatted_prompt

explanation = """"""

def plexos_prompt_sheet_categorization(request, categories, search_term):
    prompt = f"""
    Format the response as a plain JSON object. Do not include markdown, backticks, or any additional text. 
    Here are the details you need to consider: 
    Input request: '{request}'. Search term to extract: '{search_term}'. Available categories: {categories}. 
    Return a JSON object with the appropriate category key and value. 
    For example, if the applicable category is Electricity, respond with ['0':'Electricity']. 
    If the applicable category is Water, respond with ['1':'Water']. 
    Ensure the format is consistent regardless of the number of items in the JSON object.")
    You can only respond in json. IT IS CRITICAL the prompt meets the template guidelines before it is returned! NO EXTRA TEXT!
    {explanation}
    """
    return prompt
    
def general_task_flow_creation(user_prompt):
    
    formatted_prompt = f"""Please create a task list from the prompt, categorized into 'Tasks'.
    Do not include any reasoning in your response. Only return the tasks in the following as a Dictionary:
    Prompt Exmaple: Run an expansion model, then run a dispatch model. Give me a chart of the Electrolyser capacities built.    
    Output Example:
    {{
      "Task 1": "Run an expansion model.", 
      "Task 2": "Format expansion model output for next stage.", 
      "Task 3": "Run a dispatch model.", 
      "Task 4": "Extract data from dispatch model output.", 
      "Task 5": "Create chart of Electrolyser capacities built."
    }}
    # Your prompt to be processed
    Prompt: {user_prompt}
    IT IS CRITICAL the prompt meets the template guidelines before it is returned! NO EXTRA TEXT!
    {explanation}

    """
    
    return formatted_prompt

def extract_countries(countries_embed, high_level_prompt):
    formatted_prompt = formatted_prompt = f"""
                                            Here is the request: {high_level_prompt}. 
                                            There may or may not be a predicted from an embedding model: {countries_embed}. Consider the decision made if data is available.                                             
                                            Ensure to format each country with its correct two-letter ISO code in a JSON dictionary, using the following structure: {{Country_Name: ISO_Code}}. For instance:
                                                {{"France": "FR", "Germany": "DE", "United Kingdom": "UK"}}
                                            In case of Greece being in the response, use the ISO code GR accordingly. Ensure your output adheres to this single dictionary format within a JSON object: 
                                            All data must be on 1 level key:value
                                            {explanation}

                                                """
    return formatted_prompt

def create_invoice(invoice_no, days_worked, amount_in_words, price = 900):
    today_date = datetime.now()
    due_date = today_date + timedelta(days=14)
    total = days_worked * price
    invoice_format = {
                        "invoice_no": invoice_no,
                        "date": today_date,
                        "due_date": due_date,
                        "addressee": {
                            "name": "ENTSOG AISBL",
                            "address": "Avenue de Cortenbergh 100",
                            "city": "1000 - Brussels",
                            "country": "Belgium"
                        },
                        #add IBAN: LT03 3250 0045 4383 9788 and BIC code: REVOLT21 under bank details
                        "Bank_details": {
                            "IBAN": "LT03 3250 0045 4383 9788", 
                            "BIC code": "REVOLT21"},
                        "sender": {
                            "name": "Terajoule LTD",
                            "address": "3314 Lakeside Park, Newbridge",
                            "city": "Kildare",
                            "country": "Ireland",
                            "vat_no": "IE04090642NH",
                            "bank_details": {
                            "iban": "LT03 3250 0045 4383 9788",
                            "bic_code": "REVOLT21"
                            }
                        },
                        "items": [
                            {
                            "description": "days",
                            "unit": "days",
                            "quantity": days_worked,
                            "price": price,
                            "sum": total
                            }
                        ],
                        
                        "vat_customer_vat_no": "BE0822 653 040",
                        "totals": {
                            "sum": total,
                            "vat": {
                            "percentage": 0,
                            "amount": 0
                            },
                            "total_eur": total
                        },
                        "in_words": amount_in_words,
                        "terms": "To be paid in 20 days"
                        }
    return invoice_format
    
def invoice_header(work_period):
    header_format = {
                        "project": {
                            "title": "Project Timesheet",
                            "description": "Activities Related to the tasks and deliverables as described in the Consultancy Contract"
                        },
                        "consultant": {
                            "name": "Dante Powell",
                            "position": "Innovation Manager",
                            "work_period": work_period,
                            "contact_number": "+32473810490",
                            "contact_email": "Dantepowell@terajouleenergy.com"
                        },
    }
    return header_format

def create_invoice_activity_summary(calendar_extrait, date):
    activity_summary_format = [
        {"date": "1991-06-28", "day": "Wednesday", "days": 1, "task": "Scenario Meeting, DHEM Model, Core Team, SRG", "approver": "TVDG"},
        {"date": "1991-06-29", "day": "Thursday", "days": 0, "task": "Annual Leave", "approver": ""}
            ]
    
    activity_summary_format_json = json.dumps(activity_summary_format, indent=4)
    
    prompt = f"""Create an activity summary for this date: {date}.
                Here is an extract from my calendar: {calendar_extrait}. 
                Here is an example of the activity summary format: {activity_summary_format_json}. 
                Strictly conform to this format, but discard all the data as it is not relevant. Replace the 'date' (found in the key), 
                'day' (found in the key), 'days', 'task' using the date: {date} and activities from the calendar extract.
                Do not add anything not work-related such as doctor's appointments
                Do not add dinners
                Do not all 
                Do not add other life admin.
                Please proceed in creating the summary sheet using the activity summary format. 
                Think about it step by step, going day by day.
                Data is a binary dtype. If any data has reference to holidays or annual leave, days = 0, task = 'Annual leave'. else days = 1
                The output should be 1 dictionary entry. If there are multiple items in the dictionary summarize the tasks to 3 words maximum each!.
                Ignore any personal admin such as doctor appointmetns or life admin. Ignore the meeting 'PLEXOS API'. ignore 'Pay Otis - Invoice' or any other payments.
                Return the output as a JSON NOT list, with no other text as the output will be used directly in the next process.
            """
    return prompt

def invoice_footer(work_days):
    footer_format = {
                        "total_days": work_days,
                        "approval": {
                        "approved": "",
                        "date": "",
                        "signature": ""
                                    }
                    }
    return footer_format

def days_worked(invoice_summary): 
    return f"""based on this invoice summary: {invoice_summary} how many days did I work?. 
        If there is any activity undertaken in a day, it is classed as a day. 
        If annual leave is present in the day, it is not classed as a day worked.
        Think about it carefully as it has real world impacts. Show your logic and working
        """
        
def upgrade_prompt(request, response, checkpoints = 'Formaating'): 
    return f"""
        here was the request: {request}.
        here was the response: {response}. 
        please check for conformance and quality of the response.
        Here is a list of parameters that have been requested to pay particular detail to: {checkpoints}
        Please return an improved response as a json with no other text or punctation in the response.
    
    """
