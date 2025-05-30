import math
import sys 
from shutil import copyfile
import pandas as pd 
import datetime
import pyttsx3
import time
#import clr
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import os

# sys.path.append('utils')
# sys.path.append('functions\\Input')
# sys.path.append('functions\\plexos_functions')
# sys.path.append('functions')
# sys.path.append('C:\\TeraJoule\\AI Assistants\\Emil - AI Engineer')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from enums_as_df import extract_df as ead
from enums_as_df import main as eadm
import plexos_database_core_methods as pdcm
import clr
# import functions.plexos_database_core_methods as pdcm
# import enums_as_df

# Add references to PLEXOS and EEUTILITY assemblies
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(base_dir, 'PLEXOS 10.0 API'))

clr.AddReference('PLEXOS_NET.Core')
clr.AddReference('EEUTILITY')
clr.AddReference('EnergyExemplar.PLEXOS.Utility')

# you can import .NET modules just like you used to...
from PLEXOS_NET.Core import DatabaseCore
from EEUTILITY.Enums import *
from EnergyExemplar.PLEXOS.Utility.Enums import *
from System import Enum

enum_list, unique_enums = eadm()
not_passed = pd.DataFrame(columns = ['Membership','Property','Value Given'])

p2x_constraint_store = set()

blank = {'':''}
       
def load_plexos_xml(blank = False, source_file = False, new_copy = False, destination_file = 'copy', display_alerts = False):
    db = DatabaseCore()
    if blank == True:
        blank_file = os.path.join(base_dir, "PLEXOS_models", "new_blank_file.xml")
        copyfile(blank_file, source_file)
        db.Connection(source_file)        
    elif new_copy == True:
        if destination_file == 'copy':
            destination_file = source_file.replace('.xml','_copy.xml')
            copyfile(source_file, destination_file)
        else: 
            copyfile(source_file, destination_file)
        db.Connection(destination_file)
    else:
        db.Connection(source_file)   
    db.set_DisplayAlerts(display_alerts)
    return db

def clean_kwargs(**kwargs) -> dict:
    """
    Removes any key-value pairs where the value is NaN, None or '-'.

    Returns
    -------
    dict
        Cleaned dictionary with invalid values removed.
    """
    clean_args = {}
    for key, value in kwargs.items():
        if not (pd.isna(value) or value == '-'):
            clean_args[key] = value
    return clean_args

def add_properties_to_object(db, objclassidname, objname, properties_sheet_path, destination_plexos_manager, category, dict_name):
    properties_df = pd.read_excel(properties_sheet_path, sheet_name='Properties') # Load the 'Properties' sheet       
    relevant_properties = properties_df[properties_df['objclassid'] == objclassidname] # Filter properties for the given objclassid
    child_name = objname 
    for _, row in relevant_properties.iterrows():
        parentclassname = row['ParentClassName']
        if parentclassname == 'Power2X': 
            parent_object_name = str(objname)
            child_name = objname[0:2]
        else: 
            parent_object_name = row['Parent_Object_Name']            
        
        childclassname = row['ChildClassName']
        
        if childclassname == 'Constraint': child_name = objname[0:2]
        
        collection_name  = row['strCollectionName']
        objclassid = row['objclassid']
        
        collection_id_name =  f'{parentclassname}{collection_name}'
        
        originalclass = row['Original_class']
        collection_id = extract_enum(db, collection_id_name, 'Collection')
        print(collection_id)
        
        original_collection_id_name = '%s%s'%(parentclassname,originalclass)
        originalproperty = row['Property']
        # original_collection_id = extract_enum(original_collection_id_name, 'Collection')
        property_name = row['enum_id']
        from_unit = row['Units']
        to_unit = row.get('Convert')
        scenario = row['Scenario']
        if is_nan(scenario) == True: scenario = None
        value = 1
        
        datafile_name = row['Datafile']
        
        if is_nan(datafile_name) != True: 
            if objclassid == 'DataFile': 
                parts = dict_name.split('_', 1)
                result = parts[1] if len(parts) > 1 else ''
                datafile = f'H2\{category}\{objname}.csv'
            elif datafile_name[0:4] == 'file':
                datafile = datafile_name.replace('file_','')
                if '{objname}' in datafile: 
                    datafile = datafile.format(objname=objname)
                    datafile = datafile.replace('_dres','')
            elif datafile_name.split('_')[0] == 'dictionary':
                parts = datafile_name.split('_', 1)
                dictionary_key = parts[1]
                value = globals()[dictionary_key]['ES00solpv_dres']
            else:
                datafile = property_name
        else: 
            datafile = None
            value = row['Value']
        
        expression = None
        variable = None
        action = '='
        if property_name == 'Target' or property_name == 'Initial Volume':
            if childclassname == 'Gas Storage':
                datafile = 'Max Volume'
                action = '÷'
                variable = 'Divide'
        
        enum_id = destination_plexos_manager.PropertyName2EnumId(parentclassname, childclassname, collection_name, property_name)    
        
        add_property(destination_plexos_manager, collection_id, enum_id,   parent_object_name, child_name, value, 
                          scenario, datafile, 1, None, None, action, None, expression, period_type=PeriodEnum.Interval)
                
def Transfer_properties_from_object(db, objclassidname, objname, properties_sheet_path, destination_plexos_manager):
    if    objclassidname == 'Power2X':  collection_id_name = 'System%s'%(objclassidname)
    else: collection_id_name = 'System%ss'%(objclassidname)   
    
    collection_id = extract_enum(db, collection_id_name, 'Collection')
    properties_dicts = get_properties_table_to_dict(collection_id, "System", objname)
    for prop in properties_dicts:            
        property_name = prop['Property']
        value = prop['Value']
        
        if value == 'Yes': value = 1
        if value == 'No': value = 0
        date_from = None 
        date_to = None
        datafile = prop['Data_x0020_File']
        scenario  = None
        timeslice = prop['Timeslice']
        parentclassname = prop['Parent_x0020_Object']
        childobjectname = prop['Child_x0020_Object']
        strcollection =  prop['Collection']
        childclassname = strcollection[:-1]
        parent_object_name = parentclassname
        
        try: add_property(destination_plexos_manager, strcollection, childclassname, collection_id, parentclassname, parent_object_name, objname, property_name, value, scenario, datafile, 1, date_from, date_to, None, timeslice, 0, period_type=PeriodEnum.Interval)
        except: 
            property_name = property_name.replace(' Day','')
            add_property(destination_plexos_manager, strcollection, childclassname, collection_id, parentclassname, parent_object_name, objname, property_name, value, scenario, datafile, 1, date_from, date_to, None, timeslice, 0, period_type=PeriodEnum.Day)
   
def get_membership_parameters(db, membership_row, collection_name):
   parent_name = membership_row['parent_name']
   child_name = membership_row['child_name']
   child_class_name = membership_row['child_class_name']
   assignment = membership_row['Assignment']
   child_object_code = membership_row['child_object_code']
   collection_name = f"{parent_name}{child_name}"
   collection_id = extract_enum(db, collection_name, 'Collection')
   
def add_memberships_from_excel(db, objclassidname, objname,  xls, membershipcode, category, originalclassid, destination_plexos_manager, sub_node):
    if membershipcode != '-':
        membership_df = pd.read_excel(xls, sheet_name='Memberships')
        membership_child = pd.read_excel(xls, sheet_name='Membership_child_format').set_index('child_object_code')
        filtered_memberships = membership_df[membership_df['Interlinked Membership Code'] == membershipcode]
                    
        for _, membership_row in filtered_memberships.iterrows():
            parent_name, child_name, child_class_name, assignment, child_object_code, collection_name, collection_id = get_membership_parameters(membership_row, collection_name)
            
            if assignment == 'Extraction':
                old_parent_name = membership_row['parent_name_original']
                old_child_name = membership_row['child_name_original']
                original_collection_name = f"{old_parent_name}{old_child_name}"
                original_collection_id = extract_enum(db, original_collection_name, 'Collection')
                try: 
                    child_name = get_object_memberships(original_collection_id, objname)
                except:
                    if parent_name == 'GasPipeline': child_object = missing_pipelines(objname, child_name)
                add_membership(destination_plexos_manager, collection_id, objname, child_name, child_class_name, None)
                
            else: #if assignment != 'New Class':
                if assignment == 'Replace': 
                    remove_membership(destination_plexos_manager, collection_id, parent_name, objname) #Delete the membership
                    
                operation_1 = membership_child.loc[child_object_code, 'Operation_1']
                operation_2 = membership_child.loc[child_object_code, 'Operation_2']
                try:  reduce_val = int(membership_child.loc[child_object_code, 'Reduce Val'])
                except: reduce_val = 0
                split_1_str = membership_child.loc[child_object_code, 'Split_1_str']
                split_1_num = membership_child.loc[child_object_code, 'Split_1_num']
                split_2_str = membership_child.loc[child_object_code, 'Split_2_str']
                split_2_num = membership_child.loc[child_object_code, 'Split_2_num']
                addition = membership_child.loc[child_object_code, 'Addition']
                addspace = membership_child.loc[child_object_code, 'Add Space?']
                
                if category in alternate_categories: child_object = remove_words_from_strings(objname, ['-HIGH', '-LOW'])
                elif child_object_code > 0:
                    try:
                        if operation_1 == 'Reduce': child_object = objname[0:reduce_val]
                    except Exception as e:
                        print(f'Reduce {objname} to the first characters', e)
                    if operation_1 == 'objname': child_object = objname
                    if operation_1 == 'subitem': child_object = sub_node
                    if operation_1 == 'Split': child_object = objname.split(split_1_str)[int(split_1_num)]
                    if operation_1 == 'Trim': child_object = objname[3:]
                    if operation_1 == 'Add':  child_object = addition
                    if operation_2 == 'Add':  child_object = f"{child_object}{addition}"
                    if 'Z1' in category and 'Z1' not in objname: child_object = f"{child_object}Z1"
                    
                    if category == 'H2 Import':
                        if objname == 'UA => SK' and collection_name == 'GasPipelineGasNodeTo': child_object = 'SKh2E'
                    
                    if operation_2 == 'Split': 
                        split_value_2 = objname.split(split_2_str)[split_2_num] 
                        if space == 'Yes': child_object = f"{child_object} {split_value_2}"
                        else: child_object = f"{child_object} {split_value_2}"
                
                objname_check = objname.split(' ')[0]
                if objname_check in all_subnodes and parent_name == 'Power2X' and child_name == 'GasNodes':  
                    try: child_object = objname_check
                    except : child_object = objname 
                
                if objname == 'DKh2' and child_name == 'Nodes':  child_object = 'DKE1'
                if objname == 'DKh2 Bornholm' and child_name == 'Nodes': child_object = "DKBH"
                if objname == 'PL00ccgh': child_object = "PLh2N"
                if objname == 'DKh2 Bornholm' and child_name == 'Nodes': child_object = "DKBH"
                if len(objname) > 4 and parent_name == 'GasDemand':  child_object = objname 
                
                if objname in smr and parent_name == 'GasField': child_object = smr[objname]
                    
                if parent_name == 'Emission':  
                    parent_obj_name = child_object
                    child_object = objname
                else:
                    parent_obj_name = objname
                add_membership(destination_plexos_manager, collection_id = collection_id, parent_name = parent_obj_name, child_name = child_object, child_class_name = child_class_name, sub_item = sub_node)
                
def get_properties_table_to_dict(db, collection_id, parent_name, child_name):
    recordset = db.GetPropertiesTable(collection_id, parent_name, child_name)
    properties_list = []
    field_names = [field.Name for field in recordset.Fields]
    if not recordset.BOF:
        recordset.MoveFirst()
    while not recordset.EOF:
        record_dict = {field_name: recordset.Fields[field_name].Value for field_name in field_names}
        properties_list.append(record_dict)
        recordset.MoveNext()
    return properties_list

def get_object_memberships(db, collection_id, objname):
    child_members = db.GetChildMembers(collection_id, objname)
    parent_members = db.GetParentMembers(collection_id, objname)
    return child_members[0]
    
def missing_pipelines( line_name, child_class_name):
    if child_class_name == 'GasNodeFrom':  child_name = line_name.split(' => ')[0]
    if child_class_name == 'GasNodeTo':  child_name = line_name.split(' => ')[1]
    return child_name
   
def add_object_to_plexos(db,
                         df: pd.DataFrame,
                         idx: int,
                         missing_objects: dict) -> None:
    """
    Adds an object to the PLEXOS database.

    Parameters
    ----------
    db : object
        PLEXOS database object or connection.
    df : pd.DataFrame
        DataFrame containing at least 'Name', 'Class', and 'Category' columns.
    idx : int
        Row index in df to create the object from.
    """
    
    object_name = str(df.loc[idx, 'Name']) if not isinstance(df.loc[idx, 'Name'], str) else df.loc[idx, 'Name']
    objclassname = df.loc[idx, 'Class'].replace(' ','')
    category = df.loc[idx, 'Category']
    objclassid = extract_enum(db, objclassname, 'Class')
    objs = pdcm.get_objects(db, objclassid)
    try:
        if objs is None or object_name not in objs:
            pdcm.add_object(db, strName = object_name, nClassId = objclassid, strCategory = category)
    except Exception as e:
        missing_objects[object_name] = f"Error: {e}"

def create_memberships(db,
                       object_name: str,
                       membership_df: pd.DataFrame,
                       missing_memberships: dict) -> None:
    """
    Creates membership relationships in the PLEXOS database for a specified object.

    Parameters
    ----------
    db : object
        The PLEXOS database instance or connection.
    object_name : str
        The parent object name to link memberships from.
    membership_df : pd.DataFrame
        DataFrame containing membership relationships.
    missing_memberships : dict
        Dictionary to log memberships that fail or are not found.
    """
    obj_df = membership_df.loc[membership_df['Parent Name'] == object_name]

    if obj_df.empty:
        missing_memberships[object_name] = f'Error: Not found in membership database, skipped!'
        return
    
    for membership in obj_df.index:
        child_name = str(obj_df.loc[membership, 'Child Name']) if not isinstance(obj_df.loc[membership, 'Child Name'], str) else obj_df.loc[membership, 'Child Name']
        collection_name = obj_df.loc[membership, 'Collection'].replace('.','').replace(' ','')
        collection_id = extract_enum(db, collection_name, 'Collection')
        try:
            db.GetMembershipID(nCollectionId = collection_id, 
                                    strParent = object_name, 
                                    strChild = child_name)
            # print(f"Membership already exists: {object_name} -> {child_name} -> {collection_name}")
        except:
            try:
                # Add the missing membership
                pdcm.add_membership(db = db,
                                nCollectionId = collection_id, 
                                strParent = object_name, 
                                strChild=child_name)
                
                # Verify that it was successfully added
                db.GetMembershipID(nCollectionId = collection_id, 
                                    strParent = object_name, 
                                    strChild = child_name)
                # print(f"Successfully added membership: {object_name} -> {child_name}")
            except Exception as e:
                missing_memberships[object_name] = f'Error with child {child_name} and collection {collection_name}: \n {str(e)}'
               
def remove_object(db, objclassid, objname):
    db.RemoveObject(strName=objname, nClassId=objclassid)

def remove_category(db, catid, catname):
    cats = db.GetCategories(catid)
    if catname in cats:
        db.RemoveCategory(strCategory=catname, nClassId=catid, bRemoveObjects=True)

def remove_membership(db, collection_id, child_name, objname):
    db.RemoveMembership(collection_id, child_name, objname)

def get_objects_in_category(db, class_id, catname):
    return db.GetObjectsInCategory(class_id, catname)

def get_categories(db, class_enum_id): 
    return db.GetCategories(class_enum_id)

def get_objects(db, catid):
    return db.GetObjects(nClassId=catid)

def add_property_to_plexos(db,
                           filtered_object_cats: pd.DataFrame,
                           category: str,
                           objclassname: str,
                           carrier_property_cat_sheet: pd.DataFrame,
                           object_name: str,
                           filtered_object_memberships: pd.DataFrame,
                           missing_properties: dict) -> None:
    """
    Adds a set of default properties to an object in the PLEXOS database.

    Parameters
    ----------
    db : object
        PLEXOS database object or connection.
    filtered_object_cats : pd.DataFrame
        Filtered object categories, with MultiIndex on (Object_Category, Class).
    category : str
        The 'Category' from the row (e.g., 'Solar PV').
    objclassname : str
        The 'Class' from the row (e.g., 'Generator').
    carrier_property_cat_sheet : pd.DataFrame
        A sheet describing property types and their default values.
    object_name : str
        The target object to which these properties are assigned.
    """
    key = (category, objclassname)
    if key in filtered_object_cats.index:
        prop_type = filtered_object_cats.loc[key, 'Property_Type'][0]
    else:
        print(f"⚠️ Warning: Key {key} not found in DataFrame!")
        prop_type = None  # Handle missing case appropriately
        
    if not prop_type or str(prop_type).strip() == "":
        # prop_type is None or blank → use fallback filter
        if not carrier_property_cat_sheet[carrier_property_cat_sheet['Property_type'] == objclassname].empty:
            property_list = carrier_property_cat_sheet[carrier_property_cat_sheet['Property_type'] == objclassname].fillna(pd.NA)
        elif not carrier_property_cat_sheet[carrier_property_cat_sheet['ChildClassName'] == objclassname].empty:
            property_list = carrier_property_cat_sheet[carrier_property_cat_sheet['ChildClassName'] == objclassname].fillna(pd.NA)
    else:
        # prop_type is valid → filter by Property_type
        property_list = carrier_property_cat_sheet[carrier_property_cat_sheet['Property_type'] == prop_type].fillna(pd.NA)
    
    property_list = property_list.where(pd.notna(property_list), None)

    for x in property_list.index:
        if (category, objclassname) not in filtered_object_cats.index:
            continue # No property to add if not in the index
        strcollection = property_list.loc[x, 'strCollectionName']
        parent_class_name = property_list.loc[x, 'ParentClassName']
        parent_object_name = property_list.loc[x, 'Parent_Object_Name'] or ''
        collection_name = f"{parent_class_name}{strcollection}"

        try:
            property_name = property_list.loc[x, 'Property_name']
        except Exception as e:
            property_name = property_list.loc[x, 'ChildClassName']

        if parent_object_name == 'Object_name':
            parent_object_name = object_name  # If placeholder

        # Finding Child Object
        if parent_object_name == 'System':
            child_name = object_name
        else:
            child_name = filtered_object_memberships.loc[
                (filtered_object_memberships['Parent Name'] == parent_object_name) &
                (filtered_object_memberships['Collection'].str.replace(' ', '').str.replace('.', '') == collection_name),
                'Child Name'
            ].iloc[0] if not filtered_object_memberships.empty else None

        value = property_list.loc[x, 'Value_default']

        scenario = str(property_list.loc[x, 'Scenario']) if 'Scenario' in property_list.columns else None
        scenario = scenario if pd.notna(scenario) else None  # Replace NaN with None
        band_id = property_list.loc[x, 'Band']
        date_from = property_list.loc[x, 'Date From']
        date_to = property_list.loc[x, 'Date To']
        variable = property_list.loc[x, 'Expression']
        pattern = property_list.loc[x, 'Timeslice']
        action = property_list.loc[x, 'Action']

        # Clean up optional kwargs
        kwargs = clean_kwargs(datafile=None,
                              band_id=band_id,
                              date_from=date_from,
                              date_to=date_to,
                              variable=variable,
                              pattern=pattern,
                              action=action)

        add_property(
            db,
            collection_name,
            parent_object_name,
            child_name,
            property_name,
            value,
            scenario,
            missing_properties,
            **kwargs
        )

def add_property(db, collection_name, parent_object_name, child_name, property_name, value, scenario, missing_properties,
                  datafile = None, band_id = 1, date_from=None, date_to=None, variable = None, pattern = None, action = None, period_type=PeriodEnum.Interval):
    """
    Adds a property using PLEXOS API and collects any failures.
    Returns a list of dicts describing any missing or failed operations.
    """

    property_name = property_name.replace(' ', '').replace('.', '')
    property_id = extract_enum(db, f'{collection_name}.{property_name}', 'Properties')

    try:
        collection_id = extract_enum(db, collection_name, 'Collection')
        membership_id = pdcm.get_membership_id(db, collection_id, parent_object_name, child_name)
    except Exception as e:
        missing_properties[parent_object_name] = f"Error with child {child_name} and {collection_id}: Failed to retrieve Membership ID: {str(e)}"
        return
    
    try:
        pdcm.add_property(db,
                        membership_id,
                        property_id,
                        band_id, value, date_from, date_to, variable,
                        datafile, pattern, scenario, action, period_type
                        )
    except KeyError as e:
        missing_properties[parent_object_name] = f"Error with child {child_name} and {collection_id}: KeyError: Missing key {str(e)} in collections or properties."
        
    except AttributeError as e:
        missing_properties[parent_object_name] = f"Error with child {child_name} and {collection_id}: AttributeError: Database object might be missing a method - {str(e)}"
        
    except Exception as e:
        missing_properties[parent_object_name] = f"Error while adding property:  {str(e)}"

def iterations_and_updates(): #rag_system
    pass

def remove_property(db, collection_id, parent_name, child_name, property_name, date_from=None, date_to=None, band_id=1, period_type=PeriodEnum.Interval):
    membership_id = db.GetMembershipID(collection_id, parent_name, child_name)
    enum_id = db.PropertyName2EnumId(parent_name, child_name, collection_id, property_name)
    print(enum_id)
    db.RemoveProperty(MembershipId=membership_id, EnumId=enum_id, BandId=band_id, DateFrom=date_from, DateTo=date_to, PeriodTypeId=period_type)

def get_properties_table_to_dict(db, collection_id, parent_name=None, child_name=None):
    recordset = db.GetPropertiesTable(collection_id, parent_name, child_name)
    properties_list = []
    field_names = [field.Name for field in recordset.Fields]
    if not recordset.BOF:
        recordset.MoveFirst()
    while not recordset.EOF:
        record_dict = {field_name: recordset.Fields[field_name].Value for field_name in field_names}
        properties_list.append(record_dict)
        recordset.MoveNext()
    return properties_list
    
def find_duplicate_objects_in_classes(db, child_class_name):
    try:
        child_class_name_id = extract_enum(db, child_class_name, 'Class') #extract the enum id from the classname
        objects = db.GetObjects(child_class_name_id) #extract the object from the class
        python_list = []
        for y in objects: # Loop through each string in the .NET array and append it to the Python list
            python_list.append(y)
        object_counts = Counter(python_list)
        duplicates = [obj for obj, count in object_counts.items() if count > 1] # Find strings with more than 1 occurrence
        empty_count = python_list.count("")
        print(f"Found {duplicates} duplicates in {child_class_name} Class and {empty_count} empty cells")
    except: 
        print(f'Class {child_class_name} not found in model')
        
def find_duplicate_memberships_in_classes(db, collection_name):
    collection_name_id = extract_enum(db, collection_name, 'Collection') #extract the enum id from the classname
    
    objects = db.GetMemberships(collection_name_id) #extract the object from the class
    python_list = []
    try:
        
        for y in objects: # Loop through each string in the .NET array and append it to the Python list
            python_list.append(y)
        object_counts = Counter(python_list)
        duplicates = [obj for obj, count in object_counts.items() if count > 1] # Find strings with more than 1 occurrence
        empty_count = python_list.count("")
        print(f"Found {duplicates} duplicates in {collection_name} Class and {empty_count} empty cells")
    except:
        print(f'{collection_name} not found')
        
def manual_property_enum_id(collection_name, property_name, parent_class_name): # A fill in until property ID can be correctly extracted. Temp solution.
    collection_name = collection_name.replace(' ','')
    property_name = property_name.replace(' ','')
    property_name = property_name.replace('&','')
    column_val = f'{parent_class_name}{collection_name}Enum'
    x = ead(enum_list, 'Enum', column_val, 'Name', property_name)['ID'].sum()
    if x == 0:
        not_passed.loc[len(not_passed)] = [column_val, property_name, x]
    return x

def close(db, df_collections = '', df_class = '', run_classes = False, run_collections = False):
    try:
        db.Close()
        print('Model Build Complete')
    except: 
        if run_classes == True:
            for class_ in df_class:
                find_duplicate_objects_in_classes(db, class_)
        
        if run_collections == True:
            for collection in df_collections:
                find_duplicate_memberships_in_classes(db, collection)        
        print('Model Build Failed')    

def add_category(db, class_id, catname):
    db.AddCategory(class_id, catname)

def update_plexos_model_from_excel(db, source_plexos_manager, destination_plexos_manager, xls):
    object_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith('Object_')] #find all sheet with starting with object and loop file
    for sheet_index, sheet in enumerate(object_sheets): #enumerate all sheet with starting with object and loop file
        df = pd.read_excel(xls, sheet_name=sheet) # open datafile
        for _, row in df.iterrows(): #good through file
            class_change_type = row['Class Change Type'] #determine how how the object in a catgegory will be handled e.g. Transfered between model, add based on another class or add new category to class
            obj_class_name = row['objclassid']
            objclassid = extract_enum(db, row['objclassid'], 'Class') #get the class id, the plexos api works with id's rather name class names
            objclassidname = row['objclassid'] #also get the classes name
            membershipcode = row['Interlinked Membership Code'] #the membership codes determines which memberships will be added
            originalclassid = extract_enum(db, row['Original_class'], 'Class' ) #this is the original class, if the objclassid is different the code will create object in the new class
            category = row['Category'] # get the category name
            copy_from_class = row['Copy from Class']
                        
            if class_change_type == 'New Category': #this create object in a new class based on a category which exists in an exsiting class
                add_category(destination_plexos_manager, class_id=objclassid, catname=str(category)) # add the category name to the new class
                it = 1
                dict_name = row['new_class_python_dict']
                dictionary = eval(dict_name)
                
                if copy_from_class != '-':
                    class_enum = extract_enum(db, copy_from_class, 'Class')
                    objects_in_class = get_objects(destination_plexos_manager, class_enum)
                    class_enum_set = set(objects_in_class)
                    country_names = {element[:2] for element in class_enum_set}
                    for country in country_names:
                        add_object(destination_plexos_manager, objclassid=objclassid, objname=country, category=category)                        
                    
                    for objname in objects_in_class:
                        country = objname[0:2]
                        add_memberships_from_excel(destination_plexos_manager, objclassidname, objname, xls, membershipcode, category, originalclassid, destination_plexos_manager, country)
                        add_properties_to_object(source_plexos_manager, objclassidname, objname, xls, destination_plexos_manager, category, dict_name)
                else:
                    for node, value in dictionary.items():
                        printProgressBar(it, len(dictionary) ,category, '%s %s'%(objclassidname, node), decimals = 1, length = 20, fill = '█')
                        it = it + 1
                        if node in p_to_h2Z2_dict and 'Z2' in category: #if a country has multiple nodes in zone 2 this will be triggered
                            node_list = extract_dict_items(p_to_h2Z2_dict, node) # extract all subnodes
                            if isinstance(node_list, tuple): #if more than 1 subnode i.e. a tuple, which should be the case for all
                                for sub_item in node_list: #for each 
                                    objname = f'{sub_item}' # set the object name to e.g BEhZ Z2
                                    add_object(destination_plexos_manager, objclassid=objclassid, objname=objname, category=category) #add new object
                                    if isinstance(membershipcode, (int, float)):  destination_plexos_manager.add_memberships_from_excel(objclassidname, objname, xls, membershipcode, category, originalclassid, destination_plexos_manager, sub_item) #Add Membership
                                    add_properties_to_object(source_plexos_manager, objclassidname, objname, xls, destination_plexos_manager, category, dict_name) #Add property
                            else:
                                objname = p_to_h2Z2_dict[node] # set the object name to e.g BEhZ Z2
                                add_object(destination_plexos_manager, objclassid=objclassid, objname=objname, category=category) #add new object
                                if isinstance(membershipcode, (int, float)): destination_plexos_manager.add_memberships_from_excel(objclassidname, objname, xls, membershipcode, category, originalclassid, destination_plexos_manager, sub_item) #Add Membership
                                add_properties_to_object(source_plexos_manager, objclassidname, objname, xls, destination_plexos_manager, category, dict_name) #Add property
                       
                        elif node in p_to_h2Z1_dict  and 'Z1' in category: #if a country has multiple nodes in zone 1 this will be triggered
                            if 'Z1' in category:
                                node_list = extract_dict_items(p_to_h2Z1_dict, node)
                                if isinstance(node_list, tuple):
                                    for sub_item in node_list:
                                        objname = f'{sub_item} Z1'
                                        add_object(destination_plexos_manager, objclassid=objclassid, objname=objname, category=category)
                                        if isinstance(membershipcode, (int, float)): destination_plexos_manager.add_memberships_from_excel(objclassidname, objname, xls, membershipcode, category, originalclassid, destination_plexos_manager, sub_item)
                                        add_properties_to_object(source_plexos_manager, objclassidname, objname, xls, destination_plexos_manager, category, dict_name)
                        else: 
                            catcode = category[-2:]
                            if obj_class_name == 'Power2X': objname = f'{node} {catcode}'
                            else : objname = node
                            add_object(destination_plexos_manager, objclassid=objclassid, objname=objname, category=category)
                            if isinstance(membershipcode, (int, float)): destination_plexos_manager.add_memberships_from_excel(objclassidname, objname, xls, membershipcode, category, originalclassid, destination_plexos_manager, None)
                            add_properties_to_object(objclassidname, objname, xls, destination_plexos_manager, category, dict_name)
            else:
                if class_change_type == 'Remove': remove_category(destination_plexos_manager, catid=originalclassid, catname=category)
                elif class_change_type == 'Update Membership': original_objects = get_objects_in_category(destination_plexos_manager, catid=originalclassid, catname=category)
                elif class_change_type == 'Copy':
                    original_category = row['Copy From Category'] #the membership codes determines which memberships will be added
                    original_objects = get_objects_in_category(destination_plexos_manager, catid = originalclassid, catname = original_category)
                    add_category(destination_plexos_manager, objclassid, category)                                                                      
                else: original_objects = get_objects_in_category(source_plexos_manager, originalclassid, category)
                
                add_category(destination_plexos_manager, objclassid, str(category))
                it = 1
                for objname in original_objects: 
                    printProgressBar(it, len(original_objects) ,category, '%s %s'%(objclassidname, objname), decimals = 1, length = 20, fill = '█')
                    it = it + 1
                    add_object(destination_plexos_manager, objclassid=objclassid, objname=objname, category=category)
                    add_memberships_from_excel(objclassidname, objname, xls, membershipcode, category, originalclassid, destination_plexos_manager, None)                        
                    if class_change_type == 'Transfer': Transfer_properties_from_object(objclassidname, objname, xls, destination_plexos_manager)
                    elif class_change_type != 'Update Membership': add_properties_to_object(source_plexos_manager, objclassidname, objname, xls, destination_plexos_manager, category, dict_name)
    
def remove_categories_from_excel(db, plexos_manager, xls):
    df = pd.read_excel(xls, sheet_name='Remove_Objects')
    for _, row in tqdm(df.iterrows(), desc="Processing sheets"):
        class_id = extract_enum(db, row['Class'], 'Class')  # Convert class name to ClassEnum, ensure this matches your enum structure
        category_name = row['Category']
        remove_category(plexos_manager, catid=class_id, catname=category_name)
    print("Category removal complete.")

def remove_words_from_strings(string, words_to_remove):
    for word in words_to_remove:
        string = string.replace(f"{word} ", " ")  # Ensures words are removed with surrounding spaces to avoid merging words
    return string

def extract_enum(db, enum_name, enum_type):
    '''
    Return an enum value (Int32) for PLEXOS API functions.
    Parameters
    ----------
    db : object
        PLEXOS database connection.
    enum_name: string
        Name of the enum
    type_ : string
        Type of the enum (e.g. Class, Collection, etc.)
    Returns
    -------
    Int32
        Integer value for final enum closure steps.
    '''
    try: 
        if enum_type == 'Class': 
            classes = db.FetchAllClassIds()
            return classes[enum_name]
        elif enum_type == 'Collection': 
            collections = db.FetchAllCollectionIds()
            return collections[enum_name]
        elif enum_type == 'Properties': 
            properties = db.FetchAllPropertyEnums()
            return properties[enum_name]
        elif enum_type == 'Attributes':
            attributes = db.FetchAllAttributeEnums()
            return attributes[enum_name]
        elif enum_type == 'Period':
            return Enum.Parse(PeriodEnum, enum_name)
    except Exception as e:
        print(f'Enum {enum_name} not found: {e}') 

def convert_units(value, from_unit, to_unit):
    if from_unit == "EUR/MWh" and to_unit == "EUR/GJ":
        return float(value) / 3.6  # Assuming 1 MWh = 3.6 GJ
    elif from_unit == "MW" and to_unit == "TJ":
        return float(value) * 3.6  # Assuming 1 MW = 3.6 TJ (for a specific time period)
    else:
        return float(value)  # No conversion needed

def printProgressBar (iteration, total, prefix = '-', suffix = '', decimals = 1, length = 20, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print()

def extract_dict_items(df, key):
    if key in df:
        value = df[key]
        return value
        if isinstance(value, tuple):
            for sub_item in value:
                return sub_item
        else:
            return value

def is_nan(value):
    try:
        numeric_value = float(value) # Attempt to convert the value to a float
        return math.isnan(numeric_value)# Check if the converted float is nan
    except (ValueError, TypeError):
        return False

def export_sheets_to_csv(file_path, output_directory):
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        if sheet_name == 'End': break
        # print('Exporting',sheet_name)
        df = pd.read_excel(xls, sheet_name=sheet_name)
        category = sheet_name.split('_')[0]
        property_name = sheet_name.split('_')[1]
        output_dir = f'{output_directory}\{category}'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                    
        csv_file_name = f"{output_dir}\{property_name}.csv"
        df.to_csv(csv_file_name, index=False)

if __name__ == '__main__':
    transferfile = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DGM\TYNDP_2024_DGM_Model.xml'
    source_file = r'C:\Users\ENTSOE\OneDrive\PLEXOS\Scenario Building 2024\2024 Scenarios\TY2024 ENTSO-G SB Model Sharing\Model_CY_NT30_PEMMDB_2.5_v40_gasalingned\NT30_CY (9.100 R01).xml'
    destination_file = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\TYNDP_2024_DHEM_Model_2030_V5.xml'

    excel_path = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\Model Prep\DGM to DHEM Model Strucuture NT.xlsx' 
    datafile_input = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\H2\DHEM_Model_H2_Inputs.xlsx'
    output_directory = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\H2'
    xls = pd.ExcelFile(excel_path)
    
    manager = load_plexos_xml(transferfile, new_copy = False)
    manager_NT = load_plexos_xml(source_file, new_copy = True, destination_file = destination_file)
    
    update_plexos_model_from_excel(manager, manager_NT, xls)
    remove_categories_from_excel(manager_NT, xls)
    
    export_sheets_to_csv(datafile_input, output_directory)
    
    manager_NT.close()

# DHEM_Reporting.main()

