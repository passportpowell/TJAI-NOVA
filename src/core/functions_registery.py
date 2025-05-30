"""
functions_registery.py
This module defines the functions that can be called by our AI agents.
It includes functions for model building, running simulations, analyzing results,
report writing, and a wrapper that integrates Emil's original energy modeling logic.

Using CSV-based function maps for flexibility and extensibility.
"""

import os
import sys
import datetime
import logging
import json
import xml.etree.ElementTree as ET
from shutil import copyfile
from utils.csv_function_mapper import FunctionMapLoader
from utils.function_logger import log_function_call
from core.knowledge_base import KnowledgeBase

# Check if PLEXOS is available
PLEXOS_AVAILABLE = False
try:
    import clr
    # Try to locate the PLEXOS path
    plexos_path = 'E:\\Program Files\\Energy Exemplar\\PLEXOS 10.0 API'
    if os.path.exists(plexos_path):
        sys.path.append(plexos_path)
    
    # Only try to add references if clr was successfully imported
    clr.AddReference('PLEXOS_NET.Core')
    clr.AddReference('EEUTILITY')
    clr.AddReference('EnergyExemplar.PLEXOS.Utility')
    from System import Enum
    from EnergyExemplar.PLEXOS.Utility.Enums import *
    from EEUTILITY.Enums import *
    from PLEXOS_NET.Core import DatabaseCore
    PLEXOS_AVAILABLE = True
    print("PLEXOS libraries loaded successfully")
except Exception as e:
    PLEXOS_AVAILABLE = False
    print(f"PLEXOS libraries not available - will use simple XML instead. Error: {e}")

if os.environ.get("VERBOSE_MODE", "1") == "1":
    print("PLEXOS availability: " + str(PLEXOS_AVAILABLE))

# ----------------------------------------------------------------------
# Global Constants
# ----------------------------------------------------------------------

GENERATION_TYPES = {
    "wind": ["Onshore Wind", "Onshore Wind Expansion", "Offshore Wind Radial"],
    "solar": ["Solar PV", "Solar PV Expansion", "Solar Thermal Expansion", 
              "Rooftop Solar Tertiary", "Rooftop Tertiary Solar Expansion"],
    "hydro": ["RoR and Pondage", "Pump Storage - closed loop"],
    "thermal": ["Hard coal", "Heavy oil"],
    "bio": ["Bio Fuels"],
    "other": ["Other RES", "DSR Industry"]
}

LOCATIONS = [
    # EU members
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", 
    "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
    "Slovenia", "Spain", "Sweden",
    
    # Non-EU European countries
    "Albania", "Andorra", "Armenia", "Azerbaijan", "Belarus", 
    "Bosnia", "Bosnia and Herzegovina", "Georgia", "Iceland", 
    "Kosovo", "Liechtenstein", "Moldova", "Monaco", "Montenegro", 
    "North Macedonia", "Norway", "Russia", "San Marino", "Serbia", 
    "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City",
    
    # Common abbreviations and alternate names
    "UK", "Great Britain", "Czechia", "Holland"
]

# ----------------------------------------------------------------------
# Function Definitions
# ----------------------------------------------------------------------

@log_function_call
def build_plexos_model(kb: KnowledgeBase, location: str = "all", generation: str = "all"):
    """
    Constructs a new PLEXOS model using the specified location and generation type.
    
    Parameters:
      - kb (KnowledgeBase): The shared knowledge base.
      - location (str): The location for the model. Defaults to "all" if not provided.
      - generation (str): The generation type for the model. Defaults to "all" if not provided.
    
    Returns:
      - str: A message confirming the creation of the model.
    """
    # Capitalize the values for display consistency.
    location_cap = location.capitalize()
    generation_cap = generation.capitalize()
    
    # Construct the model information string.
    model_info = f"PLEXOS model built for location {location_cap} with generation {generation_cap}."
    
    # Save the result in the shared knowledge base.
    kb.set_item("current_model", model_info)
    
    # Output the result.
    print(model_info)
    
    # Return the result for further processing.
    return model_info


@log_function_call
def run_plexos_model(kb: KnowledgeBase, scenario_name: str = "default_scenario", **kwargs):
    """
    Runs the model stored in the knowledge base with a given scenario.
    
    Parameters:
      - scenario_name (str): Name of the scenario to run.
      - **kwargs: Additional parameters if needed.
    
    Returns:
      - str: A message with the simulation result.
    """
    current_model = kb.get_item("current_model")
    if not current_model:
        result = "No model found. Please build a model first."
    else:
        result = f"Model ({current_model}) executed for scenario: {scenario_name}"
    kb.set_item("last_run_result", result)
    print(result)
    return result


@log_function_call
def analyze_results(kb, prompt=None, full_prompt=None, analysis_type="basic", model_file=None, model_details=None):
    """
    Analyzes energy model results.
    
    Parameters:
        kb (KnowledgeBase): Knowledge base
        prompt (str, optional): The original prompt
        full_prompt (str, optional): The full original prompt (added parameter)
        analysis_type (str): Type of analysis to perform
        model_file (str): Path to the model file to analyze
        model_details (dict): Details about the model
        
    Returns:
        dict: Analysis results
    """
    print(f"Emil analyzing model with {analysis_type} analysis...")
    
    # If file_path wasn't provided, try to get from KB
    if model_file is None:
        model_file = kb.get_item("latest_model_file")
        
    if model_details is None:
        model_details = kb.get_item("latest_model_details")
    
    # Check if we have a model to analyze
    if not model_file or not os.path.exists(model_file):
        print(f"No model file found to analyze.")
        result = {
            "status": "error",
            "message": "No model file found for analysis",
            "key_findings": [
                "Analysis could not proceed due to missing model file."
            ]
        }
        kb.set_item("latest_analysis_results", result)
        return result
    
    try:
        # Extract model information (if available)
        location = model_details.get('location', 'Unknown location') if model_details else 'Unknown location'
        
        # FIXED: Use generation_type explicitly and use fallback to 'generation' field
        generation = model_details.get('generation_type', 
                 model_details.get('generation', 'Unknown generation type')) if model_details else 'Unknown generation type'
        
        energy_carrier = model_details.get('energy_carrier', 'Unknown carrier') if model_details else 'Unknown carrier'
        
        # Print for debugging
        print(f"Analysis using: location={location}, generation={generation}, energy_carrier={energy_carrier}")
        
        # Parse XML to extract model data
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(model_file)
            root = tree.getroot()
            
            # Extract regions, nodes, etc.
            regions = []
            nodes = []
            
            # Look for regions
            region_elements = root.findall(".//Region") or root.findall(".//*[@name='Region']")
            for region in region_elements:
                regions.append(region.get('name'))
            
            # Look for nodes
            node_elements = root.findall(".//Node") or root.findall(".//*[@name='Node']")
            for node in node_elements:
                nodes.append({
                    'name': node.get('name'),
                    'type': node.get('type'),
                    'carrier': node.get('carrier'),
                    'region': node.get('region')
                })
                
            # Create analysis results based on extracted data
            result = {
                "status": "success",
                "message": f"Completed {analysis_type} analysis of {generation} {energy_carrier} model for {location}",
                "key_findings": [
                    f"Model contains {len(regions)} region(s): {', '.join(regions) if regions else 'None'}",
                    f"Model contains {len(nodes)} node(s)",
                    f"Primary generation type: {generation}",
                    f"Primary energy carrier: {energy_carrier}"
                ],
                "recommendations": [
                    f"Consider expanding {location} model with additional generation types",
                    f"Compare results with historical data from {location}",
                    f"Run sensitivity analysis on key parameters for {generation} generation"
                ],
                "summary": f"The {generation} {energy_carrier} model for {location} has been successfully analyzed using {analysis_type} analysis approach.",
                "data": {
                    "regions": regions,
                    "nodes": nodes,
                    "generation_type": generation,
                    "energy_carrier": energy_carrier,
                    "location": location
                }
            }
            
            # Store the analysis results for use by reporting functions
            kb.set_item("latest_analysis_results", result)
            kb.set_item("final_report", f"Analysis completed for {location} model. Key findings: {', '.join(result['key_findings'][:2])}")
            
            print(f"Analysis complete with {len(result['key_findings'])} findings.")
            return result
            
        except Exception as xml_error:
            print(f"Error parsing model file: {str(xml_error)}")
            # Provide basic analysis based on model_details
            result = {
                "status": "partial",
                "message": f"Partial analysis of {generation} {energy_carrier} model for {location}",
                "key_findings": [
                    f"Model was created for {location}",
                    f"Model uses {generation} generation type",
                    f"Model uses {energy_carrier} as energy carrier"
                ],
                "recommendations": [
                    "Consider running a more detailed analysis",
                    "Validate model parameters against industry standards"
                ],
                "summary": f"Basic analysis of {generation} {energy_carrier} model for {location}, without detailed structure extraction."
            }
            kb.set_item("latest_analysis_results", result)
            kb.set_item("final_report", f"Basic analysis completed for {location} model.")
            return result
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        result = {
            "status": "error",
            "message": f"Error during analysis: {str(e)}",
            "key_findings": [
                "Analysis encountered errors."
            ]
        }
        kb.set_item("latest_analysis_results", result)
        return result


@log_function_call
def generate_python_script(kb: KnowledgeBase, script_name: str, **kwargs):
    """
    Generates a Python script with the given script name.
    
    Parameters:
      - script_name (str): The desired name for the generated script.
      - **kwargs: Additional parameters if needed.
    
    Returns:
      - str: A message confirming script generation.
    """
    script_info = f"Generated Python script: {script_name}.py"
    kb.set_item("generated_script", script_info)
    print(script_info)
    return script_info


@log_function_call
def extract_model_parameters(prompt):
    """
    Extract energy modeling parameters from the prompt using keyword matching.
    ENHANCED to better detect generation types in model-building phrases.
    """
    import re
    print("Extracting model parameters from prompt...")
    prompt_lower = prompt.lower()
    params = {"locations": [], "generation_types": [], "energy_carriers": [], "model_type": "single"}
    
    # Extract locations with better pattern matching
    found_locations = []
    for loc in LOCATIONS:
        patterns = [
            f" for {loc.lower()}",    # "model for spain"
            f" in {loc.lower()}",     # "model in spain"  
            f"{loc.lower()} model",   # "spain model"
            f"model.*{loc.lower()}",  # "model ... spain"
            f"{loc.lower()}.*model",  # "spain ... model"
        ]
        
        if any(re.search(pattern, prompt_lower) for pattern in patterns):
            found_locations.append(loc)
    
    params["locations"] = list(set(found_locations))
    
    # ENHANCED: Extract generation types with comprehensive patterns
    found_gen_types = []
    for gen in GENERATION_TYPES.keys():
        patterns = [
            f"build.*{gen}.*model",      # "build a wind model"
            f"create.*{gen}.*model",     # "create wind model" 
            f"make.*{gen}.*model",       # "make a wind model"
            f"{gen}.*model.*for",        # "wind model for spain"
            f"a {gen} model",            # "a wind model"
            f"build {gen}",              # "build wind"
            f"create {gen}",             # "create wind"
            f"{gen} power",              # "wind power"
            f"{gen} generation",         # "wind generation"
            f"{gen} energy",             # "wind energy"
            f"develop.*{gen}",           # "develop wind"
            f"design.*{gen}.*model",     # "design wind model"
        ]
        
        # Check each pattern
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                found_gen_types.append(gen)
                print(f"🔍 Found generation type '{gen}' using pattern: {pattern}")
                break  # Found this generation type, move to next
    
    params["generation_types"] = list(set(found_gen_types))  # Remove duplicates
    
    # Extract energy carriers
    carriers = ["electricity", "hydrogen", "methane"]
    found_carriers = []
    for carrier in carriers:
        if carrier in prompt_lower:
            found_carriers.append(carrier)
    
    params["energy_carriers"] = found_carriers or ["electricity"]  # Default to electricity
    
    # Set location default only if none found
    if not params["locations"]:
        params["locations"] = ["Unknown"]
    
    print("Extracted parameters:", params)
    return params


@log_function_call
def create_single_location_model(kb, location, generation, energy_carrier="electricity"):
    """
    Create an energy model for a specific location.
    Fixed to correctly handle parameter ordering.
    
    Parameters:
      - kb (KnowledgeBase): The knowledge base
      - location (str): The location for the model.
      - generation (str): The generation type.
      - energy_carrier (str): Energy carrier.
    
    Returns:
      - dict: Information about the created model.
    """
    try:
        # Ensure proper capitalization and type handling
        location_cap = location.capitalize()
        generation_cap = generation.lower()  # Keep generation type lowercase
        carrier_cap = energy_carrier.capitalize() 
        
        print(f"Creating model for {generation_cap} {carrier_cap} in {location_cap}")
        
        # Determine paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, ".."))  # Go up one level from core
        models_dir = os.path.join(script_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{location_cap}_{generation_cap}_{carrier_cap}_{timestamp}"
        safe_filename = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
        output_xml = os.path.join(models_dir, f"{safe_filename}.xml")
        
        # Check if PLEXOS is available and if the basefile exists
        basefile_path = os.path.join(project_dir, "basefile.xml")
        if os.path.exists(basefile_path) and PLEXOS_AVAILABLE:
            # PLEXOS processing code
            try:
                print("Using PLEXOS API to create model...")
                copyfile(basefile_path, output_xml)
                db = DatabaseCore()
                db.DisplayAlerts = False
                db.Connection(output_xml)
                
                # Add categories
                print(f"Adding generation category: {generation_cap}")
                db.AddCategory(int(ClassEnum.Node), generation_cap)
                print(f"Adding energy carrier category: {carrier_cap}")
                db.AddCategory(int(ClassEnum.Node), carrier_cap)
                
                # Add region
                region_exists = False
                try:
                    region_id = db.GetObjectID(location_cap, int(ClassEnum.Region))
                    region_exists = True
                    print(f"Region {location_cap} already exists with ID {region_id}")
                except:
                    region_exists = False
                
                if not region_exists:
                    db.AddObject(location_cap, int(ClassEnum.Region), True, '', '')
                    db.AddMembership(int(CollectionEnum.SystemRegions), '', location_cap)
                
                # Add node
                node_name = f"{location_cap}_{generation_cap}_{carrier_cap}"
                node_exists = False
                try:
                    node_id = db.GetObjectID(node_name, int(ClassEnum.Node))
                    node_exists = True
                    print(f"Node {node_name} already exists with ID {node_id}")
                except:
                    node_exists = False
                
                if not node_exists:
                    db.AddObject(node_name, int(ClassEnum.Node), True, '', '')
                    db.AddMembership(int(CollectionEnum.SystemNodes), '', node_name)
                    
                    # Add memberships
                    try:
                        if hasattr(CollectionEnum, 'RegionNodes'):
                            db.AddMembership(int(CollectionEnum.RegionNodes), location_cap, node_name)
                    except Exception as e:
                        print(f"Warning: Could not add region membership: {str(e)}")
                    
                    try:
                        if hasattr(CollectionEnum, 'CategoryObjects'):
                            db.AddMembership(int(CollectionEnum.CategoryObjects), generation_cap, node_name)
                    except Exception as e:
                        print(f"Warning: Could not add category membership: {str(e)}")
                    
                    try:
                        if hasattr(CollectionEnum, 'CategoryObjects'):
                            db.AddMembership(int(CollectionEnum.CategoryObjects), carrier_cap, node_name)
                    except Exception as e:
                        print(f"Warning: Could not add carrier category membership: {str(e)}")
                
                db.Close()
                print(f"✅ Created PLEXOS XML file: {output_xml}")
                return {
                    "status": "success",
                    "message": f"Created {generation_cap} {carrier_cap} model for {location_cap}",
                    "file": output_xml,
                    "location": location_cap,
                    "generation_type": generation_cap,
                    "energy_carrier": carrier_cap
                }
            except Exception as plexos_error:
                print(f"❌ PLEXOS error: {str(plexos_error)}")
                print("Falling back to simple XML creation...")
                return create_simple_xml(location_cap, generation_cap, carrier_cap, output_xml)
        else:
            print("PLEXOS not available or basefile not found, creating simple XML...")
            return create_simple_xml(location_cap, generation_cap, carrier_cap, output_xml)
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return {"status": "error", "message": f"Failed to create model: {str(e)}"}


@log_function_call
def create_simple_xml(location_name, gen_type_name, carrier_name, output_xml):
    """
    Create a simple XML model file when PLEXOS is not available.
    
    Parameters:
      - location_name (str): Location name.
      - gen_type_name (str): Generation type.
      - carrier_name (str): Energy carrier.
      - output_xml (str): Output file path.
    
    Returns:
      - dict: Result information.
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.Element("PLEXOSModel", version="7.4")
        root.set("xmlns", "https://custom-energy-model/non-plexos/v1")
        comment = ET.Comment("This is a placeholder XML file and is not a PLEXOS database")
        root.append(comment)
        
        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        timestamp = ET.SubElement(metadata, "Timestamp")
        timestamp.text = datetime.datetime.now().isoformat()
        
        modeltype = ET.SubElement(metadata, "ModelType")
        modeltype.text = gen_type_name
        
        location_elem = ET.SubElement(metadata, "Location")
        location_elem.text = location_name
        
        carrier_elem = ET.SubElement(metadata, "EnergyCarrier")
        carrier_elem.text = carrier_name
        
        ET.SubElement(metadata, "Format").text = "Custom XML (Not PLEXOS)"
        
        # Add entities
        entities = ET.SubElement(root, "Entities")
        
        # Add region
        regions = ET.SubElement(entities, "Regions")
        region = ET.SubElement(regions, "Region", id="1", name=location_name)
        
        # Add node
        nodes = ET.SubElement(entities, "Nodes")
        node = ET.SubElement(nodes, "Node", id="1", 
                            name=f"{location_name}_{gen_type_name}_{carrier_name}", 
                            type=gen_type_name,
                            carrier=carrier_name,
                            region=location_name)
        
        # Write the XML to file
        with open(output_xml, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write('<!-- NOT A PLEXOS DATABASE. -->\n')
            xml_str = ET.tostring(root, encoding='utf-8').decode('utf-8')
            f.write(xml_str)
            
        print(f"✅ Created simple XML: {output_xml}\n------------------------------")
        return {
            "status": "success",
            "message": f"Created {gen_type_name} {carrier_name} model for {location_name}",
            "file": output_xml,
            "location": location_name,  # FIXED: This now correctly returns location_name instead of gen_type_name
            "generation_type": gen_type_name,
            "energy_carrier": carrier_name
        }
    except Exception as e:
        print(f"❌ Error creating simple XML: {str(e)}")
        return {"status": "error", "message": f"Failed to create simple XML: {str(e)}"}


@log_function_call
def create_multi_location_model(locations, generation_type, energy_carrier="electricity"):
    """
    Create a combined energy model for multiple locations.
    
    Parameters:
      - locations (list): List of locations.
      - generation_type (str): Generation type.
      - energy_carrier (str): Energy carrier.
    
    Returns:
      - dict: Information about the created model.
    """
    try:
        location_names = [loc.capitalize() for loc in locations]
        gen_type_name = generation_type
        carrier_name = energy_carrier.capitalize()
        locations_display = ", ".join(location_names)
        print(f"Creating combined model for {gen_type_name} {carrier_name} in {locations_display}")
        
        # Set up paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, ".."))
        models_dir = os.path.join(script_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"MultiLoc_{gen_type_name.replace(' ', '_')}_{carrier_name}_{timestamp}"
        safe_filename = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
        output_xml = os.path.join(models_dir, f"{safe_filename}.xml")
        
        # Check if we can use PLEXOS
        basefile_path = os.path.join(project_dir, "basefile.xml")
        if os.path.exists(basefile_path) and PLEXOS_AVAILABLE:
            try:
                print("Using PLEXOS API to create multi-location model...")
                copyfile(basefile_path, output_xml)
                from PLEXOS_NET.Core import DatabaseCore
                db = DatabaseCore()
                db.DisplayAlerts = False
                db.Connection(output_xml)
                
                # Add categories
                print(f"Adding category: {gen_type_name}")
                db.AddCategory(int(ClassEnum.Node), gen_type_name)
                print(f"Adding energy carrier category: {carrier_name}")
                db.AddCategory(int(ClassEnum.Node), carrier_name)
                
                # Process each location
                for location_name in location_names:
                    region_exists = False
                    try:
                        region_id = db.GetObjectID(location_name, int(ClassEnum.Region))
                        region_exists = True
                        print(f"Region {location_name} already exists with ID {region_id}")
                    except:
                        region_exists = False
                    
                    if not region_exists:
                        db.AddObject(location_name, int(ClassEnum.Region), True, '', '')
                        db.AddMembership(int(CollectionEnum.SystemRegions), '', location_name)
                    
                    # Add node for this location
                    node_name = f"{location_name}_{gen_type_name}_{carrier_name}"
                    node_exists = False
                    try:
                        node_id = db.GetObjectID(node_name, int(ClassEnum.Node))
                        node_exists = True
                        print(f"Node {node_name} already exists with ID {node_id}")
                    except:
                        node_exists = False
                    
                    if not node_exists:
                        db.AddObject(node_name, int(ClassEnum.Node), True, '', '')
                        db.AddMembership(int(CollectionEnum.SystemNodes), '', node_name)
                        
                        # Add memberships
                        try:
                            if hasattr(CollectionEnum, 'RegionNodes'):
                                db.AddMembership(int(CollectionEnum.RegionNodes), location_name, node_name)
                        except Exception as e:
                            print(f"Warning: Could not add region membership: {str(e)}")
                        
                        try:
                            if hasattr(CollectionEnum, 'CategoryObjects'):
                                db.AddMembership(int(CollectionEnum.CategoryObjects), gen_type_name, node_name)
                        except Exception as e:
                            print(f"Warning: Could not add category membership: {str(e)}")
                        
                        try:
                            if hasattr(CollectionEnum, 'CategoryObjects'):
                                db.AddMembership(int(CollectionEnum.CategoryObjects), carrier_name, node_name)
                        except Exception as e:
                            print(f"Warning: Could not add carrier category membership: {str(e)}")
                
                db.Close()
                print(f"✅ Created multi-location PLEXOS XML file: {output_xml}")
                return {
                    "status": "success",
                    "message": f"{gen_type_name} {carrier_name} model created for multiple regions: {locations_display}",
                    "file": output_xml,
                    "locations": location_names,
                    "generation_type": gen_type_name,
                    "energy_carrier": carrier_name
                }
            except Exception as plexos_error:
                print(f"❌ PLEXOS error: {str(plexos_error)}")
                print("Falling back to simple multi-location XML creation...")
                return create_simple_multi_location_xml(location_names, gen_type_name, carrier_name, output_xml)
        else:
            print("PLEXOS not available or basefile not found, creating simple multi-location XML...")
            return create_simple_multi_location_xml(location_names, gen_type_name, carrier_name, output_xml)
    except Exception as e:
        print(f"❌ Error creating multi-location model: {str(e)}")
        return {"status": "error", "message": f"Failed to create multi-location model: {str(e)}"}


@log_function_call
def create_simple_multi_location_xml(location_names, gen_type_name, carrier_name, output_xml):
    """
    Create a simple XML model file for multiple locations when PLEXOS is not available.
    
    Parameters:
      - location_names (list): List of location names.
      - gen_type_name (str): Generation type.
      - carrier_name (str): Energy carrier.
      - output_xml (str): Output file path.
    
    Returns:
      - dict: Result information.
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.Element("PLEXOSModel", version="7.4")
        root.set("xmlns", "https://custom-energy-model/non-plexos/v1")
        comment = ET.Comment("This is a placeholder XML file and is not a PLEXOS database")
        root.append(comment)
        
        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        timestamp = ET.SubElement(metadata, "Timestamp")
        timestamp.text = datetime.datetime.now().isoformat()
        
        modeltype = ET.SubElement(metadata, "ModelType")
        modeltype.text = gen_type_name
        
        carrier_elem = ET.SubElement(metadata, "EnergyCarrier")
        carrier_elem.text = carrier_name
        
        ET.SubElement(metadata, "Format").text = "Custom XML (Not PLEXOS)"
        
        # Add locations to metadata
        locations_elem = ET.SubElement(metadata, "Locations")
        for loc in location_names:
            location_elem = ET.SubElement(locations_elem, "Location")
            location_elem.text = loc
        
        # Add entities
        entities = ET.SubElement(root, "Entities")
        
        # Add regions
        regions = ET.SubElement(entities, "Regions")
        for idx, location_name in enumerate(location_names, 1):
            region = ET.SubElement(regions, "Region", id=str(idx), name=location_name)
        
        # Add nodes
        nodes = ET.SubElement(entities, "Nodes")
        node_id = 1
        for location_name in location_names:
            node = ET.SubElement(nodes, "Node", 
                               id=str(node_id), 
                               name=f"{location_name}_{gen_type_name}_{carrier_name}", 
                               type=gen_type_name,
                               carrier=carrier_name,
                               region=location_name)
            node_id += 1
        
        # Write the XML to file
        with open(output_xml, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write('<!-- NOT A PLEXOS DATABASE. -->\n')
            xml_str = ET.tostring(root, encoding='utf-8').decode('utf-8')
            f.write(xml_str)
        
        print(f"✅ Created simple multi-location XML: {output_xml}")
        return {
            "status": "success",
            "message": f"Created {gen_type_name} {carrier_name} model for multiple regions: {', '.join(location_names)}",
            "file": output_xml,
            "locations": location_names,
            "generation_type": gen_type_name,
            "energy_carrier": carrier_name
        }
    except Exception as e:
        print(f"❌ Error creating simple multi-location XML: {str(e)}")
        return {"status": "error", "message": f"Failed to create simple multi-location XML: {str(e)}"}


@log_function_call
def create_simple_comprehensive_xml(location_names, gen_type_names, carrier_names, output_xml):
    """
    Create a simple XML model file for multiple locations and generation types when PLEXOS is not available.
    
    Parameters:
      - location_names (list): List of location names.
      - gen_type_names (list): List of generation types.
      - carrier_names (list): List of energy carriers.
      - output_xml (str): Output file path.
    
    Returns:
      - dict: Result information.
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.Element("PLEXOSModel", version="7.4")
        root.set("xmlns", "https://custom-energy-model/non-plexos/v1")
        comment = ET.Comment("This is a placeholder XML file and is not a PLEXOS database")
        root.append(comment)
        
        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        timestamp = ET.SubElement(metadata, "Timestamp")
        timestamp.text = datetime.datetime.now().isoformat()
        
        ET.SubElement(metadata, "Format").text = "Custom XML (Not PLEXOS)"
        
        # Add generation types to metadata
        generationTypes = ET.SubElement(metadata, "GenerationTypes")
        for gen_type in gen_type_names:
            genType = ET.SubElement(generationTypes, "GenerationType")
            genType.text = gen_type
        
        # Add energy carriers to metadata
        energyCarriers = ET.SubElement(metadata, "EnergyCarriers")
        for carrier in carrier_names:
            carrierElem = ET.SubElement(energyCarriers, "EnergyCarrier")
            carrierElem.text = carrier
        
        # Add locations to metadata
        locations = ET.SubElement(metadata, "Locations")
        for location in location_names:
            locationElem = ET.SubElement(locations, "Location")
            locationElem.text = location
        
        # Add entities
        entities = ET.SubElement(root, "Entities")
        
        # Add regions
        regions = ET.SubElement(entities, "Regions")
        for idx, location_name in enumerate(location_names, 1):
            region = ET.SubElement(regions, "Region", id=str(idx), name=location_name)
        
        # Add nodes
        nodes = ET.SubElement(entities, "Nodes")
        node_id = 1
        for location_name in location_names:
            for gen_type in gen_type_names:
                for carrier in carrier_names:
                    node = ET.SubElement(nodes, "Node", 
                                       id=str(node_id), 
                                       name=f"{location_name}_{gen_type}_{carrier}", 
                                       type=gen_type,
                                       carrier=carrier,
                                       region=location_name)
                    node_id += 1
        
        # Write the XML to file
        with open(output_xml, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write('<!-- NOT A PLEXOS DATABASE. -->\n')
            xml_str = ET.tostring(root, encoding='utf-8').decode('utf-8')
            f.write(xml_str)
        
        print(f"✅ Created simple comprehensive XML: {output_xml}")
        locations_display = ", ".join(location_names)
        gen_types_display = ", ".join(gen_type_names)
        carriers_display = ", ".join(carrier_names)
        return {
            "status": "success",
            "message": f"Created comprehensive model with {len(gen_type_names)} generation types for {len(location_names)} locations",
            "details": f"Generation types: {gen_types_display}; Locations: {locations_display}; Carriers: {carriers_display}",
            "file": output_xml,
            "locations": location_names,
            "generation_types": gen_type_names,
            "energy_carriers": carrier_names
        }
    except Exception as e:
        print(f"❌ Error creating simple comprehensive XML: {str(e)}")
        return {"status": "error", "message": f"Failed to create simple comprehensive XML: {str(e)}"}


@log_function_call
def create_comprehensive_model(locations, generation_types, energy_carriers=["electricity"]):
    """
    Create a comprehensive energy model for multiple locations and generation types.
    FIXED to properly handle location lists without character-splitting.
    
    Parameters:
      - locations (list): List of locations.
      - generation_types (list): List of generation types.
      - energy_carriers (list): List of energy carriers.
    
    Returns:
      - dict: Information about the created model.
    """
    try:
        # Normalize inputs to ensure proper processing
        # Fix for location parsing issue - properly handle both strings and lists
        if isinstance(locations, str):
            # If locations is a string, split by commas if present
            if ',' in locations:
                location_names = [loc.strip().capitalize() for loc in locations.split(',')]
            else:
                location_names = [locations.strip().capitalize()]
        else:
            # If locations is already a list, capitalize each item
            location_names = [loc.strip().capitalize() for loc in locations]
        
        # Safeguard against individual characters being treated as separate locations
        # If any location is a single character (except 'A' which could be Austria), join them
        if all(len(loc) == 1 for loc in location_names) and len(location_names) > 2:
            combined_location = ''.join(location_names)
            
            # Check if it might be a common country name
            potential_countries = ['France', 'Spain', 'Italy', 'Germany', 'Poland', 'Sweden', 'Norway', 'Finland']
            
            for country in potential_countries:
                # Case-insensitive check
                if combined_location.lower() == country.lower():
                    # Use the proper country name instead
                    location_names = [country]
                    break
            else:
                # If no match found, use the combined string as a single location
                location_names = [combined_location]
        
        # Similarly normalize generation types and carriers
        if isinstance(generation_types, str):
            if ',' in generation_types:
                gen_type_names = [gen.strip().lower() for gen in generation_types.split(',')]
            else:
                gen_type_names = [generation_types.strip().lower()]
        else:
            gen_type_names = [gen.strip().lower() for gen in generation_types]
            
        if isinstance(energy_carriers, str):
            if ',' in energy_carriers:
                carrier_names = [carrier.strip().capitalize() for carrier in energy_carriers.split(',')]
            else:
                carrier_names = [energy_carriers.strip().capitalize()]
        else:
            carrier_names = [carrier.strip().capitalize() for carrier in energy_carriers]
        
        locations_display = ", ".join(location_names)
        gen_types_display = ", ".join(gen_type_names)
        carriers_display = ", ".join(carrier_names)
        
        print(f"Creating comprehensive model for {gen_types_display} ({carriers_display}) in {locations_display}")
        
        # Set up paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, ".."))
        models_dir = os.path.join(script_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"Comprehensive_Model_{timestamp}.xml"
        output_xml = os.path.join(models_dir, model_name)
        
        # Check if we can use PLEXOS
        basefile_path = os.path.join(project_dir, "basefile.xml")
        if os.path.exists(basefile_path) and PLEXOS_AVAILABLE:
            try:
                print("Using PLEXOS API to create comprehensive model...")
                copyfile(basefile_path, output_xml)
                from PLEXOS_NET.Core import DatabaseCore
                db = DatabaseCore()
                db.DisplayAlerts = False
                db.Connection(output_xml)
                
                # Add all generation type categories
                for gen_type in gen_type_names:
                    print(f"Adding generation type category: {gen_type}")
                    db.AddCategory(int(ClassEnum.Node), gen_type)
                
                # Add all energy carrier categories
                for carrier in carrier_names:
                    print(f"Adding energy carrier category: {carrier}")
                    db.AddCategory(int(ClassEnum.Node), carrier)
                
                # Process each location
                for location in location_names:
                    region_exists = False
                    try:
                        region_id = db.GetObjectID(location, int(ClassEnum.Region))
                        region_exists = True
                        print(f"Region {location} exists with ID {region_id}")
                    except:
                        region_exists = False
                    
                    if not region_exists:
                        db.AddObject(location, int(ClassEnum.Region), True, '', '')
                        db.AddMembership(int(CollectionEnum.SystemRegions), '', location)
                    
                    # Add nodes for each generation type and carrier combination
                    for gen_type in gen_type_names:
                        for carrier in carrier_names:
                            node_name = f"{location}_{gen_type}_{carrier}"
                            node_exists = False
                            try:
                                node_id = db.GetObjectID(node_name, int(ClassEnum.Node))
                                node_exists = True
                            except:
                                node_exists = False
                            
                            if not node_exists:
                                db.AddObject(node_name, int(ClassEnum.Node), True, '', '')
                                db.AddMembership(int(CollectionEnum.SystemNodes), '', node_name)
                                
                                # Add memberships
                                try:
                                    if hasattr(CollectionEnum, 'RegionNodes'):
                                        db.AddMembership(int(CollectionEnum.RegionNodes), location, node_name)
                                except Exception as e:
                                    print(f"Warning: Could not add region membership: {str(e)}")
                                
                                try:
                                    if hasattr(CollectionEnum, 'CategoryObjects'):
                                        db.AddMembership(int(CollectionEnum.CategoryObjects), gen_type, node_name)
                                except Exception as e:
                                    print(f"Warning: Could not add category membership: {str(e)}")
                                
                                try:
                                    if hasattr(CollectionEnum, 'CategoryObjects'):
                                        db.AddMembership(int(CollectionEnum.CategoryObjects), carrier, node_name)
                                except Exception as e:
                                    print(f"Warning: Could not add carrier membership: {str(e)}")
                
                db.Close()
                print(f"✅ Created comprehensive PLEXOS XML file: {output_xml}")
                return {
                    "status": "success",
                    "message": f"Created comprehensive model with {len(gen_type_names)} generation types for {len(location_names)} locations",
                    "details": f"Generation types: {gen_types_display}; Locations: {locations_display}; Carriers: {carriers_display}",
                    "file": output_xml,
                    "locations": location_names,
                    "generation_types": gen_type_names,
                    "energy_carriers": carrier_names
                }
            except Exception as plexos_error:
                print(f"❌ PLEXOS error: {str(plexos_error)}")
                print("Falling back to simple comprehensive XML creation...")
                return create_simple_comprehensive_xml(location_names, gen_type_names, carrier_names, output_xml)
        else:
            print("PLEXOS not available or basefile not found, creating simple comprehensive XML...")
            return create_simple_comprehensive_xml(location_names, gen_type_names, carrier_names, output_xml)
    except Exception as e:
        print(f"❌ Error creating comprehensive model: {str(e)}")
        return {"status": "error", "message": f"Failed to create comprehensive model: {str(e)}"}


@log_function_call
def process_emil_request(kb: KnowledgeBase, prompt: str = None, location=None, generation=None, energy_carrier=None, **kwargs):
    """
    This function wraps Emil's energy model processing logic with parameter validation.
    
    Parameters:
        - kb (KnowledgeBase): The knowledge base to store results
        - prompt (str, optional): The user's original prompt
        - location (str, optional): Explicit location parameter from Nova
        - generation (str, optional): Explicit generation type parameter from Nova
        - energy_carrier (str, optional): Explicit energy carrier parameter from Nova
        - **kwargs: Additional flexible parameters for model creation
        
    Returns:
        - dict: Information about the created model
    """
    # Ensure prompt is set
    if prompt is None:
        prompt = kwargs.get('prompt', 'Create an energy model')

    print(f"Processing Emil request with prompt: {prompt}")
    
    # Extract parameters from the prompt
    extracted_params = extract_model_parameters(prompt)
    
    # Get the parameters passed explicitly by Nova
    passed_location = location
    passed_generation = generation
    passed_energy_carrier = energy_carrier
    
    # Debug logging of both sources of parameters - IMPROVED
    print(f"Original Parameters: generation_types={extracted_params['generation_types']}, energy_carriers={extracted_params['energy_carriers']}")
    print(f"User Added Parameters: generation={passed_generation or 'Not provided'}, energy_carrier={passed_energy_carrier or 'Not provided'}")
    
    # FIXED: Always prioritize user-provided parameters
    
    # For location
    if passed_location:
        # Always use user-provided location parameter
        final_location = passed_location
    elif extracted_params['locations'] and extracted_params['locations'][0] != "Unknown":
        # Location was successfully extracted from prompt
        final_location = extracted_params['locations'][0]
    else:
        # If no location specified, return error - never default silently
        return {
            "status": "error", 
            "message": "Location is required for energy modeling"
        }
    
    # For generation type - FIXED to always prioritize user input
    if passed_generation:
        # Always use user-provided generation parameter
        final_generation = passed_generation
    elif extracted_params['generation_types']:
        final_generation = extracted_params['generation_types'][0]
    else:
        # Never default to a value, require explicit parameter
        return {
            "status": "error",
            "message": "Generation type is required"
        }
    
    # For energy carrier
    valid_carriers = ["electricity", "hydrogen", "methane"]
    if passed_energy_carrier:
        # Always use user-provided energy carrier
        final_energy_carrier = passed_energy_carrier
    elif extracted_params['energy_carriers']:
        final_energy_carrier = extracted_params['energy_carriers'][0]
    else:
        # Default to electricity only if not specified
        final_energy_carrier = "electricity"
    
    # Log what we decided to use - IMPROVED
    print(f"Parameters from Nova to Emil: location={final_location}, generation={final_generation}, energy_carrier={final_energy_carrier}\n")
    
    # Create the model with the validated parameters
    result = create_single_location_model(kb, final_location, final_generation, final_energy_carrier)
    
    # Store the result in the knowledge base
    kb.set_item("emil_result", result)
    kb.set_item("latest_model_details", result)
    if 'file' in result:
        kb.set_item("latest_model_file", result['file'])
    
    return result


@log_function_call
def write_report(kb: KnowledgeBase, prompt=None, style="executive_summary", model_file=None, model_details=None, analysis_results=None):
    """
    Writes a report based on model and analysis results.
    
    Parameters:
        kb (KnowledgeBase): Knowledge base
        prompt (str): Original user prompt
        style (str): Report style ("executive_summary", "technical_report", etc.)
        model_file (str): Path to the model file
        model_details (dict): Details about the model
        analysis_results (dict): Results from previous analysis
        
    Returns:
        str: The generated report
    """
    print(f"Writing report in {style} style")
    
    # Get model information from KB if not provided
    if model_file is None:
        model_file = kb.get_item("latest_model_file") 
    
    if model_details is None:
        model_details = kb.get_item("latest_model_details")
    
    if analysis_results is None:
        analysis_results = kb.get_item("latest_analysis_results")
    
    # Check if we have the necessary information to write a report
    if not model_details:
        return "Error: No model details available for report generation."
    
    if not analysis_results:
        # Create a basic analysis result if none exists
        analysis_results = {
            "key_findings": [
                f"Model was created successfully for {model_details.get('location', 'unknown location')}"
            ],
            "recommendations": [
                "Consider running a detailed analysis on this model",
                "Validate with historical data"
            ],
            "summary": f"A {model_details.get('generation_type', 'energy')} model was created for {model_details.get('location', 'unknown location')}."
        }
    
    # Extract key information
    location = model_details.get('location', 'Unknown location')
    generation = model_details.get('generation_type', 'Unknown generation type')
    energy_carrier = model_details.get('energy_carrier', 'Unknown carrier')
    
    # Determine report type based on style
    if style == "executive_summary":
        report = generate_executive_summary(prompt, model_details, analysis_results, location, generation, energy_carrier)
    elif style == "technical_report":
        report = generate_technical_report(prompt, model_details, analysis_results, location, generation, energy_carrier)
    elif style == "presentation_report":
        report = generate_presentation_report(prompt, model_details, analysis_results, location, generation, energy_carrier)
    else:
        # Default to executive summary
        report = generate_executive_summary(prompt, model_details, analysis_results, location, generation, energy_carrier)
    
    # Store the final report in the knowledge base
    kb.set_item("latest_report", report)
    kb.set_item("final_report", report)  # This is used as the final output
    
    # Call the new store_report method on the KnowledgeBase
    kb.store_report(report, prompt, model_details)
    
    return report


def generate_executive_summary(prompt, model_details, analysis_results, location, generation, energy_carrier):
    """
    Generates an executive summary report.
    
    Parameters:
        prompt (str): Original user prompt
        model_details (dict): Details about the model
        analysis_results (dict): Analysis results
        location (str): Location for the model
        generation (str): Generation type
        energy_carrier (str): Energy carrier
        
    Returns:
        str: The executive summary report
    """
    # Extract relevant information
    key_findings = analysis_results.get('key_findings', ['No specific findings available'])
    recommendations = analysis_results.get('recommendations', ['Consider further analysis'])
    
    # Generate the report
    report = f"""
# Executive Summary: {generation.capitalize()} Energy Model for {location.capitalize()}

## Overview
This report summarizes the energy model created for {location}, focusing on {generation} generation with {energy_carrier} as the primary energy carrier.

## Key Findings
"""
    
    # Add key findings as bullet points
    for finding in key_findings:
        report += f"- {finding}\n"
    
    # Add recommendations
    report += f"\n## Recommendations\n"
    for rec in recommendations:
        report += f"- {rec}\n"
    
    # Add summary
    report += f"""
## Summary
{analysis_results.get('summary', 'A model has been successfully created according to specifications.')}

Report generated based on user request: "{prompt}"
"""
    
    return report


def generate_technical_report(prompt, model_details, analysis_results, location, generation, energy_carrier):
    """
    Generates a technical report with detailed information.
    
    Parameters:
        prompt (str): Original user prompt
        model_details (dict): Details about the model
        analysis_results (dict): Analysis results
        location (str): Location for the model
        generation (str): Generation type
        energy_carrier (str): Energy carrier
        
    Returns:
        str: The technical report
    """
    # Technical report includes more details
    report = f"""
# Technical Report: {generation.capitalize()} Energy Model for {location.capitalize()}

## Model Specifications
- **Location**: {location}
- **Generation Type**: {generation}
- **Energy Carrier**: {energy_carrier}
- **File Path**: {model_details.get('file', 'Not specified')}
- **Status**: {model_details.get('status', 'Unknown')}

## Model Structure
"""
    
    # Add information about regions
    regions = analysis_results.get('data', {}).get('regions', [])
    if regions:
        report += "### Regions\n"
        for region in regions:
            report += f"- {region}\n"
    
    # Add information about nodes
    nodes = analysis_results.get('data', {}).get('nodes', [])
    if nodes:
        report += "\n### Nodes\n"
        for node in nodes:
            report += f"- **{node.get('name', 'Unnamed')}**\n"
            report += f"  - Type: {node.get('type', 'Not specified')}\n"
            report += f"  - Carrier: {node.get('carrier', 'Not specified')}\n"
            report += f"  - Region: {node.get('region', 'Not specified')}\n"
    
    # Add key findings
    key_findings = analysis_results.get('key_findings', ['No specific findings available'])
    report += "\n## Analysis Results\n"
    for finding in key_findings:
        report += f"- {finding}\n"
    
    # Add technical recommendations
    recommendations = analysis_results.get('recommendations', ['Consider further analysis'])
    report += "\n## Technical Recommendations\n"
    for rec in recommendations:
        report += f"- {rec}\n"
    
    report += f"""
Report generated based on user request: "{prompt}"
"""
    
    return report


def generate_presentation_report(prompt, model_details, analysis_results, location, generation, energy_carrier):
    """
    Generates a presentation-style report.
    
    Parameters:
        prompt (str): Original user prompt
        model_details (dict): Details about the model
        analysis_results (dict): Analysis results
        location (str): Location for the model
        generation (str): Generation type
        energy_carrier (str): Energy carrier
        
    Returns:
        str: The presentation-style report
    """
    # Presentation report is more concise with clear sections
    report = f"""
# {generation.capitalize()} Energy Model for {location.capitalize()}

## Highlights
- Successfully created {generation} model for {location}
- Model uses {energy_carrier} as primary energy carrier
- File generated: {os.path.basename(model_details.get('file', 'model.xml'))}

## Key Insights
"""
    
    # Add key findings
    key_findings = analysis_results.get('key_findings', ['Model created successfully'])
    for i, finding in enumerate(key_findings, 1):
        report += f"{i}. {finding}\n"
    
    # Add next steps
    recommendations = analysis_results.get('recommendations', ['Consider further analysis'])
    report += f"\n## Next Steps\n"
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"
    
    report += f"""
_Report generated based on: "{prompt}"_
"""
    
    return report


# ----------------------------------------------------------------------
# Enhanced PLEXOS Integration
# ----------------------------------------------------------------------

# Import the working PLEXOS model builder
PLEXOS_BASE_AVAILABLE = False
try:
    import sys
    import os
    # Add path to agents directory to find plexos_base_model_final.py
    agents_path = os.path.join(os.path.dirname(__file__), '..', 'agents')
    if agents_path not in sys.path:
        sys.path.append(agents_path)
    
    # Try to import the PLEXOS base model functions
    from plexos_base_model_final import (
        process_base_model_task, 
        initiate_file,
        ai_call,
        extract_countries_with_retries,
        filter_data,
        load_plexos_xml
    )
    import plexos_base_model_final as pbf
    PLEXOS_BASE_AVAILABLE = True
    print("✅ Successfully imported PLEXOS base model functionality")
except ImportError as e:
    PLEXOS_BASE_AVAILABLE = False
    print(f"❌ Could not import PLEXOS base model: {e}")
except Exception as e:
    PLEXOS_BASE_AVAILABLE = False
    print(f"❌ Error importing PLEXOS base model: {e}")

@log_function_call
def process_emil_request_enhanced(kb: KnowledgeBase, prompt: str = None, location=None, generation=None, energy_carrier=None, **kwargs):
    """
    Enhanced Emil request processor that uses the full PLEXOS model building pipeline.
    
    Parameters:
        - kb (KnowledgeBase): The knowledge base to store results
        - prompt (str, optional): The user's original prompt
        - location (str, optional): Explicit location parameter
        - generation (str, optional): Explicit generation type parameter  
        - energy_carrier (str, optional): Explicit energy carrier parameter
        - **kwargs: Additional flexible parameters for model creation
        
    Returns:
        - dict: Information about the created model
    """
    print("\n" + "="*50)
    print("🚀 ENHANCED PLEXOS MODEL BUILDING STARTED")
    print("="*50)
    print(f"📝 Prompt: {prompt}")
    print(f"📍 Location: {location}")
    print(f"⚡ Generation: {generation}")
    print(f"🔋 Energy Carrier: {energy_carrier}")
    print("="*50 + "\n")
    
    if not PLEXOS_BASE_AVAILABLE:
        print("❌ PLEXOS base model not available, falling back to simple XML creation")
        return process_emil_request(kb, prompt, location, generation, energy_carrier, **kwargs)
    
    print("🔧 Using enhanced PLEXOS model building functionality")
    print("🔧 Processing Emil request with enhanced PLEXOS functionality")
    
    # Ensure prompt is set
    if prompt is None:
        prompt = kwargs.get('prompt', 'Create an energy model')
    
    print(f"Prompt: {prompt}")
    
    try:
        # Load the PLEXOS Model Builder Excel file
        script_dir = os.path.dirname(__file__)
        excel_path = os.path.join(script_dir, '..', 'agents', 'PLEXOS_inputs', 'PLEXOS_Model_Builder_v2.xlsx')
        
        if not os.path.exists(excel_path):
            print(f"❌ PLEXOS Model Builder Excel file not found at: {excel_path}")
            print("📁 Expected path structure:")
            print(f"   {excel_path}")
            print("💡 Make sure PLEXOS_Model_Builder_v2.xlsx is in src/agents/PLEXOS_inputs/")
            # Fall back to simple model creation
            return process_emil_request(kb, prompt, location, generation, energy_carrier, **kwargs)
        
        print("✅ Successfully imported required modules")
        
        # Parse the prompt using PLEXOS prompt sheet builder
        print("📋 Parsing prompt into PLEXOS task structure...")
        
        # Check API key availability
        api_key_available = hasattr(pbf.oaic, 'API_KEY') and pbf.oaic.API_KEY is not None
        print(f"✅ API Key available: {api_key_available}")
        
        if api_key_available:
            print(f"✅ Using model: {'gpt-4.1-nano'}")
            
            # Use PLEXOS prompt parsing
            try:
                from PLEXOS_functions import prompt_templates as pt
                
                context = "You are an assistant helping to build an energy model through the PLEXOS API."
                formatted_prompt = pt.plexos_prompt_sheet_builder(prompt)
                
                task_response = pbf.oaic.run_open_ai_ns(formatted_prompt, context, model="gpt-4.1-nano")
                import json
                task_list = json.loads(task_response)
                print(f"✅ Parsed tasks from prompt: {task_list}")
            except Exception as e:
                print(f"❌ Error parsing tasks: {e}")
                task_list = {"Base Model Task": prompt, "Modifications": []}
        else:
            print("⚠️ API key not available, using simple task parsing")
            task_list = {"Base Model Task": prompt, "Modifications": []}
        
        # Extract countries from prompt
        print("🌍 Extracting countries from prompt...")
        if api_key_available:
            try:
                # Load node mapping for country extraction
                import pandas as pd
                node_map_path = os.path.join(script_dir, '..', 'agents', 'PLEXOS_inputs', 'node_mapping.csv')
                if os.path.exists(node_map_path):
                    node_map = pd.read_csv(node_map_path)
                    reduced_map = node_map.drop(['Longitude', 'Latitude', 'PLEXOS_Region', 'Node'], axis=1).drop_duplicates().reset_index(drop=True)
                    
                    # Use the enhanced country extraction
                    from PLEXOS_functions.search_embedding import find_multiple_values as fmv
                    from PLEXOS_functions import prompt_templates as pt
                    
                    countries_embed = fmv(prompt, reduced_map, ['Country'], threshold=0.5)
                    country_prompt = pt.extract_countries(countries_embed, prompt)
                    
                    context = "You are building a PLEXOS model for a client."
                    countries = pbf.extract_countries_with_retries(country_prompt, context, model='gpt-4.1-nano')
                    print(f"✅ Extracted countries from LLM: {countries}")
                else:
                    print("⚠️ Node mapping file not found, using parameter-based country extraction")
                    countries = [location] if location else ['Unknown']
            except Exception as e:
                print(f"❌ Error extracting countries: {e}")
                countries = [location] if location else ['Unknown']
        else:
            countries = [location] if location else ['Unknown']
        
        # Create model file name
        print("📁 Creating model file...")
        if "Base Model Task" in task_list and task_list["Base Model Task"]:
            base_task = task_list["Base Model Task"]
            try:
                filename = pbf.initiate_file(base_task, model='gpt-4.1-nano' if api_key_available else 'test')
                print(f"✅ Proposed file name: {os.path.basename(filename)}")
            except Exception as e:
                print(f"❌ Error creating filename: {e}")
                # Create simple filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                models_dir = os.path.join(script_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                filename = os.path.join(models_dir, f"energy_model_{timestamp}.xml")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            models_dir = os.path.join(script_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            filename = os.path.join(models_dir, f"energy_model_{timestamp}.xml")
        
        # Load PLEXOS data sheets
        print("🏗️ Building comprehensive PLEXOS model...")
        try:
            import pandas as pd
            plexos_prompt_sheet = pd.read_excel(excel_path, sheet_name=None)
            print(f"✅ {list(plexos_prompt_sheet.keys())}")
            
            # Create database connection
            db = pbf.load_plexos_xml(source_file=filename, blank=True)
            
            # Process the base model task using the full PLEXOS pipeline
            if "Base Model Task" in task_list and task_list["Base Model Task"]:
                base_task = task_list["Base Model Task"]
                print(f"✅ Created comprehensive PLEXOS model: {filename}")
                
                # Use the full PLEXOS processing pipeline
                pbf.process_base_model_task(db, plexos_prompt_sheet, base_task, model='gpt-4.1-nano' if api_key_available else 'test')
                
                print("✅ Successfully created comprehensive PLEXOS model")
                
                # Extract model details
                location_name = countries[0] if countries else (location or 'Unknown')
                generation_type = generation or 'solar'  # Default based on common usage
                energy_carrier = energy_carrier or 'electricity'
                
                result = {
                    "status": "success",
                    "message": f"Created {generation_type} {energy_carrier} model for {location_name}",
                    "file": filename,
                    "location": location_name,
                    "generation_type": generation_type,
                    "energy_carrier": energy_carrier,
                    "countries": countries,
                    "model_type": "comprehensive_plexos"
                }
                
                print(f"📊 Model details: {len(countries)} countries, {generation_type} generation")
                
            else:
                result = {
                    "status": "error", 
                    "message": "No base model task identified"
                }
        
        except Exception as e:
            print(f"❌ Error during PLEXOS model creation: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to simple model creation
            return process_emil_request(kb, prompt, location, generation, energy_carrier, **kwargs)
        
        print("✅ Enhanced PLEXOS processing completed")
        
        # Store results in knowledge base
        kb.set_item("emil_result", result)
        kb.set_item("latest_model_details", result)
        if 'file' in result:
            kb.set_item("latest_model_file", result['file'])
        
        print("📊 Comprehensive PLEXOS model information stored in knowledge base")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in enhanced Emil processing: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fall back to the original simple processing
        return process_emil_request(kb, prompt, location, generation, energy_carrier, **kwargs)


# In src/core/functions_registery.py, update the process_emil_request_enhanced function:

@log_function_call
def process_emil_request_enhanced(kb: KnowledgeBase, prompt: str = None, location=None, generation=None, energy_carrier=None, **kwargs):
    """
    Enhanced Emil request processor with proper multi-country support.
    """
    print("\n" + "="*50)
    print("🚀 ENHANCED PLEXOS MODEL BUILDING STARTED")
    print("="*50)
    print(f"📝 Prompt: {prompt}")
    print(f"📍 Location: {location}")
    print(f"⚡ Generation: {generation}")
    print(f"🔋 Energy Carrier: {energy_carrier}")
    print("="*50 + "\n")
    
    if not PLEXOS_BASE_AVAILABLE:
        print("❌ PLEXOS base model not available, falling back to simple model creation")
        return process_emil_request(kb, prompt, location, generation, energy_carrier, **kwargs)
    
    # Ensure prompt is set
    if prompt is None:
        prompt = kwargs.get('prompt', 'Create an energy model')
    
    print(f"🔧 Using enhanced PLEXOS model building functionality")
    
    try:
        # Extract parameters from the prompt using enhanced extraction
        extracted_params = extract_model_parameters(prompt)
        
        # FIXED: Handle multiple countries properly
        final_locations = []
        if location:
            # Handle comma-separated locations
            if isinstance(location, str):
                final_locations = [loc.strip().capitalize() for loc in location.split(',')]
            else:
                final_locations = [location]
        elif extracted_params['locations'] and extracted_params['locations'][0] != "Unknown":
            final_locations = extracted_params['locations']
        else:
            return {
                "status": "error", 
                "message": "Location is required for energy modeling"
            }
        
        # Handle generation type
        if generation:
            final_generation = generation
        elif extracted_params['generation_types']:
            final_generation = extracted_params['generation_types'][0]
        else:
            return {
                "status": "error",
                "message": "Generation type is required"
            }
        
        # Handle energy carrier
        if energy_carrier:
            final_energy_carrier = energy_carrier
        elif extracted_params['energy_carriers']:
            final_energy_carrier = extracted_params['energy_carriers'][0]
        else:
            final_energy_carrier = "electricity"
        
        print(f"🎯 Final parameters:")
        print(f"   Locations: {final_locations}")
        print(f"   Generation: {final_generation}")
        print(f"   Energy Carrier: {final_energy_carrier}")
        
        # Create appropriate model based on number of locations
        if len(final_locations) == 1:
            print(f"🏗️ Creating single-location model for {final_locations[0]}")
            result = create_single_location_model(kb, final_locations[0], final_generation, final_energy_carrier)
        else:
            print(f"🏗️ Creating multi-location model for {len(final_locations)} countries")
            result = create_multi_location_model(final_locations, final_generation, final_energy_carrier)
        
        # Store the result in the knowledge base
        kb.set_item("emil_result", result)
        kb.set_item("latest_model_details", result)
        if 'file' in result:
            kb.set_item("latest_model_file", result['file'])
        
        # Store individual location details for context
        if len(final_locations) == 1:
            kb.set_item("latest_model_location", final_locations[0])
        else:
            kb.set_item("latest_model_location", ", ".join(final_locations))
            
        kb.set_item("latest_model_generation_type", final_generation)
        kb.set_item("latest_model_energy_carrier", final_energy_carrier)
        
        print(f"✅ Model creation completed successfully")
        print(f"📁 Model file: {result.get('file', 'Not specified')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in enhanced Emil processing: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fall back to the original simple processing
        return process_emil_request(kb, prompt, location, generation, energy_carrier, **kwargs)


# Also update the extract_model_parameters function to handle multiple countries better:

@log_function_call
def extract_model_parameters(prompt):
    """
    Enhanced model parameter extraction with better multi-country support.
    """
    import re
    print("🔍 Extracting model parameters from prompt...")
    prompt_lower = prompt.lower()
    params = {"locations": [], "generation_types": [], "energy_carriers": [], "model_type": "single"}
    
    # Enhanced location extraction with better patterns
    found_locations = []
    
    # First check for conjunction patterns indicating multiple countries
    multi_country_patterns = [
        r'for\s+([^.]+?)\s*,\s*([^.]+?)\s+and\s+([^.]+?)(?:\s|$)',  # "for spain, greece and denmark"
        r'in\s+([^.]+?)\s*,\s*([^.]+?)\s+and\s+([^.]+?)(?:\s|$)',   # "in spain, greece and denmark"
        r'([a-zA-Z]+)\s*,\s*([a-zA-Z]+)\s+and\s+([a-zA-Z]+)',      # "spain, greece and denmark"
    ]
    
    for pattern in multi_country_patterns:
        matches = re.findall(pattern, prompt_lower)
        if matches:
            for match in matches:
                for country in match:
                    country = country.strip()
                    # Check if it's a valid country name
                    for loc in LOCATIONS:
                        if country == loc.lower() or country in loc.lower():
                            found_locations.append(loc)
                            print(f"🌍 Found country in multi-pattern: {loc}")
            break  # Stop after first successful multi-country pattern
    
    # If no multi-country pattern found, check individual locations
    if not found_locations:
        for loc in LOCATIONS:
            patterns = [
                f" for {loc.lower()}",
                f" in {loc.lower()}",
                f"{loc.lower()} model",
                f"model.*{loc.lower()}",
                f"{loc.lower()}.*model",
            ]
            
            if any(re.search(pattern, prompt_lower) for pattern in patterns):
                found_locations.append(loc)
                print(f"🌍 Found country: {loc}")
    
    params["locations"] = list(set(found_locations))  # Remove duplicates
    
    # Enhanced generation type extraction
    found_gen_types = []
    generation_patterns = {
        "wind": [
            r"build.*wind.*model",
            r"create.*wind.*model", 
            r"wind.*model.*for",
            r"a wind model",
            r"wind power",
            r"wind generation",
            r"wind energy"
        ],
        "solar": [
            r"build.*solar.*model",
            r"create.*solar.*model",
            r"solar.*model.*for",
            r"a solar model",
            r"solar power",
            r"solar generation",
            r"solar pv"
        ],
        "hydro": [
            r"build.*hydro.*model",
            r"create.*hydro.*model",
            r"hydro.*model.*for",
            r"a hydro model",
            r"hydro power",
            r"hydroelectric"
        ]
    }
    
    for gen_type, patterns in generation_patterns.items():
        if any(re.search(pattern, prompt_lower) for pattern in patterns):
            found_gen_types.append(gen_type)
            print(f"⚡ Found generation type: {gen_type}")
            break  # Take first match
    
    params["generation_types"] = found_gen_types
    
    # Extract energy carriers
    carriers = ["electricity", "hydrogen", "methane"]
    found_carriers = []
    for carrier in carriers:
        if carrier in prompt_lower:
            found_carriers.append(carrier)
    
    params["energy_carriers"] = found_carriers or ["electricity"]
    
    # Set location default only if none found
    if not params["locations"]:
        params["locations"] = ["Unknown"]
    
    # Determine model type
    if len(params["locations"]) > 1:
        params["model_type"] = "multi"
    
    print(f"📊 Extracted parameters: {params}")
    return params



# ----------------------------------------------------------------------
# Initialize Function Map Loader
# ----------------------------------------------------------------------

# Create a function map loader instance
function_loader = FunctionMapLoader()

function_loader.register_functions({
    "build_plexos_model": build_plexos_model,
    "run_plexos_model": run_plexos_model,
    "analyze_results": analyze_results,  # Make sure this is registered
    "write_report": write_report,
    "generate_python_script": generate_python_script,
    "extract_model_parameters": extract_model_parameters,
    "create_single_location_model": create_single_location_model,
    "create_simple_xml": create_simple_xml,
    "create_multi_location_model": create_multi_location_model,
    "create_simple_multi_location_xml": create_simple_multi_location_xml,
    "create_comprehensive_model": create_comprehensive_model,
    "create_simple_comprehensive_xml": create_simple_comprehensive_xml,
    "process_emil_request": process_emil_request_enhanced,  # ⭐ Use enhanced version
    "process_emil_request_simple": process_emil_request     # Keep simple version as backup
})

# Load function maps from CSV files
NOVA_FUNCTIONS = function_loader.load_function_map("Nova")
EMIL_FUNCTIONS = function_loader.load_function_map("Emil")
IVAN_FUNCTIONS = function_loader.load_function_map("Ivan")
LOLA_FUNCTIONS = function_loader.load_function_map("Lola")

# Fallback to hardcoded dictionaries if CSV loading fails
if not NOVA_FUNCTIONS:
    NOVA_FUNCTIONS = {}

if not EMIL_FUNCTIONS:
    EMIL_FUNCTIONS = {
        "build_plexos_model": build_plexos_model,
        "run_plexos_model": run_plexos_model,
        "analyze_results": analyze_results,
        "process_emil_request": process_emil_request_enhanced,  # ⭐ Use enhanced version
        "extract_model_parameters": extract_model_parameters,
        "create_single_location_model": create_single_location_model, 
        "create_simple_xml": create_simple_xml,
        "create_multi_location_model": create_multi_location_model,
        "create_simple_multi_location_xml": create_simple_multi_location_xml,
        "create_comprehensive_model": create_comprehensive_model,
        "create_simple_comprehensive_xml": create_simple_comprehensive_xml
    }

if not IVAN_FUNCTIONS:
    IVAN_FUNCTIONS = {
        "generate_python_script": generate_python_script,
    }

if not LOLA_FUNCTIONS:
    LOLA_FUNCTIONS = {
        "write_report": write_report,
    }
    