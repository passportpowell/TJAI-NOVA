# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:22:38 2022

@author: ENTSOE
"""
import os
import subprocess as sp

def run_model(filename, modelname, version = 'PLEXOS 10.0'):
    # Get credentials from environment
    username = os.environ.get("PLEXOS_USERNAME", "")
    password = os.environ.get("PLEXOS_PASSWORD", "")
    
    if not username or not password:
        print("Username or password not set in environment variables.")
        return
    
    foldername = '.'
    plexospath = fr'C:\Program Files\Energy Exemplar\{version}'
    print('Running %s Model' % modelname)
    sp.call([
        os.path.join(plexospath, 'PLEXOS64.exe'),
        filename,
        r'\n',             # Automatically close the execution window
        r'\cu', username,
        r'\cp', password,
        r'\o', foldername,
        r'\m', modelname
    ])

def main(filename, modelname):
    run_model(filename, modelname, 'PLEXOS 10.0')
    print('Simulation Complete')
    
if __name__ == "__main__":
    filename = r'C:\TeraJoule\AI Assistants\Emil - AI Engineer\LLM Calls\functions\plexos_functions\PLEXOS_models\Benelux_Electricity_Methane_Model.xml'
    modelname = 'TJ Dispatch_Future_Nuclear+'
    run_model(filename, modelname, version = 'PLEXOS 10.0')
