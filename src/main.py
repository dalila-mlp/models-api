import base64
import os
from typing import List
import numpy as np
import requests
import json
import uuid

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.train_model import main


ap = Path(__file__).parent.parent.resolve()
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    model_id: str
    datafile_id: str
    target_column: str
    features : List[str]
    test_size: float
    model_type: str
    parameters: dict

class PredictRequest(BaseModel):
    model_id: str
    datafile_id: str
    features: List[str]
    model_type: str


def get_github_token() -> str:
    """ Retrieve GitHub token from environment variables."""
    return os.getenv("GITHUB_TOKEN")


def fetch_model_script(model_id: str, github_token: str) -> str:
    """ Fetch the model script from GitHub based on the model ID."""

    if ((response := requests.get(
        f"https://api.github.com/repos/dalila-mlp/models/contents/{model_id}.py",
        headers={"Authorization": f"token {github_token}"},
    )).status_code == 200):
        return base64.b64decode(response.json()['content']).decode('utf-8')

    raise HTTPException(status_code=response.status_code, detail="Model file not found on GitHub")

def fetch_model_save(model_id: str, github_token: str, model_type: str) -> str:
    """ Fetch the model script from GitHub based on the model ID."""
    if model_type == 'Tf':
        if ((response := requests.get(
            f"https://api.github.com/repos/dalila-mlp/models-trained/contents/{model_id}.keras",
            headers={"Authorization": f"token {github_token}"},
        )).status_code == 200):
            response = requests.get(response.json()['download_url'])
            return response.content
    else :
        if ((response := requests.get(
            f"https://api.github.com/repos/dalila-mlp/models-trained/contents/{model_id}.pkl",
            headers={"Authorization": f"token {github_token}"},
        )).status_code == 200):
            response = requests.get(response.json()['download_url'])
            return response.content

    raise HTTPException(status_code=response.status_code, detail="Model file not found on GitHub")



def fetch_dataset(dataset_id: str, github_token: str) -> str:
    """Fetch the dataset file from GitHub based on the dataset ID."""

    url = f"https://api.github.com/repos/dalila-mlp/datafiles/contents/{dataset_id}.csv"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        response = requests.get(response.json()['download_url'])
        temp_dataset_path = f'{ap}/dataset/temp_{dataset_id}.csv'

        with open(temp_dataset_path, 'wb') as file:
            file.write(response.content)

        return temp_dataset_path
    else:
        raise HTTPException(status_code=response.status_code, detail="Dataset file not found on GitHub")


def dynamic_import(script_content, test_size, model_id, dataset_content, target_column,features, github_token, request_model_type, parameters):
    """ Dynamically import and execute training from the fetched script. """
    # Save the fetched script content to a temporary Python file
    # a modifier
    temp_script_path = f'{ap}/models/temp_{model_id}.py'
    with open(temp_script_path, 'w') as file:
        file.write(script_content)
    save_model_path = f'{ap}/models/{model_id}.pkl'

    # Execute the training process
    plot_ids, metrics, model_type = main("train",temp_script_path, dataset_content, target_column,features, test_size, model_id, request_model_type, parameters)

    if model_type == 'Tf':
        save_model_path = f'{ap}/models/{model_id}.keras'
    else :
        save_model_path = f'{ap}/models/{model_id}.pkl'

    # Upload plots to GitHub
    for plot_id in plot_ids:
        plot_filename = f"{plot_id}.png"
        upload_plot_to_github(plot_filename, github_token)
    model_save_id = upload_model_to_github(model_id, model_type, github_token)

    # Clean up: Remove the temporary script file and plot files
    os.remove(temp_script_path)
    for plot_id in plot_ids:
        plot_filename = f"{ap}/charts/{plot_id}.png"
        os.remove(plot_filename)
    os.remove(save_model_path)

    return plot_ids, metrics, model_save_id, model_type

def dynamic_import_predict(script_content, model_id, dataset_content,features, github_token, model_type):
    """ Dynamically import and execute training from the fetched script. """
    # Save the fetched script content to a temporary Python file
    if model_type == "Tf":
        temp_script_path = f'{ap}/models/temp_{model_id}.keras'
    else:
        temp_script_path = f'{ap}/models/temp_{model_id}.pkl'
    with open(temp_script_path, 'wb') as file:
        file.write(script_content)
    # Execute the training process
    result = main("predict",temp_script_path, dataset_content, None, features, None, model_id, None, None)

    # # Clean up: Remove the temporary script file and plot files
    os.remove(temp_script_path)

    return result

def upload_plot_to_github(plot_filename, github_token):
    """ Upload the plot file to a GitHub repository. """
    with open(f"{ap}/charts/{plot_filename}", 'rb') as file:
        content = base64.b64encode(file.read()).decode('utf-8')

    url = f"https://api.github.com/repos/dalila-mlp/models-chart/contents/{plot_filename}"
    headers = {"Authorization": f"token {github_token}"}
    data = {
        "message": f"Add plot {plot_filename}",
        "content": content
    }
    response = requests.put(url, headers=headers, json=data)
    if response.status_code not in (201, 200):
        raise HTTPException(status_code=response.status_code, detail="Failed to upload plot to GitHub")

def upload_model_to_github(model_id, model_type, github_token):
    """ Upload the model file to a GitHub repository. """
    
    file_name = f'{uuid.uuid4()}'
    try:
        if model_type == 'Tf':
            file_path = f"{ap}/models/{model_id}.keras"
            url = f"https://api.github.com/repos/dalila-mlp/models-trained/contents/{file_name}.keras"
        else:
            file_path = f"{ap}/models/{model_id}.pkl"
            url = f"https://api.github.com/repos/dalila-mlp/models-trained/contents/{file_name}.pkl"
        
        with open(file_path, 'rb') as file:
            content = base64.b64encode(file.read()).decode('utf-8')
        
        headers = {"Authorization": f"token {github_token}"}
        data = {
            "message": f"Add model {file_name}",
            "content": content
        }
        
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        
        if response.status_code in (201, 200):
            print(f"Successfully uploaded {model_id} to GitHub.")
        else:
            print(f"Failed to upload {model_id} to GitHub. Status code: {response.status_code}")
        return file_name
    except FileNotFoundError:
        raise Exception(f"File {file_path} not found.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to upload model to GitHub: {e}")

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Convert numpy types to Python scalars
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj

@app.post("/train")
def train_model(request: TrainRequest, github_token: str = Depends(get_github_token)):
    if not github_token:
        raise HTTPException(status_code=500, detail="GitHub token not configured")

    try:
        script_content = fetch_model_script(request.model_id, github_token)
        dataset_temp_path = fetch_dataset(request.datafile_id, github_token)
        plot_ids, metrics, model_save_id, response_model_type = dynamic_import(
            script_content,
            request.test_size,
            request.model_id,
            dataset_temp_path,
            request.target_column, 
            request.features, 
            github_token,
            request.model_type,
            request.parameters
        )
        
        #remove the dataset temp file
        os.remove(f"{ap}/dataset/temp_{request.datafile_id}.csv")
        
        # Convert all numpy data types to native Python types for JSON serialization
        metrics = convert_numpy(metrics)
        return {
            "metrics": metrics,
            "plot_id_list": plot_ids,
            "model_save_id": model_save_id,
            "model_type": response_model_type,
        }
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    
@app.post("/predict")
def predict_model(request: PredictRequest, github_token: str = Depends(get_github_token)):
    if not github_token:
        raise HTTPException(status_code=500, detail="GitHub token not configured")

    try:
        script_content = fetch_model_save(request.model_id, github_token, request.model_type)
        dataset_temp_path = fetch_dataset(request.datafile_id, github_token)
        result = dynamic_import_predict(script_content, request.model_id, dataset_temp_path, request.features, github_token, request.model_type)
        #remove the dataset temp file
        os.remove(f"{ap}/dataset/temp_{request.datafile_id}.csv")
        
        # Convert all numpy data types to native Python types for JSON serialization
        # metrics = convert_numpy(metrics)
        return result
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
