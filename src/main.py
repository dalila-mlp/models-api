import base64
import os
from typing import List
import numpy as np
import requests

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


def fetch_dataset(dataset_id: str, github_token: str) -> str:
    """Fetch the dataset file from GitHub based on the dataset ID."""

    url = f"https://api.github.com/repos/dalila-mlp/datafiles/contents/{dataset_id}.csv"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    print(response)

    if response.status_code == 200:
        content = base64.b64decode(response.json()['content'])
        temp_dataset_path = f'{ap}/dataset/temp_{dataset_id}.csv'

        with open(temp_dataset_path, 'wb') as file:
            file.write(content)

        return temp_dataset_path
    else:
        raise HTTPException(status_code=response.status_code, detail="Dataset file not found on GitHub")


def dynamic_import(script_content, test_size, model_id, dataset_content, target_column,features, github_token):
    """ Dynamically import and execute training from the fetched script. """
    # Save the fetched script content to a temporary Python file
    # a modifier
    temp_script_path = f'{ap}/models/temp_{model_id}.py'
    with open(temp_script_path, 'w') as file:
        file.write(script_content)

    # Execute the training process
    plot_ids, metrics = main(temp_script_path, dataset_content, target_column,features, test_size)

    # Upload plots to GitHub
    for plot_id in plot_ids:
        plot_filename = f"{plot_id}.png"
        upload_plot_to_github(plot_filename, github_token)

    # Clean up: Remove the temporary script file and plot files
    os.remove(temp_script_path)
    for plot_id in plot_ids:
        plot_filename = f"{ap}/charts/{plot_id}.png"
        os.remove(plot_filename)

    return plot_ids, metrics

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
        plot_ids, metrics = dynamic_import(script_content, request.test_size, request.model_id, dataset_temp_path, request.target_column, request.features, github_token)
        
        #remove the dataset temp file
        os.remove(f"{ap}/dataset/temp_{request.datafile_id}.csv")
        
        # Convert all numpy data types to native Python types for JSON serialization
        metrics = convert_numpy(metrics)
        return {
            "metrics": metrics,
            "plot_id_list": plot_ids
        }
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
