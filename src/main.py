import base64
import os
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.train_model import main


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


def dynamic_import(script_content, test_size, model_id, github_token):
    """ Dynamically import and execute training from the fetched script. """
    # Save the fetched script content to a temporary Python file
    temp_script_path = f'temp_{model_id}.py'
    with open(temp_script_path, 'w') as file:
        file.write(script_content)

    # Execute the training process
    plot_ids, metrics = main(temp_script_path, test_size)

    # Upload plots to GitHub
    for plot_id in plot_ids:
        plot_filename = f"{plot_id}.png"
        upload_plot_to_github(plot_filename, github_token)

    # Clean up: Remove the temporary script file and plot files
    os.remove(temp_script_path)
    for plot_id in plot_ids:
        plot_filename = f"{plot_id}.png"
        os.remove(plot_filename)

    return plot_ids, metrics

def upload_plot_to_github(plot_filename, github_token):
    """ Upload the plot file to a GitHub repository. """
    with open(plot_filename, 'rb') as file:
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

@app.post("/train")
def train_model(request: TrainRequest, github_token: str = Depends(get_github_token)):
    if not github_token:
        raise HTTPException(status_code=500, detail="GitHub token not configured")

    try:
        script_content = fetch_model_script(request.model_id, github_token)
        plot_ids, metrics = dynamic_import(script_content, request.test_size, request.model_id, github_token)
        return {
            "metrics": metrics,
            "plot_id_list" : plot_ids
        }
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
