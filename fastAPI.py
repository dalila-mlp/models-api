from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import requests
import os
import base64
from train_model import main
app = FastAPI()

class TrainRequest(BaseModel):
    transaction_id: str
    model_id: str
    test_size: float

def get_github_token():
    """ Retrieve GitHub token from environment variables. """
    return os.getenv("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")

def fetch_model_script(model_id: str, github_token: str):
    """ Fetch the model script from GitHub based on the model ID. """
    url = f"https://api.github.com/repos/dalila-mlp/models/contents/{model_id}.py"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        script_content = base64.b64decode(content['content']).decode('utf-8')
        return script_content
    else:
        raise HTTPException(status_code=response.status_code, detail="Model file not found on GitHub")

def dynamic_import(script_content, test_size, model_id, transaction_id, github_token):
    """ Dynamically import and execute training from the fetched script. """
    # Save the fetched script content to a temporary Python file
    temp_script_path = f'temp_{model_id}.py'
    with open(temp_script_path, 'w') as file:
        file.write(script_content)

    # Execute the training process
    plot_ids, metrics = main(temp_script_path, test_size, transaction_id)

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
async def train_model(request: TrainRequest, github_token: str = Depends(get_github_token)):
    if not github_token:
        raise HTTPException(status_code=500, detail="GitHub token not configured")

    try:
        script_content = fetch_model_script(request.model_id, github_token)
        plot_ids, metrics = dynamic_import(script_content, request.test_size, request.model_id, request.transaction_id)
        return {
            "transaction_id": request.transaction_id,
            "metrics": metrics,
            "plot_id_list" : plot_ids
        }
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
