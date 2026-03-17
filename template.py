import os
from pathlib import Path

project_name = "src"

list_of_files = [
    # core src
    f"{project_name}/__init__.py",

    # components
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/dkt_trainer.py",
    f"{project_name}/components/rl_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",

    # configuration
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/db_connection.py",
    f"{project_name}/configuration/aws_connection.py",

    # cloud storage
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/cloud_storage/aws_storage.py",

    # data access
    f"{project_name}/data_access/__init__.py",
    f"{project_name}/data_access/attempt_data.py",
    f"{project_name}/data_access/question_data.py",

    # constants
    f"{project_name}/constants/__init__.py",

    # entity
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/db_models.py",
    f"{project_name}/entity/schemas.py",

    # exception
    f"{project_name}/exception/__init__.py",

    # logger
    f"{project_name}/logger/__init__.py",

    # ml models
    f"{project_name}/ml/__init__.py",
    f"{project_name}/ml/dkt_model.py",
    f"{project_name}/ml/rl_agent.py",
    f"{project_name}/ml/difficulty_selector.py",
    f"{project_name}/ml/nlp_embedder.py",

    # pipeline
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",

    # utils
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",

    # routes
    "routes/__init__.py",
    "routes/auth.py",
    "routes/questions.py",
    "routes/attempts.py",
    "routes/users.py",

    # root files
    "app.py",
    "demo.py",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",

    # config
    "config/config.yaml",
    "config/schema.yaml",
    "config/model.yaml",

    # notebook
    "notebook/eda.ipynb",

    # artifacts (kept empty, gitignored)
    "artifacts/.gitkeep",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")