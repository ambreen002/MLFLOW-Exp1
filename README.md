# MLFlow LLM Evaluation Example

This repository demonstrates how to evaluate a Generative AI model using MLFlow with a practical example of integrating with OpenAI's GPT-4 model. The evaluation uses various metrics to assess the performance of the model on predefined question-answering tasks.

## Requirements

Ensure you have the following dependencies specified in your `Pipfile`:

```toml
[packages]
langchain = "*"
faiss-cpu = "*"
langchain-community = "*"
langchain-openai = "*"
openai = "*"
tiktoken = "*"
load-dotenv = "*"
mlflow = "*"
dagshub = "*"
evaluate = "*"
transformers = "*"
textstat = "*"

[dev-packages]

[requires]
python_version = "3.11"
```

## Setup
Initialize DagsHub: Initialize the DagsHub repository for MLFlow tracking.

python
Copy code
import dagshub
dagshub.init("MLFlow-test1", "mailambreen", mlflow=True)
Set MLFlow Tracking URI: Set the tracking URI to your DagsHub repository.

```
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/mailambreen/MLFLOW-Exp1.mlflow")
```

## Data Preparation
Prepare the evaluation data in a Pandas DataFrame format. The example below shows the input questions and their corresponding ground truth answers.

```
import pandas as pd

eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)
```

## Experiment Setup
Set Experiment: Define the MLFlow experiment for LLM evaluation.

```
mlflow.set_experiment("LLM Evaluation")
```

Start Run: Start an MLFlow run to log the model and perform evaluation.
```
with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    
    # Wrap "gpt-4" as an MLFlow model.
    logged_model_info = mlflow.openai.log_model(
        model="gpt-4",
        task=openai.chat.completions,
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )

    # Use predefined question-answering metrics to evaluate our model.
    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        targets="ground_truth",
        model_type="question-answering",
        extra_metrics=[
            mlflow.metrics.toxicity(),
            mlflow.metrics.latency(),
            mlflow.metrics.genai.answer_similarity()
        ]
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    # Evaluation result for each data record is available in `results.tables`.
    eval_table = results.tables["eval_results_table"]
    df = pd.DataFrame(eval_table)
    df.to_csv('eval.csv')
    print(f"See evaluation table below: \n{eval_table}")
```

## Output
### Metrics: 
The aggregated evaluation results are printed, providing insight into the model's performance on various metrics.
### Evaluation Table: 
The detailed evaluation results for each data record are saved to a CSV file (eval.csv).
### DAGSHUB Experiment Dashboards Generated:
<img width="1471" alt="image" src="https://github.com/ambreen002/MLFLOW-Exp1/assets/36915142/1a89572b-9199-4e1b-9576-e26db01f38f8">
<img width="981" alt="image" src="https://github.com/ambreen002/MLFLOW-Exp1/assets/36915142/9e2015aa-b8e5-46ce-b449-cc4b08bc765e">
<img width="792" alt="image" src="https://github.com/ambreen002/MLFLOW-Exp1/assets/36915142/34c1d5a3-657f-48d6-8498-bb12e1f582a6">

