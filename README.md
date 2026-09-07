# Azure-Native MLOps Starter

An end-to-end starter for fine-tuning a small Hugging Face classifier with PEFT/LoRA, tracking runs in Azure Machine Learning (AML), and serving the registered model through an AML Managed Online Endpoint.

## Architecture

- **Data:** Azure Blob Storage, exposed to jobs as AML Data assets. DVC can be added when Git-backed data lineage is required.
- **Training:** Hugging Face Transformers + PEFT/LoRA in an AML command job.
- **Tracking and registry:** AML's native MLflow integration and model assets.
- **Serving:** FastAPI scoring script behind an AML Managed Online Endpoint.
- **Automation:** GitHub Actions authenticates with Azure via OIDC, submits training, and deploys the endpoint.
- **Monitoring:** Azure Monitor/Application Insights for the endpoint; add Evidently drift checks in the retraining pipeline.

## Repository layout

```text
.
├── .github/workflows/ci-cd.yml
├── aml/
│   ├── compute.yml
│   ├── endpoint.yml
│   ├── job.yml
│   └── environment.yml
├── data/raw/.gitkeep
├── src/
│   ├── train.py
│   └── score.py
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Quick start

1. Create an AML workspace and storage account, then set the values in `.env` from `.env.example`.
2. Install the local dependencies with `python -m pip install -r requirements.txt`.
3. Put a CSV at `data/raw/tickets.csv` with `text` and `label` columns. Run `python src/train.py` to iterate locally.
4. Submit the cloud job:

	```bash
	az ml job create --file aml/job.yml --resource-group "$AZURE_RESOURCE_GROUP" --workspace-name "$AZURE_ML_WORKSPACE"
	```

5. Register the resulting model in AML Studio, update `MODEL_NAME` in `aml/endpoint.yml`, and deploy it:

	```bash
	az ml online-endpoint create --file aml/endpoint.yml
	az ml online-deployment create --file aml/deployment.yml --all-traffic
	```

6. Add the GitHub repository variables and secrets documented in `.github/workflows/ci-cd.yml` before enabling CI/CD.

The included files are deliberately small scaffolds. Replace the sample dataset and labels, then add evaluation gates and approval before production promotion.