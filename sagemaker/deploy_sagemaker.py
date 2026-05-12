import os
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# -----------------------------
# Configuration (use env vars)
# -----------------------------
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "mlops-predictive-maintenance-endpoint")
MODEL_NAME = os.environ.get("MODEL_NAME", "mlops-predictive-maintenance-model")

S3_MODEL_PATH = f"s3://{S3_BUCKET}/models/model.tar.gz"


def main():
    if not S3_BUCKET:
        raise ValueError("❌ Missing environment variable: S3_BUCKET_NAME")

    if not ROLE_ARN:
        raise ValueError("❌ Missing environment variable: SAGEMAKER_ROLE_ARN")

    print("---------------------------------------------------")
    print("Deploying model to SageMaker...")
    print("Region:", AWS_REGION)
    print("Bucket:", S3_BUCKET)
    print("Model Path:", S3_MODEL_PATH)
    print("Endpoint:", ENDPOINT_NAME)
    print("---------------------------------------------------")

    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    model = SKLearnModel(
        name=MODEL_NAME,
        model_data=S3_MODEL_PATH,
        role=ROLE_ARN,
        entry_point="inference.py",
        source_dir="sagemaker",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session,
    )

    # Deploy (this automatically creates or updates endpoint)
    predictor = model.deploy(
        endpoint_name=ENDPOINT_NAME,
        initial_instance_count=1,
        instance_type="ml.m5.large",
        update_endpoint=True
    )

    print("✅ Deployment completed successfully!")
    print("Endpoint Name:", ENDPOINT_NAME)


if __name__ == "__main__":
    main()