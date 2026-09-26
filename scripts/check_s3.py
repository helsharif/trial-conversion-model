import boto3
from dotenv import load_dotenv

load_dotenv()
boto3.client("s3").head_bucket(Bucket="futureproofds-trial-conversion-artifacts-bucket-001")
print("credentials work and bucket is accessible.")
