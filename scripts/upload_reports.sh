#!/bin/bash

export AWS_S3_HOST=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioKey
export AWS_SECRET_ACCESS_KEY=minioSecret
export AWS_REGION=eu-west-1

###################################################################

PROJECT="neuro-evolution"
ALGORITHM_VERSION="bayes-neat"


DATASET="breast_cancer"
CORRELATION_ID="bayesian_neat_ft_20_breast_cancer"
NEW_CORRELATION_ID="test_upload"

BASE_DIR="/home/alberto/Downloads/"

pipenv run python experiments/reporting/migration/migration.py upload_reports \
        --project=$PROJECT \
        --algorithm_version=$ALGORITHM_VERSION \
        --dataset=$DATASET \
        --correlation_id=$CORRELATION_ID \
        --new_correlation_id=$NEW_CORRELATION_ID \
        --base_dir=$BASE_DIR
