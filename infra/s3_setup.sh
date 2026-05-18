#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: ./s3_setup.sh <bucket-name> [region]"
  exit 1
fi

BUCKET_NAME="$1"
REGION="${2:-us-east-1}"

aws s3api create-bucket \
  --bucket "${BUCKET_NAME}" \
  --region "${REGION}" \
  --create-bucket-configuration "LocationConstraint=${REGION}" || true

echo "Bucket ready: s3://${BUCKET_NAME}"

