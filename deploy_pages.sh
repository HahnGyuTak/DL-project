#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${1:-model-dock}"
ASSET_DIR="${2:-web}"

if ! command -v npx >/dev/null 2>&1; then
  echo "npx not found. Install Node.js first." >&2
  exit 1
fi

if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
  echo "CLOUDFLARE_API_TOKEN is not set." >&2
  echo "Create token: https://developers.cloudflare.com/fundamentals/api/get-started/create-token/" >&2
  exit 1
fi

echo "Deploying ${ASSET_DIR} to Cloudflare Pages project: ${PROJECT_NAME}"
npx --yes wrangler pages deploy "${ASSET_DIR}" --project-name "${PROJECT_NAME}"
