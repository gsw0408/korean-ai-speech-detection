#!/bin/sh
# Assets only. Run only after explicit approval to deploy and use AWS resources.
set -eu
if [ "$#" -ne 1 ]; then
    echo "Usage: IMAGE_REPOSITORY=private-ECR/repository ./scripts/deploy.sh approved-tag" >&2
    exit 2
fi
: "${IMAGE_REPOSITORY:?Set the private ECR repository}"
case "$1" in *[!a-zA-Z0-9_.-]*|'') echo "Invalid image tag" >&2; exit 2;; esac
cd "$(dirname "$0")/.."
tag=$1
new_image="$IMAGE_REPOSITORY:$tag"
state_file=.deployed-image
previous_image=
if [ -f "$state_file" ]; then previous_image=$(cat "$state_file"); fi
if [ -z "$previous_image" ]; then
    existing=$(IMAGE="$new_image" docker compose -f docker-compose.prod.yml ps -q service 2>/dev/null || true)
    if [ -n "$existing" ]; then previous_image=$(docker inspect --format '{{.Config.Image}}' "$existing"); fi
fi
# ECR authentication uses the instance role; no credentials are placed in this repository.
registry=${IMAGE_REPOSITORY%%/*}
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin "$registry"
docker pull "$new_image"
if IMAGE="$new_image" docker compose -f docker-compose.prod.yml up -d --wait --wait-timeout 120; then
    service_id=$(IMAGE="$new_image" docker compose -f docker-compose.prod.yml ps -q service)
    if docker exec "$service_id" python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=5)"; then
        printf '%s\n' "$new_image" > "$state_file.tmp"
        mv "$state_file.tmp" "$state_file"
        echo "Deployment healthy: $new_image"
        exit 0
    fi
fi
echo "New image failed health checks; restoring previous image" >&2
if [ -n "$previous_image" ]; then
    IMAGE="$previous_image" docker compose -f docker-compose.prod.yml up -d --wait --wait-timeout 120
else
    IMAGE="$new_image" docker compose -f docker-compose.prod.yml stop service
    echo "First deployment failed; no previous image exists" >&2
fi
exit 1
