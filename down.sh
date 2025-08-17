#!/bin/bash

docker compose -f docker-compose.database.yaml down
docker compose -f docker-compose.api.yaml down
docker compose -f docker-compose.batch.yaml down
docker compose -f docker-compose.inference.yaml down
