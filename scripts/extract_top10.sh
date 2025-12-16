#!/bin/bash
# Extract concepts from top 10 most useful sources using Ollama
# Run with: nohup ./scripts/extract_top10.sh > /tmp/extract_top10.log 2>&1 &

set -e

SOURCES=(
  "c48dc7ff-665c-4daf-a60d-ce9068a9fdf3:Koller Friedman PGM:3171"
  "24af0a79-be81-4c29-8426-fcc97ef1ec81:Econometric Analysis:2702"
  "1d3a81a8-cec4-401c-9fe1-39aa8374020d:Causal Mediation:2005"
  "fd238b2c-bf52-4e36-8e49-0b0df0754a25:Box Jenkins Time Series:1470"
  "a2e2d729-3730-4120-b155-28dfe5a7d0a4:The Effect:1422"
  "f6596a6e-a1be-438b-a761-1b82cb94c8dd:Shumway Stoffer Time Series:1337"
  "2640f892-a2eb-4aad-9622-d825206e9e51:Manning Causal AI:1069"
  "1ea17aef-1b16-442f-84f0-9d78df3cdb42:Applied Bayesian Causal:934"
  "69fdbfbe-20f4-4342-a050-4f8fc9e81284:Causal Inference Data Science:705"
)

TOTAL_CHUNKS=15815
PROCESSED=0

echo "========================================"
echo "TOP 10 SOURCE EXTRACTION - OLLAMA"
echo "Total chunks: $TOTAL_CHUNKS"
echo "Started: $(date)"
echo "========================================"

for source in "${SOURCES[@]}"; do
  IFS=':' read -r id name chunks <<< "$source"

  echo ""
  echo "========================================"
  echo "Processing: $name"
  echo "Source ID: $id"
  echo "Chunks: $chunks"
  echo "Progress: $PROCESSED / $TOTAL_CHUNKS"
  echo "Started: $(date)"
  echo "========================================"

  python3 scripts/extract_concepts.py \
    --backend ollama \
    --model llama3.1:8b \
    --source-id "$id" \
    --skip-backup \
    --no-neo4j \
    --concurrency 1

  PROCESSED=$((PROCESSED + chunks))

  echo "Completed: $name at $(date)"
  echo "Total progress: $PROCESSED / $TOTAL_CHUNKS chunks"
done

echo ""
echo "========================================"
echo "ALL SOURCES COMPLETE"
echo "Finished: $(date)"
echo "========================================"
