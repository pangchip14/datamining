#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw
curl -L -C - \
  "https://www.thedatum.org/datasets/TSB-UAD-Public-v2.zip" \
  -o "data/raw/TSB-UAD-Public-v2.zip"

mkdir -p data/raw/TSB-UAD-Public-v2
unzip -n "data/raw/TSB-UAD-Public-v2.zip" -d "data/raw/TSB-UAD-Public-v2"
