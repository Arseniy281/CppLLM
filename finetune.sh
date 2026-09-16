#!/bin/bash

set -e

echo "========================================"
echo "BOOK CHARACTER AI - FINETUNE"
echo "========================================"
echo

echo
echo "[1/2] finetuning..."
./build/finetune

echo
echo "[2/2] Generating answers..."
./build/generate

echo
echo "========================================"
echo "FINETUNE COMPLETE"
echo "========================================"