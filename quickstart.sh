#!/bin/bash

set -e

echo "========================================"
echo "BOOK CHARACTER AI - QUICKSTART"
echo "========================================"
echo

echo "[1/4] Configuring project..."
cmake -S . -B build

echo
echo "[2/4] Building project..."
cmake --build build -j2

echo
echo "[3/4] Tokenizing and training..."
./build/tokenize
./build/train

echo
echo "[4/4] Generating answers..."
./build/generate

echo
echo "========================================"
echo "QUICKSTART COMPLETE"
echo "========================================"