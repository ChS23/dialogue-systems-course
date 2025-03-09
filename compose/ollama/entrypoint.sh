#!/bin/bash
set -e

ollama serve &
sleep 5
ollama pull nomic-embed-text
kill %1
exec ollama serve
