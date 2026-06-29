#!/bin/bash

# Get the folder path and model choice from the user's input
FOLDER_PATH=""
MODEL_CHOICE=""

SUPPORTED_MODELS=("gemini" "anthropic" "openai" "vllm_models")

is_model_choice() {
  local val="$1"
  for m in "${SUPPORTED_MODELS[@]}"; do
    if [ "$val" == "$m" ]; then
      return 0
    fi
  done
  return 1
}

if [ $# -ge 2 ]; then
  FOLDER_PATH="$1"
  MODEL_CHOICE="$2"
elif [ $# -eq 1 ]; then
  if is_model_choice "$1"; then
    MODEL_CHOICE="$1"
  else
    FOLDER_PATH="$1"
  fi
fi

# Set default model choice to openai if not specified
MODEL_CHOICE="${MODEL_CHOICE:-openai}"

# Resolve target folders
TARGET_FOLDERS=()

if [ -n "$FOLDER_PATH" ] && [ -d "$FOLDER_PATH" ]; then
  TARGET_FOLDERS+=("$FOLDER_PATH")
else
  # If FOLDER_PATH was provided but is a model choice, set MODEL_CHOICE
  if [ -n "$FOLDER_PATH" ] && is_model_choice "$FOLDER_PATH"; then
    MODEL_CHOICE="$FOLDER_PATH"
  fi

  # Fallback to scanning the input/ directory
  INPUT_DIR="input"
  if [ -d "$INPUT_DIR" ]; then
    # Find all subdirectories in the input folder
    PACKAGES=($(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort))
    if [ ${#PACKAGES[@]} -gt 0 ]; then
      for pkg in "${PACKAGES[@]}"; do
        TARGET_FOLDERS+=("$pkg")
      done
      echo "No valid package path provided. Processing all packages in input folder: ${TARGET_FOLDERS[*]}"
    else
      echo "Error: No package path provided, and no packages found in '$INPUT_DIR' directory."
      echo "Please provide a folder path or upload a package to the '$INPUT_DIR' directory."
      exit 1
    fi
  else
    echo "Error: No package path provided, and '$INPUT_DIR' directory does not exist."
    echo "Please provide a folder path or create an '$INPUT_DIR' directory and upload a package."
    exit 1
  fi
fi

# Define the output directories locally
OUTPUT_DIR="output"
GRAPH_DIR="$OUTPUT_DIR/graph"
SUMMARIES_DIR="$OUTPUT_DIR/summaries"

# Create directories if they don't exist
mkdir -p "$GRAPH_DIR"
mkdir -p "$SUMMARIES_DIR"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for CURRENT_FOLDER in "${TARGET_FOLDERS[@]}"; do
  DIR_NAME=$(basename "$CURRENT_FOLDER")
  # Ensure CURRENT_FOLDER ends with a slash for downstream python scripts
  [[ "$CURRENT_FOLDER" != */ ]] && CURRENT_FOLDER="$CURRENT_FOLDER/"
  
  echo "============================================================="
  echo "Processing package: $DIR_NAME ($CURRENT_FOLDER)"
  echo "============================================================="

  # Define the output file path in the graph directory
  OUTPUT_FILE="$GRAPH_DIR/$DIR_NAME-file.json"

  # Check if the graph file already exists
  if [ -f "$OUTPUT_FILE" ]; then
    echo "Graph file '$OUTPUT_FILE' already exists. Skipping code2flow generation."
  else
    echo "Graph file not found. Running code2flow..."
    # python3 "$SCRIPT_DIR/code2flow_isolated.py" "$CURRENT_FOLDER" --output "$OUTPUT_FILE"
    python3 "$SCRIPT_DIR/code2flow_isolated.py" "$CURRENT_FOLDER" --output "$OUTPUT_FILE" --skip-parse-errors --language py --quiet
  fi

  python3 all_code_to_one.py "$CURRENT_FOLDER" "$MODEL_CHOICE"
  python3 main.py "$CURRENT_FOLDER" "$MODEL_CHOICE"
  python3 main2.py "$CURRENT_FOLDER" "$MODEL_CHOICE"
  python3 main3.py "$CURRENT_FOLDER" "$MODEL_CHOICE"
  python3 main4.py "$CURRENT_FOLDER" "$MODEL_CHOICE"
done