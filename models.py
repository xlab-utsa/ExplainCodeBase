from google import genai
from anthropic import AnthropicVertex, AnthropicError
import os
import time
from openai import OpenAI

def load_env_file():
    # Look for .env in the current directory or the script's directory
    for path in [".env", os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
            break

load_env_file()

def resolve_args():
    import sys
    supported_models = ["gemini", "anthropic", "vllm_models", "openai"]
    package_path = None
    model_choice = "openai"

    if len(sys.argv) > 2:
        package_path = sys.argv[1]
        model_choice = sys.argv[2]
    elif len(sys.argv) == 2:
        if sys.argv[1] in supported_models:
            model_choice = sys.argv[1]
        else:
            package_path = sys.argv[1]

    if not package_path or not os.path.isdir(package_path):
        input_dir = "input"
        if os.path.exists(input_dir) and os.path.isdir(input_dir):
            subdirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
            if subdirs:
                package_path = sorted(subdirs)[0]
                print(f"No package path provided or invalid. Found package in input folder: {package_path}")
            else:
                print(f"Error: No package path provided, and no subdirectories found in '{input_dir}' folder.")
                sys.exit(1)
        else:
            print(f"Error: No package path provided, and '{input_dir}' directory does not exist.")
            sys.exit(1)

    sys.argv = [sys.argv[0], package_path, model_choice]

# Initialize OpenAI client with custom base URL and API key if provided
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")
client_args = {}
if openai_api_key:
    client_args["api_key"] = openai_api_key
if openai_base_url:
    client_args["base_url"] = openai_base_url
client = OpenAI(**client_args)

LOCATION = os.getenv("VERTEX_LOCATION", os.getenv("LOCATION", "us-east5"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
anthropic_client = AnthropicVertex(region=LOCATION, project_id=os.getenv("PROJECT_ID"))

token_tracker = {
    "input_tokens": 0,
    "output_tokens": 0
}

openai_token_tracker = {
    "input_tokens": 0,
    "output_tokens": 0
}

def gemini(prompt):
    global input_tokens, output_tokens
    response = gemini_client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=prompt,
    )
    
    input_count = response.usage_metadata.prompt_token_count
    output_count = response.usage_metadata.candidates_token_count

    print("Prompt tokens:", input_count)
    print("Output tokens:", output_count)
    print("Total tokens:", response.usage_metadata.total_token_count)

    # Update tracker
    token_tracker["input_tokens"] += input_count
    token_tracker["output_tokens"] += output_count

    return response.text

def anthropic(prompt, max_retries=5, backoff_factor=2):
    retries = 0
    wait_time = 1  

    while retries < max_retries:
        try:
            message = anthropic_client.messages.create(
                max_tokens=8000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet@20250219")
            )
            return message.content[0].text

        except AnthropicError as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                print(f"Rate limit hit (429). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= backoff_factor
            else:
                raise  # re-raise other errors
    raise Exception("Max retries exceeded due to rate limiting.")

def openai(prompt):
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    openai_input_count = response.usage.prompt_tokens
    openai_output_count = response.usage.completion_tokens

    print("Prompt tokens:", openai_input_count)
    print("Output tokens:", openai_output_count)
    print("Total tokens:", response.usage.total_tokens)

    # Update tracker
    openai_token_tracker["input_tokens"] += openai_input_count
    openai_token_tracker["output_tokens"] += openai_output_count
    return response.choices[0].message.content

def vllm_models(prompt):

    openai_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    openai_api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
    model_name = os.getenv("VLLM_MODEL", "openai/gpt-oss-20b")

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response_text = ""
    try: 
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            # max_tokens = 2000,
        )
        print(response)
        input_count = response.usage.prompt_tokens
        output_count = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        print("Prompt tokens:", input_count)
        print("Output tokens:", output_count)
        print("Total tokens:", total_tokens)

        # Update tracker
        token_tracker["input_tokens"] += input_count
        token_tracker["output_tokens"] += output_count
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"failed with exception {e}")

    return response_text


