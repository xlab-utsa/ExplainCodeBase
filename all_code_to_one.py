import os
from models import gemini, anthropic, token_tracker, vllm_models, openai, resolve_args
resolve_args()
import json
import re
import sys

model = {
    "gemini": gemini,
    "anthropic": anthropic,
    "vllm_models": vllm_models,
    "openai": openai,
}

def collect_python_files(src_dir):
    """
    Traverse the given directory and collect all .py files.
    """
    python_files = []
    
    # Walk through the directory
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                # Get the full path of the Python file
                python_files.append(os.path.join(root, file))
    
    return python_files

def convert_llm_output_to_json(llm_output):
    # Initialize an empty dictionary to store the structured data
    result = {}
    lines = llm_output.splitlines()
    # Iterate through the lines of the LLM output
    for line in lines:
        if line and ':' in line:
            func_details = re.split(r'(?<!:):(?!:)', line)
            if len(func_details) != 2 or func_details[1] == "":
                continue

            func_details[0] = re.sub(r'\*+|```', '', func_details[0])
            func_details[0] = func_details[0].replace(" ", "").replace("-", "").replace(".py", "").replace("`", "").replace("\\", "")
            if "<no_class>." in func_details[0]:
                func_details[0] = func_details[0].replace("<no_class>", "")
            elif "<no_class>" in func_details[0]:
                func_details[0] = func_details[0].replace("<no_class>", "(global)")
            result[func_details[0]] = [func_details[1]]  # The function details (summary)

    # Convert the result to JSON format (pretty-printed)
    return json.dumps(result, indent=4)

def copy_code_to_single_file(src_dir, output_file):
    """
    Collect all Python file content and save it into a single file.
    """
    python_files = collect_python_files(src_dir)
    complete_code = ""

    for python_file in python_files:
        with open(python_file, 'r') as infile:
            file_name = python_file.split('/') if '/' in python_file else python_file
            complete_code += "Filename: " + file_name[-1] + "\n" + infile.read()
    
    complete_code += "Provide detailed summary of every function in the codebase and provide this explanation in bullet points. Display the function name in the format example: <file_name without.py>::<class_name_if_exits>.<function_name without brackets>: <explanation> e.g, derived_class::DerivedClass.display_info where derived_class is file name mentioned, DerivedClass is class name and after . is function name and make sure there is NO BACKSLASH in FUNCTION NAME. DO NOT INCLUDE STRING FORMATTING IN FUNCTION NAME. Include __ in init functions."
 
    # output_filename_anthropic = output_file + "_anthropic_result1.json"
    # with open(output_filename_anthropic, "w") as outfile:
    #     outfile.write(convert_llm_output_to_json(anthropic(complete_code)))

    output_filename_gemini = output_file + f"_{sys.argv[2]}_baseline.json"
    with open(output_filename_gemini, "w") as outfile:
        outfile.write(convert_llm_output_to_json(model[sys.argv[2]](complete_code)))

if __name__ == "__main__":
    # Set the directory of the codebase and output file
    source_directory = sys.argv[1]  # Change this to the path of your codebase
    dir_name = os.path.basename(sys.argv[1].rstrip('/'))
    
    # Ensure the output directory exists
    os.makedirs(os.path.join("output", "summaries"), exist_ok=True)
    output_file_path = os.path.join("output", "summaries", dir_name)

    copy_code_to_single_file(source_directory, output_file_path)

    print("Input Tokens so far:", token_tracker["input_tokens"])
    print("Output Tokens so far:", token_tracker["output_tokens"])