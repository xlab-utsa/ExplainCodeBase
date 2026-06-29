import os
import json
import ast
import sys
from models import gemini, anthropic, vllm_models, openai, token_tracker, openai_token_tracker
import re
import string

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Source code packages are one level up from project root
SOURCE_CODE_ROOT = os.path.dirname(PROJECT_ROOT)

model = {
    "gemini": gemini,
    "anthropic": anthropic,
    "vllm_models": vllm_models,
    "openai": openai,
}

def get_all_packages():
    """Get all packages from the output/summaries folder."""
    new_summaries_dir = os.path.join(PROJECT_ROOT, "output", "summaries")
    if not os.path.exists(new_summaries_dir):
        print(f"Error: Summaries directory not found at {new_summaries_dir}")
        return []
    
    packages = set()
    supported_models = ["gemini", "anthropic", "vllm_models", "openai"]
    suffixes = ["baseline", "hucp", "iucp", "llmucp", "docucp"]
    for file in os.listdir(new_summaries_dir):
        if file.endswith(".json"):
            for model_name in supported_models:
                for suffix in suffixes:
                    pattern = f"_{model_name}_{suffix}.json"
                    if file.endswith(pattern):
                        package_name = file[:-len(pattern)]
                        packages.add(package_name)
                        break
                    
    return sorted(list(packages))

def get_graph_file(package_name):
    """Get the graph JSON file for a package."""
    graph_path = os.path.join(PROJECT_ROOT, "output", "graph", f"{package_name}-file.json")
    
    if not os.path.exists(graph_path):
        print(f"Warning: Graph file not found for {package_name} at {graph_path}")
        return None
    
    return graph_path

def get_package_path(package_name):
    """Get the source code package path."""
    # Try input directory first
    input_path = os.path.join(PROJECT_ROOT, "input", package_name)
    if os.path.exists(input_path):
        return input_path
        
    # Try parent directory fallback
    package_path = os.path.join(SOURCE_CODE_ROOT, package_name)
    if os.path.exists(package_path):
        return package_path
        
    print(f"Warning: Package source code not found at {package_path}")
    return None

def get_inference_files(package_name):
    """Get all result files from the output/summaries folder for a package."""
    new_summaries_dir = os.path.join(PROJECT_ROOT, "output", "summaries")
    if not os.path.exists(new_summaries_dir):
        return []
    
    result_files = []
    supported_models = ["gemini", "anthropic", "vllm_models", "openai"]
    suffixes = ["baseline.json", "hucp.json", "iucp.json", "llmucp.json", "docucp.json"]
    for file in os.listdir(new_summaries_dir):
        if file.startswith(f"{package_name}_"):
            for model_name in supported_models:
                for suffix in suffixes:
                    if file.endswith(f"_{model_name}_{suffix}"):
                        result_files.append(os.path.join(new_summaries_dir, file))
                
    return sorted(result_files)

def load_all_inferences(local_files):
    total_data = []
    for i in local_files:
        with open(i, 'r') as f:
            total_data.append(json.load(f))
    
    return total_data

def extract_function_from_file(filename, function_name):
    for files in filename:
        if files[0] == '/':
            files = files[1:]
        with open(files, 'r') as file:
            file_content = file.read()

        # Parse the file content into an Abstract Syntax Tree (AST)
        tree = ast.parse(file_content)

        # Search for the function in the AST
        for node in ast.walk(tree):
            # Check if the node is a function definition
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    # If we find the function, return its source code
                    function_code = ast.unparse(node)
                    return function_code
    
    return None  # If the function is not found

def get_llm_inference(function_code, context):
    
    prompt = ""
    
    for i in context:
        prompt += i

    if function_code == None:
        prompt = prompt + '\n' + "summarize everything in one line and divide this line in bullet points."
    else:
        prompt = prompt + " \n" + function_code + '\n' + "summarize the given function in one sentence and divide this in bullet points."
    return model[sys.argv[2]](prompt)

def extract_class_function_from_file(file_paths, class_name, function_name):
    
    for file_path in file_paths:
        if file_path[0] == '/':
            file_path = file_path[1:]
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Parse the file content into an AST
        tree = ast.parse(file_content)

        # Store the function nodes
        function_nodes = []
        # Iterate over all the nodes in the AST
        for node in ast.walk(tree):
            # Check if the node is a class definition
            if isinstance(node, ast.ClassDef):
                # If the class name matches, look for the function
                if node.name == class_name:
                    # Iterate over the body of the class to find function definitions
                    for class_node in node.body:
                        if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Check if the function name matches
                            if class_node.name == function_name:
                                function_nodes.append(class_node)

        # If function_nodes is not empty, handle the nodes
        if function_nodes:
            # If it's a list of nodes (multiple functions), iterate over them
            if len(function_nodes) > 1:
                function_codes = ""
                for node in function_nodes:
                    function_codes += ast.unparse(node) + "\n"
                return function_codes  # Return combined function codes

            # If there's only one node, handle it directly
            return ast.unparse(function_nodes[0])  # Return single function code
    
    return None  # If the function was not found

def get_info_function(node, nodes_info, package_files, package_path):
    label = nodes_info[node]["name"].split('::')
    func_name = nodes_info[node]["label"].split(': ')[1]
    
    file_path = []

    for i in package_files:
        if i.endswith(label[0]+'.py'):
            file_path.append(i)

    if len(file_path) == 0:
        print(f"Warning: No Python file found for {label[0]}")
        return None  
    
    cur_dir = os.getcwd()
    os.chdir(package_path)

    if '.' in label[1]:
        class_name = label[1].split('.')[0]
        func_name = label[1].split('.')[1]
        
        function_text = extract_class_function_from_file(file_path, class_name, func_name)
    else:
        function_text = extract_function_from_file(file_path, label[1])
    
    os.chdir(cur_dir)

    return function_text



def extract_model_and_method(text):
    print(text)
    # Regex to match '**Model:** <model>' and '**Method:** <method>'
    # Updated to include underscores and handle optional trailing punctuation
    pattern = r"Model:\s*([A-Za-z0-9_]+)\s*,\s*Method:\s*([A-Za-z0-9_]+)\.?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        model_name = match.group(1).strip()
        method = match.group(2).strip()
        return model_name, method
    else:
        return None, None

def get_model_and_method_from_filename(filepath, package_name):
    """Extract model name and method number from filename.
    
    Format: {package_name}_{model_name}_{suffix}.json
    """
    filename_base = os.path.basename(filepath)
    rel_name = filename_base[len(package_name) + 1:]
    
    # Determine method
    method = None
    if 'iucp.json' in rel_name:
        method = '2'
    elif 'llmucp.json' in rel_name:
        method = '3'
    elif 'docucp.json' in rel_name:
        method = '4'
    elif 'baseline.json' in rel_name:
        method = 'baseline'
    elif 'hucp.json' in rel_name:
        method = '1'
        
    # Determine model
    model_name = None
    for m in ["gemini", "anthropic", "vllm_models", "openai"]:
        if rel_name.startswith(m):
            model_name = m
            break
            
    return model_name, method

def process_package(package_name, eval_model):
    """Process a single package."""
    print(f"\n{'='*60}")
    print(f"Processing package: {package_name}")
    print(f"{'='*60}\n")
    
    # Get graph file
    filename = get_graph_file(package_name)
    if filename is None:
        print(f"Skipping {package_name}: Graph file not found")
        return
    
    # Get package path
    package_path = get_package_path(package_name)
    if package_path is None:
        print(f"Skipping {package_name}: Source code not found")
        return
    
    # Get package files
    package_files = set()
    for root, dirs, files in os.walk(package_path):
        for fileName in files:
            package_files.add(os.path.join(root[len(package_path):], fileName))
    
    # Load graph content
    with open(filename, 'r') as f:
        content = json.load(f)
    
    nodes_info = content['graph']['nodes']
    
    # Get inference files
    local_files = get_inference_files(package_name)
    if not local_files:
        print(f"Skipping {package_name}: No inference files found")
        return
    
    # Load all inferences
    info = load_all_inferences(local_files)
    
    # Initialize result dictionary for this package
    res_dict = {
        "gemini": 0,
        "anthropic": 0,
        "vllm_models": 0,
        "openai": 0,
        "baseline": 0,
        'gpt_oss': 0,
        'deepseek': 0,
        'llama': 0,
        'qwen': 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
    }
    
    def results():
        for node in nodes_info:
            function = get_info_function(node, nodes_info, package_files, package_path)
            if function is None or '(global)' in nodes_info[node]['label']:
                continue
            prompt = "Python Code:\n" + function + "\n"
            
            for idx, inference_file in enumerate(local_files):
                model_name, method = get_model_and_method_from_filename(inference_file, package_name)
                if model_name is None or method is None:
                    continue
                
                # Check for either node ID (e.g. node_0) or node name (e.g. main::print_hello)
                node_name = content['graph']['nodes'][node]['name']
                node_key = None
                if node in info[idx]:
                    node_key = node
                elif node_name in info[idx]:
                    node_key = node_name
                
                if node_key is not None:
                    prompt += f"Output from METHOD {method} using MODEL {model_name}\n" + str(info[idx][node_key]) + "\n"
                else:
                    continue
            
            prompt += "Choose the model and method which provides best summary for the given Python Code and provide the information in 5 words only. OUTPUT FORMAT: Model: <name>, Method: <method_number>. Line 'Output from METHOD <> using MODEL <>' of the prompt contains model name and method"
            print(prompt)
            for i in range(10):   
                model_names, model_method = extract_model_and_method(eval_model(prompt))
                if model_names:
                    name_key = model_names.lower()
                    try:
                        if name_key not in res_dict:
                            res_dict[name_key] = 0
                        res_dict[name_key] += 1
                    except Exception as e:
                        print(f"Error incrementing model key '{name_key}':", e)
                if model_method:
                    try:
                        if model_method not in res_dict:
                            res_dict[model_method] = 0
                        res_dict[model_method] += 1
                    except Exception as e:
                        print(f"Error incrementing method key '{model_method}':", e)
    
    results()
    
    # Save results for this package in a new directory: output/evaluation/
    output_dir = os.path.join(PROJECT_ROOT, "output", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{package_name}.json")
    with open(output_path, 'w') as json_file:
        json.dump(res_dict, json_file)
    
    print(f"\nResults saved to {output_path}")
    print(f"Results: {res_dict}")

# Main execution
if __name__ == "__main__":
    model_choice = "openai"
    target_package = None
    
    for arg in sys.argv[1:]:
        if arg in model:
            model_choice = arg
        else:
            target_package = os.path.basename(arg.rstrip('/'))
            
    eval_model = model[model_choice]
    print(f"Using {model_choice} for evaluation.")

    all_packages = get_all_packages()
    if target_package:
        if target_package not in all_packages:
            print(f"Error: No generated summary files found for package '{target_package}' in output/summaries/.")
            sys.exit(1)
        packages = [target_package]
    else:
        packages = all_packages
        if not packages:
            print("No packages found in inferences folder")
            sys.exit(1)
            
    print(f"Found {len(packages)} package(s) to process: {packages}\n")
    
    # Process each package
    for package_name in packages:
        try:
            process_package(package_name, eval_model)
        except Exception as e:
            print(f"Error processing {package_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
        print("\n" + "="*60)
        print("OpenAI Tokens so far:", openai_token_tracker)
        print("Other LLM Tokens so far (Gemini/Anthropic/vLLM):", token_tracker)
        print("="*60)
        openai_token_tracker["input_tokens"] = 0
        openai_token_tracker["output_tokens"] = 0
        token_tracker["input_tokens"] = 0
        token_tracker["output_tokens"] = 0