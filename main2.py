import json
import os
from collections import deque
import ast
import astor
# from google import genai
import sys
import re
from condensed import get_modules_info
from models import gemini, anthropic, token_tracker, vllm_models, openai, resolve_args
resolve_args()
import time

print("Started Processing using Method 2")
start_time = time.time()

myFolder = "."
fileSet = set() 

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = {
    "gemini": gemini,
    "anthropic": anthropic,
    "vllm_models": vllm_models,
    "openai": openai,
}


for root, dirs, files in os.walk(myFolder):
    for fileName in files:
        fileSet.add( os.path.join( root[len(myFolder):], fileName ))

dir_name = os.path.basename(sys.argv[1].rstrip('/'))
file_path = os.path.join("output", "graph", f"{dir_name}-file.json")

if not os.path.exists(file_path):
    found = False
    for i in fileSet:
        if i.endswith(f"{dir_name}-file.json"):
            file_path = os.path.join(os.getcwd(), i)
            found = True
            break
    if not found:
        file_path = os.path.join(os.getcwd(), f"{dir_name}-file.json")
else:
    file_path = os.path.abspath(file_path)

try:
    with open(file_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {file_path}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

nodes_info = data['graph']['nodes']
edges_info = data['graph']['edges']

nodes = list(nodes_info.keys())
root_nodes = []

for i in nodes:
    if '(global)' in nodes_info[i]["name"]:
        root_nodes.append(i)

adjacency_list = {}
connections = {}
contexts = {}

for i in nodes:
    adjacency_list[i] = []
    connections[i] = set()
    contexts[i] = set()


for i in edges_info:
    adjacency_list[i["source"]].append(i["target"])

node_level = {}

# BFS with level information
def bfs(graph, start):
    visited = set()  # to track visited nodes
    queue = deque([(start, 0)])  # queue holds tuples of (node, level)
    parent_map = {}  # to store the parent-child relationship
    
    while queue:
        node, level = queue.popleft()  # get the node and its level from the front of the queue
        
        if node in visited:
            node_level[node] = max(node_level[node], level)

        if node not in visited:
            visited.add(node)
            node_level[node] = level
            # For each neighbor, print the connection and add to the queue if not visited
            for neighbor in graph[node]:
                connections[node].add(neighbor)
                queue.append((neighbor, level + 1))

for i in root_nodes:
    bfs(adjacency_list, i)

# sorting in decreasing order
node_level = dict(sorted(node_level.items(), key=lambda item: item[1], reverse=True))



def extract_function_from_file(filename, function_name):

    for files in filename:
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

def insert_comments_above_calls(code, function_descriptions):
    """
    Inserts comments above function calls in the given code based on provided function descriptions.
    
    :param code: The source code as a string.
    :param function_descriptions: A dictionary mapping function names to their descriptions.
    :return: The modified source code with comments.
    """
    tree = ast.parse(code)
    
    class CommentInserter(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call):
                call_node = node.value
                func_name = None
                
                if isinstance(call_node.func, ast.Name):
                    func_name = call_node.func.id
                elif isinstance(call_node.func, ast.Attribute):
                    func_name = call_node.func.attr
                
                if func_name in function_descriptions:
                    comment = ast.Expr(ast.Constant(
                        value=f"# {function_descriptions[func_name]}"))
                    return [comment, node]
            return node

    modified_tree = CommentInserter().visit(tree)
    ast.fix_missing_locations(modified_tree)
    return astor.to_source(modified_tree)


def get_llm_inference(function_code, context):

    func_desc = {}
    
    prompt = ""
    for i in context:
        try:
            function_name = re.search(r'(\w+)\(', i).group(1)
            func_desc[function_name] = i
            prompt += i
        except:
            prompt += context

    if function_code == None:
        prompt = prompt + '\n' + "summarize everything in one line and divide this line in bullet points."
    else:
        prompt = insert_comments_above_calls(function_code, func_desc) + "summarize the given function in one sentence and divide this in bullet points."
    
    # response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents=prompt,
    # )
    return model[sys.argv[2]](prompt)

local_files = set()
for root, dirs, files in os.walk(sys.argv[1]):
    for fileName in files:
        local_files.add( os.path.join( root[len(sys.argv[1]):], fileName))

def extract_class_function_from_file(file_paths, class_name, function_name):

    for file_path in file_paths:
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

def get_info_function(node):
    label = nodes_info[node]["name"].split('::')
    func_name = nodes_info[node]["label"].split(': ')[1]
    
    file_path = []

    for i in local_files:
        if i.endswith(label[0]+'.py'):
            file_path.append(i)

    if len(file_path) == 0:
        print(f"Warning: No Python file found for {label[0]}")
        return None  
    
    cur_dir = os.getcwd()
    os.chdir(sys.argv[1])

    if '.' in label[1]:
        class_name = label[1].split('.')[0]
        func_name = label[1].split('.')[1]
        
        function_text = extract_class_function_from_file(file_path, class_name, func_name)
    else:
        function_text = extract_function_from_file(file_path, label[1])
    
    os.chdir(cur_dir)

    return function_text

output_explanation = {}
def build_context():
    stack = []
    visited = set()

    # Iterate over all nodes
    for i in node_level.keys():
        if i not in visited:
            stack.append(i)
            
            while stack:
                node = stack[-1]
                if node in visited:
                    stack.pop()
                    # Process node after children
                    for j in connections[node]:
                        for k in contexts[j]:
                            if k not in contexts[node]:
                                contexts[node].add(k)
                    inf = get_llm_inference(get_info_function(node), contexts[node])
                    contexts[node].add(nodes_info[node]["label"] + " " + inf)
                    output_explanation[node] = inf
                else:
                    visited.add(node)
                    # Push children to stack first
                    for child in reversed(list(connections[node])):
                        if child not in visited:
                            stack.append(child)

build_context()


class_context = {}
class_context[""] = ""

def extract_class_definition(file_paths, class_name):
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            file_content = file.read()

        tree = ast.parse(file_content)

        # Extract import statements first
        import_statements = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_statements.append(ast.unparse(node))
            elif isinstance(node, ast.ImportFrom):
                import_statements.append(ast.unparse(node))

        # Search for class definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Return imports along with the class definition
                class_definition = ast.unparse(node)
                return '\n'.join(import_statements) + '\n' + class_definition

    return None  # If the class was not found

function_explanation = {}
for node, value in contexts.items():
    function_explanation[node] = list(value)
    if len(function_explanation[node]) == 0:
        label = nodes_info[node]["name"].split('::')
        func_name = nodes_info[node]["label"].split(': ')[1]
        
        file_path = []

        for i in local_files:
            if i.endswith(label[0]+'.py'):
                file_path.append(i)

        if len(file_path) == 0:
            print(f"Warning: No Python file found for {label[0]}")
            continue  
        
        cur_dir = os.getcwd()
        os.chdir(sys.argv[1])

        class_name = ""
        if '.' in label[1]:
            class_name = label[1].split('.')[0]
            func_name = label[1].split('.')[1]
            
            if class_name not in class_context:
                class_code = extract_class_definition(file_path, class_name)
                context_class = get_modules_info(class_code)
                if "Source Code:" in context_class:
                    class_context[class_name] = get_llm_inference(get_info_function(node), "")
                else:
                    class_context[class_name] = context_class
        function_explanation[node] = get_llm_inference(get_info_function(node), class_context[class_name])
        output_explanation[node] = function_explanation[node]
        os.chdir(cur_dir)

json_object = json.dumps(function_explanation, indent=4)
 
os.makedirs(os.path.join("output", "summaries"), exist_ok=True)

output_filename = os.path.join("output", "summaries", f"{dir_name}_{sys.argv[2]}_result2.json")
with open(output_filename, "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(output_explanation, indent=4)
output_filename = os.path.join("output", "summaries", f"{dir_name}_{sys.argv[2]}_iucp.json")
with open(output_filename, "w") as outfile:
    outfile.write(json_object)

print("Input Tokens so far:", token_tracker["input_tokens"])
print("Output Tokens so far:", token_tracker["output_tokens"])
print("Finished processing using method 2")
print("Time taken: ", time.time()-start_time)