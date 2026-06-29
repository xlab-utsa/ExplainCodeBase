import inspect
import importlib
import inspect
import ast
import os

def get_class_docs(cls):
    """
    Retrieve the documentation and methods of a class, including those
    in parent classes.
    """
    class_docs = {}
    
    # Get class-level documentation
    class_docs["class_doc"] = inspect.getdoc(cls)

    # Get the methods of the class
    class_docs["methods"] = {}
    
    for name, obj in inspect.getmembers(cls):
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            class_docs["methods"][name] = inspect.getdoc(obj)
    
    # Recursively check parent classes for methods they call implicitly
    class_docs["parent_classes"] = []
    for base in cls.__bases__:
        parent_class_docs = get_class_docs(base)
        class_docs["parent_classes"].append({
            "parent_class": base.__name__,
            "class_doc": parent_class_docs["class_doc"],
            "methods": parent_class_docs["methods"]
        })
    
    return class_docs

# Print the documentation for the LightningModule class
# print("Class Doc for LightningModule:")
# print(docs)

# print("\nMethods in LightningModule:")
# for method_name, method_doc in docs["methods"].items():
#     print(f"{method_name}: {method_doc}")

# print("\nParent Class Documentation:")
# for parent in docs["parent_classes"]:
#     print(f"Parent Class: {parent['parent_class']}")
#     print(f"Doc: {parent['class_doc']}")
#     for method_name, method_doc in parent["methods"].items():
#         print(f"  {method_name}: {method_doc}")

def extract_modules_and_classes_from_code(code_str):
    tree = ast.parse(code_str)
    modules = set()
    class_names = []
    base_class_to_module = {}

    import_aliases = {}

    # Collect imported modules and aliases
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_aliases[alias.asname or alias.name] = alias.name
                modules.add(alias.name.split('.')[0])

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    import_aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                    modules.add(node.module.split('.')[0])

    # Now check for class definitions and base classes
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)

            for base in node.bases:
                # Case: base class is just a name (e.g., BaseClass)
                if isinstance(base, ast.Name):
                    base_class = base.id
                    if base_class in import_aliases:
                        full_module = import_aliases[base_class]
                        modules.add(full_module.split('.')[0])
                        base_class_to_module[base_class] = full_module

                # Case: base class is like module.BaseClass
                elif isinstance(base, ast.Attribute):
                    parts = []
                    current = base
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                        parts.reverse()
                        base_class = parts[-1]  # The actual class name
                        module_prefix = parts[0]  # Module name or alias
                        if module_prefix in import_aliases:
                            full_module = import_aliases[module_prefix]
                            modules.add(full_module.split('.')[0])
                            base_class_to_module[base_class] = full_module

    return {
        "modules": modules,
        "class_names": class_names,
        "base_class_to_module": base_class_to_module
    }


def get_modules_info(class_code):
    """
    Extract information about base classes used in the provided code.
    Returns documentation if available, otherwise returns source code.
    """
    modules_used = extract_modules_and_classes_from_code(class_code)
    modules_info = ""
    
    for base_class, module_path in modules_used["base_class_to_module"].items():
        # Extract just the module name (first part of the path)
        module_name = module_path.split('.')[0]
        
        try:
            # Import the module (not the class)
            module = importlib.import_module(module_name)
            modules_info += f"\n--- Module: {module_name} ---\n"
            
            # Check if the base class exists in this module
            if hasattr(module, base_class):
                class_obj = getattr(module, base_class)
                if inspect.isclass(class_obj):
                    # Try to get documentation
                    docs = get_class_docs(class_obj)
                    if docs["class_doc"]:
                        modules_info += f"Class: {base_class}\n"
                        modules_info += f"Doc: {docs['class_doc']}\n"
                        for method, doc in docs['methods'].items():
                            modules_info += f"  Method: {method}\n"
                            modules_info += f"  Method Doc: {doc or 'No documentation'}\n"
                    else:
                        # If no documentation, extract source code
                        try:
                            source = inspect.getsource(class_obj)
                            modules_info += f"Class: {base_class}\n"
                            modules_info += f"Source Code:\n{source}\n"
                        except Exception:
                            modules_info += f"No documentation available for {base_class}\n"
            else:
                # If class not found in module, try to extract source directly
                try:
                    module_file = inspect.getfile(module)
                    with open(module_file, 'r') as f:
                        source = f.read()
                    
                    # Parse file to find class definition
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == base_class:
                            # Get the source code segment for this class
                            source_segment = ast.get_source_segment(source, node)
                            if source_segment:
                                modules_info += f"Class: {base_class}\n"
                                modules_info += f"Source Code:\n{source_segment}\n"
                                break
                    else:
                        modules_info += f"Class {base_class} not found in {module_file}\n"
                except Exception as e:
                    modules_info += f"Error analyzing source: {str(e)}\n"
        
        except ModuleNotFoundError:
            modules_info += f"Could not import module: {module_name}\n"
            
            # Try to find the module file directly
            module_file = f"{module_name}.py"
            if os.path.exists(module_file):
                with open(module_file, 'r') as f:
                    source = f.read()
                
                # Parse file to find class definition
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == base_class:
                        # Get the source code segment for this class
                        source_segment = ast.get_source_segment(source, node)
                        if source_segment:
                            modules_info += f"Class: {base_class}\n"
                            modules_info += f"Source Code:\n{source_segment}\n"
                            break
                else:
                    modules_info += f"Class {base_class} not found in {module_file}\n"
            else:
                # If the module file isn't found in the current directory
                modules_info += f"Module file {module_file} not found in current directory\n"
                modules_info += "Consider adding the module path to your Python path\n"
    
    return modules_info

