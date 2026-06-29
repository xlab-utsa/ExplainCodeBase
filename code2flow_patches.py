#!/usr/bin/env python3
"""
Monkey patches for code2flow to handle crashes gracefully.

This module patches code2flow functions to prevent AssertionErrors and other
exceptions that can crash the graph generation process on certain codebases.

Import this module BEFORE running code2flow to apply the patches.
"""

import ast
import code2flow.python
import code2flow.model

# Store original functions for reference
original_get_call = code2flow.python.get_call_from_func_element
original_resolve_variables = code2flow.model.Node.resolve_variables


class DummyCall:
    """
    A placeholder Call object for when code2flow can't parse a function call.
    Implements the same interface as code2flow's Call class.
    """
    def __init__(self, name="(unknown_call)"):
        self.name = name
        self.token = name
        self.line_number = 1

    def matches_variable(self, variable):
        return False

    def is_attr(self):
        return False

    def to_string(self):
        return self.name

    def __repr__(self):
        return f"<DummyCall {self.name}>"


# Global tracking for approximated nodes (useful for debugging)
APPROXIMATION_COUNT = 0
APPROXIMATED_NODES = []


def track_approximation(name, reason):
    """Track when we use a placeholder instead of the real call."""
    global APPROXIMATION_COUNT
    APPROXIMATION_COUNT += 1
    APPROXIMATED_NODES.append({"name": name, "reason": reason})


def patched_get_call_from_func_element(func):
    """
    Patched version of code2flow.python.get_call_from_func_element.
    
    Returns a DummyCall placeholder instead of crashing when encountering
    unsupported AST node types or assertion errors.
    """
    try:
        # If the type is not one of the expected ones, return a placeholder
        if type(func) not in (ast.Attribute, ast.Name, ast.Subscript, ast.Call):
            try:
                name = ast.unparse(func)
                # Truncate if too long to avoid UI issues
                if len(name) > 50:
                    name = name[:47] + "..."
            except Exception:
                name = f"({type(func).__name__})"
            
            track_approximation(name, "Unsupported Node Type")
            return DummyCall(name)
        
        return original_get_call(func)
        
    except AssertionError:
        try:
            name = ast.unparse(func)
            if len(name) > 50:
                name = name[:47] + "..."
        except Exception:
            name = "(assert_error_call)"
        
        track_approximation(name, "Assertion Error")
        return DummyCall(name)
        
    except Exception:
        track_approximation("(error_call)", "Exception")
        return DummyCall("(error_call)")


def patched_resolve_variables(self, file_groups):
    """
    Patched version of code2flow.model.Node.resolve_variables.
    
    Ignores assertion errors during variable resolution instead of crashing.
    """
    try:
        original_resolve_variables(self, file_groups)
    except AssertionError:
        # Ignore assertion errors during variable resolution
        pass
    except Exception:
        pass


def apply_patches():
    """Apply all monkey patches to code2flow."""
    code2flow.python.get_call_from_func_element = patched_get_call_from_func_element
    code2flow.model.Node.resolve_variables = patched_resolve_variables


def get_approximation_stats():
    """Get statistics about approximated nodes."""
    return {
        "count": APPROXIMATION_COUNT,
        "nodes": APPROXIMATED_NODES
    }


def reset_approximation_stats():
    """Reset the approximation tracking."""
    global APPROXIMATION_COUNT, APPROXIMATED_NODES
    APPROXIMATION_COUNT = 0
    APPROXIMATED_NODES = []


# Apply patches when this module is imported
apply_patches()
