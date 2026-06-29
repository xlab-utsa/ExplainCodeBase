#!/usr/bin/env python3
"""
Run code2flow with the hybrid approach for accurate call graphs.

This script uses a hybrid approach to generate call graphs:
- NODES from full-directory run (captures all nodes including cross-file dependencies)
- EDGES from per-file runs (avoids incorrect cross-file class/function resolution)

Includes monkey patches (via code2flow_patches) to handle code2flow crashes gracefully.
"""

import subprocess
import json
import os
import sys
import argparse
from pathlib import Path
import tempfile
import shutil

# Import and apply monkey patches for code2flow crash handling
# This must be imported before any code2flow operations
try:
    import code2flow_patches  # Patches are auto-applied on import
except ImportError:
    # If running from a different directory, try relative import
    try:
        from . import code2flow_patches
    except ImportError:
        print("Warning: code2flow_patches module not found. Crash handling may not work.", file=sys.stderr)



def find_python_files(directory: str, recursive: bool = True) -> list[Path]:
    """Find all Python files in the given directory."""
    directory = Path(directory)
    if recursive:
        return list(directory.rglob("*.py"))
    else:
        return list(directory.glob("*.py"))


def run_code2flow_on_file(python_file: Path, output_dir: Path) -> dict | None:
    """Run code2flow on a single Python file and return the JSON graph."""
    # Use hash to avoid filename collisions for same-named files in different directories
    output_file = output_dir / f"{python_file.stem}_{abs(hash(str(python_file))) % 10000}.json"
    
    try:
        result = subprocess.run(
            ["code2flow", str(python_file), "-o", str(output_file), "--quiet"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"  Warning: code2flow failed for {python_file.name}: {result.stderr.strip()}")
            return None
            
        if output_file.exists():
            with open(output_file) as f:
                return json.load(f)
        return None
        
    except subprocess.TimeoutExpired:
        print(f"  Warning: Timeout processing {python_file.name}")
        return None
    except Exception as e:
        print(f"  Warning: Error processing {python_file.name}: {e}")
        return None


def run_code2flow_on_directory(directory: Path, output_dir: Path) -> dict | None:
    """
    Run code2flow on the full directory to capture global nodes.
    Global nodes (files with only top-level code) are only generated when 
    code2flow processes the entire directory together.
    """
    output_file = output_dir / "full_directory.json"
    
    try:
        result = subprocess.run(
            ["code2flow", str(directory), "-o", str(output_file), "--quiet", "--language", "py"],
            capture_output=True,
            text=True,
            timeout=300  # Longer timeout for full directory
        )
        
        if result.returncode != 0:
            return None
            
        if output_file.exists():
            with open(output_file) as f:
                return json.load(f)
        return None
        
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def merge_graphs(graphs: list[tuple[str, dict]]) -> dict:
    """
    Merge multiple code2flow JSON graphs into a single graph.
    Each graph is a tuple of (filename, graph_dict).
    
    Nodes are deduplicated by their qualified name (e.g., 'module::ClassName.method')
    to ensure that if multiple files call the same function, only one node exists
    in the merged graph, with all edges correctly pointing to it.
    """
    merged = {
        "graph": {
            "directed": True,
            "nodes": {},
            "edges": []
        }
    }
    
    # Map from qualified node name -> canonical node ID
    name_to_canonical_id: dict[str, str] = {}
    # Map from (filename, original_id) -> canonical node ID
    id_remap: dict[tuple[str, str], str] = {}
    # Counter for generating new unique IDs when needed
    next_id = 0
    
    # First pass: collect all nodes and deduplicate by name
    for filename, graph_data in graphs:
        if not graph_data or "graph" not in graph_data:
            continue
            
        graph = graph_data["graph"]
        
        for node_id, node_data in graph.get("nodes", {}).items():
            # Get the qualified name of the node (e.g., "module::ClassName.method")
            node_name = node_data.get("name", node_id)
            
            if node_name in name_to_canonical_id:
                # Node with this name already exists, reuse its canonical ID
                canonical_id = name_to_canonical_id[node_name]
            else:
                # New node, assign a canonical ID
                canonical_id = f"node_{next_id}"
                next_id += 1
                name_to_canonical_id[node_name] = canonical_id
                # Copy node data and update uid to match the canonical ID
                updated_node_data = node_data.copy()
                updated_node_data["uid"] = canonical_id
                # Add the node to merged graph with the canonical ID
                merged["graph"]["nodes"][canonical_id] = updated_node_data
            
            # Record the mapping from this file's original ID to the canonical ID
            id_remap[(filename, node_id)] = canonical_id
    
    # Second pass: add edges with remapped IDs
    for filename, graph_data in graphs:
        if not graph_data or "graph" not in graph_data:
            continue
            
        graph = graph_data["graph"]
        
        for edge in graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            
            # Remap source and target to canonical IDs
            canonical_source = id_remap.get((filename, source), source)
            canonical_target = id_remap.get((filename, target), target)
            
            # Create remapped edge
            remapped_edge = {
                "source": canonical_source,
                "target": canonical_target
            }
            
            # Avoid duplicate edges
            if remapped_edge not in merged["graph"]["edges"]:
                merged["graph"]["edges"].append(remapped_edge)
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Run code2flow on each Python file individually and merge results."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing Python files (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output",
        default="out_merged.json",
        help="Output file path (default: out_merged.json)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Files or patterns to exclude"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--skip-parse-errors",
        action="store_true",
        help="Skip files with parse errors (handled by patches, kept for CLI compatibility)"
    )
    parser.add_argument(
        "--language",
        default="py",
        help="Language filter (default: py, kept for CLI compatibility)"
    )
    
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    
    # Auto-detection of packages inside input folder when no package is specified
    if not directory.exists() or (args.directory == "." and os.path.exists("input") and os.path.isdir("input")):
        input_dir = Path("input").resolve()
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if subdirs:
            directory = sorted(subdirs)[0]
            if not args.quiet:
                print(f"No valid directory path provided. Found package in input folder: {directory}")
        elif not directory.exists():
            print(f"Error: Directory '{directory}' does not exist and no packages found in 'input' folder")
            sys.exit(1)
    elif not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    # Find all Python files
    python_files = find_python_files(directory, recursive=not args.no_recursive)
    
    # Filter out excluded files
    if args.exclude:
        python_files = [
            f for f in python_files 
            if not any(excl in str(f) for excl in args.exclude)
        ]
    
    # Also exclude this script itself
    script_name = Path(__file__).name
    python_files = [f for f in python_files if f.name != script_name]
    
    if not python_files:
        print("No Python files found")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Found {len(python_files)} Python files:")
        for f in python_files:
            print(f"  - {f.relative_to(directory)}")
        print()
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # === HYBRID APPROACH ===
        # Problem: code2flow has two issues when processing directories:
        # 1. Per-file runs miss nodes with cross-file dependencies (e.g., global nodes)
        # 2. Full-directory runs incorrectly resolve same-named classes across files
        #
        # Solution:
        # - Get NODES from full-directory run (captures all nodes)
        # - Get EDGES from per-file runs (avoids incorrect cross-file edges)
        
        # Step 1: Run code2flow on full directory to get all nodes
        if not args.quiet:
            print("Step 1: Running code2flow on full directory for nodes...")
        full_dir_graph = run_code2flow_on_directory(directory, temp_path)
        
        if not full_dir_graph or not full_dir_graph.get("graph", {}).get("nodes"):
            print("Error: code2flow did not generate any nodes from full directory")
            sys.exit(1)
        
        full_dir_nodes = full_dir_graph["graph"]["nodes"]
        if not args.quiet:
            print(f"  Found {len(full_dir_nodes)} nodes ✓")
        
        # Step 2: Run code2flow on each file individually to get correct edges
        if not args.quiet:
            print("Step 2: Processing files individually for edges...")
        
        per_file_graphs = []
        for i, python_file in enumerate(python_files, 1):
            if not args.quiet:
                print(f"  [{i}/{len(python_files)}] {python_file.name}...", end=" ", flush=True)
            graph = run_code2flow_on_file(python_file, temp_path)
            if graph and graph.get("graph", {}).get("edges"):
                per_file_graphs.append((python_file.stem, graph))
                if not args.quiet:
                    print("✓")
            else:
                if not args.quiet:
                    print("(no edges)")
        
        if not args.quiet:
            print()
        
        # Step 3: Build merged graph with nodes from full-dir and edges from per-file
        # === TRUE HYBRID APPROACH ===
        # Handle same-named nodes (e.g., multiple main::(global)) correctly by
        # tracking multiple canonical IDs per name and using occurrence-based assignment
        if not args.quiet:
            print("Step 3: Merging (nodes from full-dir, edges from per-file)...")
        
        # Build name-to-canonical-ids mapping (note: plural - list of IDs for same-named nodes)
        # For same-named nodes (like two main::(global)), track all occurrences
        name_to_canonical_ids = {}  # name -> list of canonical IDs
        original_to_canonical = {}
        merged_nodes = {}
        next_id = 0
        
        for orig_uid, node_data in full_dir_nodes.items():
            node_name = node_data.get("name", orig_uid)
            canonical_id = f"node_{next_id}"
            next_id += 1
            
            original_to_canonical[orig_uid] = canonical_id
            
            if node_name not in name_to_canonical_ids:
                name_to_canonical_ids[node_name] = []
            name_to_canonical_ids[node_name].append(canonical_id)
            
            updated_node_data = node_data.copy()
            updated_node_data["uid"] = canonical_id
            merged_nodes[canonical_id] = updated_node_data
        
        # Track which occurrence of each name we've used per file basename
        # This ensures each file's nodes map to the correct canonical nodes
        basename_name_occurrence = {}  # basename -> {name -> occurrence_used}
        merged_edges = []
        # Track which nodes each file defines (for detecting cross-file edges)
        file_defined_nodes = {}  # basename -> set of canonical node IDs
        
        for filename, graph_data in per_file_graphs:
            graph = graph_data["graph"]
            file_nodes = graph.get("nodes", {})
            file_edges = graph.get("edges", [])
            
            # Get file basename for occurrence tracking
            basename = filename  # Already the stem from per_file_graphs tuple
            if basename not in basename_name_occurrence:
                basename_name_occurrence[basename] = {}
            if basename not in file_defined_nodes:
                file_defined_nodes[basename] = set()
            
            # Build local ID to canonical ID mapping for this file
            local_to_canonical = {}
            
            for local_id, node_data in file_nodes.items():
                node_name = node_data.get("name", local_id)
                
                if node_name in name_to_canonical_ids:
                    canonical_ids = name_to_canonical_ids[node_name]
                    
                    if len(canonical_ids) == 1:
                        # Unique name, use the only one
                        canonical_id = canonical_ids[0]
                        local_to_canonical[local_id] = canonical_id
                        file_defined_nodes[basename].add(canonical_id)
                    else:
                        # Multiple nodes with same name, use occurrence-based assignment
                        if node_name not in basename_name_occurrence[basename]:
                            basename_name_occurrence[basename][node_name] = 0
                        else:
                            basename_name_occurrence[basename][node_name] += 1
                        
                        occurrence = basename_name_occurrence[basename][node_name]
                        if occurrence < len(canonical_ids):
                            canonical_id = canonical_ids[occurrence]
                            local_to_canonical[local_id] = canonical_id
                            file_defined_nodes[basename].add(canonical_id)
            
            # Add edges from this file
            for edge in file_edges:
                source = edge.get("source")
                target = edge.get("target")
                
                canonical_source = local_to_canonical.get(source)
                canonical_target = local_to_canonical.get(target)
                
                if canonical_source and canonical_target:
                    remapped_edge = {
                        "source": canonical_source,
                        "target": canonical_target
                    }
                    if remapped_edge not in merged_edges:
                        merged_edges.append(remapped_edge)
        
        if not args.quiet:
            print()
        
        # Step 4: Add cross-file edges from full-directory run
        # Per-file runs capture correct intra-file edges but miss cross-file calls
        # Full-dir run captures cross-file edges, but may have wrong resolution for duplicate names
        # Solution: Add full-dir edges when:
        #   1. Source is orphaned (not in any per-file run), OR
        #   2. Target's NAME is unique (no risk of incorrect cross-file resolution)
        if not args.quiet:
            print("Step 4: Adding cross-file edges from full directory run...")
        
        full_dir_edges = full_dir_graph["graph"].get("edges", [])
        cross_file_edges_added = 0
        
        # Build set of all nodes defined in per-file runs
        all_per_file_nodes = set()
        for nodes in file_defined_nodes.values():
            all_per_file_nodes.update(nodes)
        
        # Build canonical_id -> node_name mapping for uniqueness check
        canonical_to_name = {cid: merged_nodes[cid].get("name", cid) for cid in merged_nodes}
        
        # Build a set of class/function names (part after ::) that appear in multiple files
        # e.g., "Net.__init__" appears in vanilla.py, mfcc.py, etc.
        class_name_counts = {}
        for name in canonical_to_name.values():
            class_part = name.split("::")[-1] if "::" in name else name
            class_name_counts[class_part] = class_name_counts.get(class_part, 0) + 1
        
        for edge in full_dir_edges:
            source_orig = edge.get("source")
            target_orig = edge.get("target")
            
            canonical_source = original_to_canonical.get(source_orig)
            canonical_target = original_to_canonical.get(target_orig)
            
            if canonical_source and canonical_target:
                remapped_edge = {
                    "source": canonical_source,
                    "target": canonical_target
                }
                
                # Skip if already captured by per-file runs
                if remapped_edge in merged_edges:
                    continue
                
                # Check if source is orphaned
                source_is_orphaned = canonical_source not in all_per_file_nodes
                
                # Check if target's class/function name is unique (not duplicated across files)
                # e.g., "utils::sample" has class_part "sample" which might be unique
                # but "vanilla::Net.__init__" has class_part "Net.__init__" which appears in many files
                target_name = canonical_to_name.get(canonical_target, "")
                target_class_part = target_name.split("::")[-1] if "::" in target_name else target_name
                target_class_is_unique = class_name_counts.get(target_class_part, 0) == 1
                
                # Add edge if source is orphaned OR target class name is unique
                if source_is_orphaned or target_class_is_unique:
                    merged_edges.append(remapped_edge)
                    cross_file_edges_added += 1
        
        if not args.quiet:
            print(f"  Added {cross_file_edges_added} cross-file edges ✓")
            print()
        
        merged = {
            "graph": {
                "directed": True,
                "nodes": merged_nodes,
                "edges": merged_edges
            }
        }
        
        # Count stats
        num_nodes = len(merged["graph"]["nodes"])
        num_edges = len(merged["graph"]["edges"])
        
        # Write output
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        
        if not args.quiet:
            print(f"✓ Merged graph saved to: {output_path}")
            print(f"  - Nodes: {num_nodes}")
            print(f"  - Edges: {num_edges}")


if __name__ == "__main__":
    main()
