import subprocess
import re
import sys

# --- Configuration ---
# Command 1: The Python DGL script
# We need to provide input '4' for the 'output feature dimension' prompt.
CMD1 = [sys.executable, "dgl_output_main.py"]
CMD1_INPUT = "4\n"

# Command 2: The C++ executable
# Note: Adjust the path if your script is not in the expected directory.
# The path '..\\graph_data.txt' implies this script should be run from
# the same directory as 'dgl_output_main.py'.
CMD2 = [".\\build\\Debug\\graph_app.exe", "..\\graph_data.txt"]

# Tolerance for comparing floating-point numbers
TOLERANCE = 1e-4

# --- Helper Functions ---

def run_command(command_args, input_data=None):
    """
    Executes a shell command and returns its output.

    Args:
        command_args (list): The command and its arguments as a list of strings.
        input_data (str, optional): String data to be passed to the process's stdin.

    Returns:
        tuple: A tuple containing (return_code, stdout_str, stderr_str).
    """
    print(f"--- Running Command: {' '.join(command_args)} ---")
    try:
        process = subprocess.run(
            command_args,
            input=input_data,
            capture_output=True,
            text=True,  # Work with text strings instead of bytes
            check=False # Don't automatically raise an exception on non-zero exit codes
        )
        if process.returncode != 0:
            print(f"Warning: Command exited with code {process.returncode}")
            print(f"Stderr:\n{process.stderr}")
        return process.returncode, process.stdout, process.stderr
    except FileNotFoundError:
        print(f"Error: The command '{command_args[0]}' was not found.")
        print("Please ensure the path is correct and the file is executable.")
        return -1, "", "File not found"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1, "", str(e)


def parse_output(output_str):
    """
    Parses the raw string output from the scripts into a structured dictionary.
    
    Args:
        output_str (str): The captured standard output from a script.

    Returns:
        dict: A dictionary with keys 'nodes', 'graph_score', and 'edges'.
    """
    parsed_data = {
        'nodes': [],
        'graph_score': None,
        'edges': []
    }
    
    # Find all floating point or integer numbers in the string
    numbers = [float(num) for num in re.findall(r'-?\d+\.\d+|-?\d+', output_str)]
    
    # Heuristically parse based on the known output format.
    # This is brittle and depends on the exact output structure.
    try:
        # Example format: 5 nodes, 4 features each = 20 node values
        # The first (5 nodes * 4 features) numbers are node features.
        # This needs to be adjusted based on the actual number of nodes and features.
        # A more robust way would be to parse section by section.
        
        # Let's find sections to be more robust
        lines = output_str.strip().split('\n')
        current_section = None
        
        for line in lines:
            if 'Node-Level' in line or 'Node Features' in line:
                current_section = 'nodes'
                continue
            elif 'Graph-Level' in line:
                current_section = 'graph'
                continue
            elif 'Edge-Level' in line:
                current_section = 'edges'
                continue

            if current_section == 'nodes':
                node_vals = [float(v) for v in line.split()]
                if node_vals:
                    parsed_data['nodes'].append(node_vals)
            elif current_section == 'graph':
                if 'Score =' in line:
                    parsed_data['graph_score'] = float(re.findall(r'-?\d+\.\d+', line)[0])
            elif current_section == 'edges':
                 if 'score =' in line:
                    edge_score = float(re.findall(r'-?\d+\.\d+', line)[0])
                    parsed_data['edges'].append(edge_score)

    except (ValueError, IndexError) as e:
        print(f"Error parsing output: {e}\nOutput was:\n{output_str}")
        return None
        
    # Sort edges for consistent comparison, as their order might differ
    parsed_data['edges'].sort()
    return parsed_data


def compare_results(data1, data2):
    """
    Compares two parsed data dictionaries with a tolerance for floats.

    Args:
        data1 (dict): Parsed data from the first command.
        data2 (dict): Parsed data from the second command.

    Returns:
        bool: True if the data is considered identical, False otherwise.
    """
    if data1 is None or data2 is None:
        return False

    # 1. Compare Graph Score
    if abs(data1['graph_score'] - data2['graph_score']) > TOLERANCE:
        print(f"Mismatch in Graph Score: {data1['graph_score']} vs {data2['graph_score']}")
        return False

    # 2. Compare Node Features
    if len(data1['nodes']) != len(data2['nodes']):
        print(f"Mismatch in number of nodes: {len(data1['nodes'])} vs {len(data2['nodes'])}")
        return False
    
    for i, (node1, node2) in enumerate(zip(data1['nodes'], data2['nodes'])):
        if len(node1) != len(node2):
            print(f"Mismatch in feature count for node {i}: {len(node1)} vs {len(node2)}")
            return False
        for j, (v1, v2) in enumerate(zip(node1, node2)):
            if abs(v1 - v2) > TOLERANCE:
                print(f"Mismatch in Node {i}, Feature {j}: {v1} vs {v2}")
                return False

    # 3. Compare Edge Scores
    if len(data1['edges']) != len(data2['edges']):
        print(f"Mismatch in number of edges: {len(data1['edges'])} vs {len(data2['edges'])}")
        return False
        
    # Edges were sorted during parsing, so we can compare them directly.
    for i, (score1, score2) in enumerate(zip(data1['edges'], data2['edges'])):
        if abs(score1 - score2) > TOLERANCE:
            print(f"Mismatch in sorted Edge {i} score: {score1} vs {score2}")
            return False

    return True

# --- Main Execution ---

if __name__ == "__main__":
    # Run the Python script
    ret1, out1, err1 = run_command(CMD1, input_data=CMD1_INPUT)
    
    # Run the C++ executable
    ret2, out2, err2 = run_command(CMD2)

    print("\n--- Parsing Outputs ---")
    parsed_out1 = parse_output(out1)
    print("Parsed data from Python script:", parsed_out1)
    
    parsed_out2 = parse_output(out2)
    print("Parsed data from C++ executable:", parsed_out2)
    
    print("\n--- Comparison ---")
    are_outputs_equal = compare_results(parsed_out1, parsed_out2)

    if are_outputs_equal:
        print("\nResult: Outputs are consistent.")
    else:
        print("\nResult: Outputs are DIFFERENT.")

    print(f"\nFinal Answer: {are_outputs_equal}")

