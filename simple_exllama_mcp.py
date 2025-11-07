#!/usr/bin/env python3
"""
Minimal example: ExLlamaV2 + LAMMPS MCP
"""

import json
import subprocess
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache_Q4, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator

# === Configuration ===
MODEL_DIR = "/media/es205/NVMe/LLM/Qwen2.5-14B-Instruct-exl2-6_5/"
MCP_SERVER = "lammps_mcp_server.py"

# === 1. Start MCP Server ===
print("Starting MCP server...")
mcp_process = subprocess.Popen(
    ['python3', MCP_SERVER],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

def call_mcp_tool(tool_name, arguments=None):
    """Simple function to call MCP tools"""
    request = {
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments or {}
        }
    }
    
    mcp_process.stdin.write(json.dumps(request) + '\n')
    mcp_process.stdin.flush()
    
    response = json.loads(mcp_process.stdout.readline())
    if "content" in response:
        return response["content"][0]["text"]
    return str(response)

# === 2. Load LLM ===
print("Loading model...")
config = ExLlamaV2Config(MODEL_DIR)
config.arch_compat_overrides()

model = ExLlamaV2(config)
cache = ExLlamaV2Cache_Q4(model, max_seq_len=8192, lazy=True)
model.load_autosplit(cache, progress=True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

generator = ExLlamaV2DynamicGenerator(
    model=model,
    cache=cache,
    tokenizer=tokenizer,
)

print("Warming up...")
generator.warmup()

# === 3. Create a prompt that explains the tools ===
system_prompt = """You are a helpful AI assistant with access to LAMMPS molecular dynamics tools.

Available tools:
- lammps_initialize: Initialize LAMMPS
- lammps_run_commands: Run LAMMPS commands (takes "commands" array)
- lammps_get_thermo: Get thermodynamic data
- lammps_get_positions: Get atom positions

To use a tool, respond with: USE_TOOL: tool_name | {"argument": "value"}

Example: USE_TOOL: lammps_run_commands | {"commands": ["units lj", "atom_style atomic"]}
"""

# === 4. Simple agentic loop ===
def run_agent(user_query, max_iterations=5):
    """Simple agent loop"""
    conversation = f"{system_prompt}\n\nUser: {user_query}\n\nAssistant:"
    
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        # Generate response
        response = generator.generate(
            prompt=conversation,
            max_new_tokens=500,
            add_bos=False,
        )
        
        # Extract just the new text
        new_text = response[len(conversation):].strip()
        print(f"LLM says: {new_text[:]}...")
        
        # Check if LLM wants to use a tool
        if "USE_TOOL:" in new_text:
            # Parse tool call (simple parsing)
            try:
                tool_part = new_text.split("USE_TOOL:")[1].split("\n")[0].strip()
                tool_name, args_str = tool_part.split("|")
                tool_name = tool_name.strip()
                arguments = json.loads(args_str.strip()) if args_str.strip() else {}
                
                print(f"Calling tool: {tool_name}")
                result = call_mcp_tool(tool_name, arguments)
                print(f"Tool result: {result[:]}...")
                
                # Add result to conversation
                conversation += f" {new_text}\n\nTool Result: {result}\n\nAssistant:"
                
            except Exception as e:
                print(f"Error parsing tool call: {e}")
                break
        else:
            # No tool call, we're done
            print(f"\nFinal answer: {new_text}")
            return new_text
    
    return "Max iterations reached"

# === 5. Run example ===
try:
    print("\n" + "="*60)
    print("EXAMPLE: Ask the LLM to create a LAMMPS system")
    print("="*60)
    
    response = run_agent(
        "Initialize LAMMPS and create a simple FCC lattice with 100 atoms. "
        "Tell me how many atoms were created and display the positions of all atoms."
    )
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    mcp_process.terminate()
    print("MCP server closed")
