#!/usr/bin/env python3
"""
ExLlamaV2 + LAMMPS MCP Integration
Allows a local LLM to control LAMMPS via MCP tools
"""

import json
import subprocess
import re
from typing import List, Dict, Any, Optional
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache_Q4, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler

class MCPClient:
    """Client for communicating with MCP servers"""
    
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.process = None
        self.tools = {}
        self._start_server()
        self._load_tools()
    
    def _start_server(self):
        """Start the MCP server process"""
        self.process = subprocess.Popen(
            ['python3', self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        print(f"✓ MCP Server started: {self.server_script}")
    
    def _load_tools(self):
        """Load available tools from the server"""
        request = {"method": "tools/list", "params": {}}
        response = self._send_request(request)
        
        if response and "tools" in response:
            for tool in response["tools"]:
                self.tools[tool["name"]] = tool
            print(f"✓ Loaded {len(self.tools)} tools: {list(self.tools.keys())}")
        else:
            print("✗ Failed to load tools")
    
    def _send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request to MCP server and get response"""
        try:
            request_json = json.dumps(request) + '\n'
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline().strip()
            if response_line:
                return json.loads(response_line)
            return None
        except Exception as e:
            print(f"✗ MCP request failed: {e}")
            return None
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return the result as a string"""
        request = {
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = self._send_request(request)
        
        if response and "content" in response:
            return response["content"][0]["text"]
        elif response and "error" in response:
            return json.dumps({"error": response["error"]})
        else:
            return json.dumps({"error": "No response from tool"})
    
    def get_tools_description(self) -> str:
        """Get a description of all available tools for the LLM"""
        tools_desc = "Available LAMMPS tools:\n\n"
        for name, tool in self.tools.items():
            tools_desc += f"- {name}: {tool['description']}\n"
            if 'inputSchema' in tool and 'properties' in tool['inputSchema']:
                props = tool['inputSchema']['properties']
                if props:
                    tools_desc += "  Parameters:\n"
                    for param, details in props.items():
                        tools_desc += f"    * {param}: {details.get('description', 'No description')}\n"
        return tools_desc
    
    def close(self):
        """Close the MCP server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("✓ MCP Server closed")


class LAMMPSAgent:
    """AI Agent that uses ExLlamaV2 LLM to control LAMMPS via MCP"""
    
    def __init__(self, model_dir: str, mcp_server_script: str):
        print("Initializing LAMMPS AI Agent...")
        print("=" * 60)
        
        # Initialize MCP Client
        self.mcp = MCPClient(mcp_server_script)
        
        # Initialize ExLlamaV2
        print("\nLoading LLM model...")
        config = ExLlamaV2Config(model_dir)
        config.arch_compat_overrides()
        
        self.model = ExLlamaV2(config)
        self.cache = ExLlamaV2Cache_Q4(self.model, max_seq_len=8192, lazy=True)
        self.model.load_autosplit(self.cache, progress=True)
        
        print("Loading tokenizer...")
        self.tokenizer = ExLlamaV2Tokenizer(config)
        
        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )
        
        print("Warming up generator...")
        self.generator.warmup()
        
        # System prompt with tool descriptions
        self.system_prompt = self._create_system_prompt()
        
        print("\n" + "=" * 60)
        print("✓ LAMMPS AI Agent ready!")
        print("=" * 60 + "\n")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with tool calling instructions"""
        return f"""You are a scientific AI assistant with access to LAMMPS molecular dynamics simulation tools.

{self.mcp.get_tools_description()}

When you need to use a tool, respond with a tool call in this exact format:
<tool_call>
{{"tool": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

After receiving tool results, analyze them and provide insights to the user.

You can use multiple tools in sequence to accomplish complex tasks. Always explain what you're doing and why."""
    
    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse tool call: {match}")
        
        return tool_calls
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Execute tool calls and return results"""
        results = []
        
        for call in tool_calls:
            tool_name = call.get("tool")
            arguments = call.get("arguments", {})
            
            print(f"  → Calling tool: {tool_name}")
            result = self.mcp.call_tool(tool_name, arguments)
            results.append(f"<tool_result tool='{tool_name}'>\n{result}\n</tool_result>")
        
        return "\n\n".join(results)
    
    def chat(self, user_message: str, max_turns: int = 5) -> str:
        """
        Chat with the agent, allowing it to use tools
        
        Args:
            user_message: The user's request
            max_turns: Maximum number of tool-calling turns
        
        Returns:
            Final response from the agent
        """
        print(f"\n{'='*60}")
        print(f"USER: {user_message}")
        print(f"{'='*60}\n")
        
        # Build conversation
        conversation = f"{self.system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        for turn in range(max_turns):
            print(f"[Turn {turn + 1}] Generating response...")
            
            # Generate response
            response = self.generator.generate(
                prompt=conversation,
                max_new_tokens=1000,
                add_bos=False,
                stop_conditions=[self.tokenizer.eos_token_id],
                gen_settings=ExLlamaV2Sampler.Settings(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                )
            )
            
            # Remove the prompt from response
            response = response[len(conversation):].strip()
            
            print(f"\n{'─'*60}")
            print(f"AGENT: {response}")
            print(f"{'─'*60}\n")
            
            # Check for tool calls
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                # No more tool calls, return final response
                return response
            
            # Execute tools
            print(f"Executing {len(tool_calls)} tool call(s)...")
            tool_results = self._execute_tool_calls(tool_calls)
            
            # Add to conversation
            conversation += f" {response}\n\n{tool_results}\n\nAssistant:"
        
        return "Maximum turns reached. Task may be incomplete."
    
    def close(self):
        """Clean up resources"""
        self.mcp.close()


# Example usage
if __name__ == "__main__":
    # Configure your paths
    MODEL_DIR = "/media/es205/NVMe/LLM/Qwen2.5-14B-Instruct-exl2-6_5/"
    MCP_SERVER = "lammps_mcp_server.py"
    
    # Create agent
    agent = LAMMPSAgent(MODEL_DIR, MCP_SERVER)
    
    try:
        # Example 1: Simple task
        print("\n" + "="*80)
        print("EXAMPLE 1: Create a simple atomic system")
        print("="*80)
        
        response = agent.chat(
            "Create a simple FCC lattice with 1000 atoms in a 10x10x10 box. "
            "Use Lennard-Jones units and report how many atoms were created."
        )
        
        print("\n" + "="*80)
        print("FINAL ANSWER:")
        print(response)
        print("="*80)
        
        # Example 2: Run a simulation
        print("\n\n" + "="*80)
        print("EXAMPLE 2: Run a molecular dynamics simulation")
        print("="*80)
        
        response = agent.chat(
            "Now run a short NVE molecular dynamics simulation for 1000 timesteps "
            "with a timestep of 0.001. Set up a Lennard-Jones pair potential and "
            "give all atoms an initial velocity at temperature 1.44. "
            "Then get the final atom positions and thermodynamic data."
        )
        
        print("\n" + "="*80)
        print("FINAL ANSWER:")
        print(response)
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        agent.close()
