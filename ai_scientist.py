#!/usr/bin/env python3
"""
Simple AI agent that uses LAMMPS MCP to run molecular dynamics
"""
import json
import subprocess

class LAMMPSAgent:
    def __init__(self):
        self.mcp = subprocess.Popen(
            ['python3', 'lammps_mcp_server.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
    
    def call_tool(self, tool_name, arguments=None):
        """Call an MCP tool"""
        request = {
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }
        self.mcp.stdin.write(json.dumps(request) + '\n')
        self.mcp.stdin.flush()
        
        response = self.mcp.stdout.readline()
        return json.loads(response)
    
    def run_simple_md(self):
        """Run a simple molecular dynamics simulation"""
        print("AI Agent: Initializing LAMMPS...")
        result = self.call_tool("lammps_initialize")
        print(f"   Result: {result['content'][0]['text']}")
        
        print("\nAI Agent: Creating atomic system...")
        script = """
        # LJ liquid simulation
        units lj
        atom_style atomic
        
        lattice fcc 0.8442
        region box block 0 10 0 10 0 10
        create_box 1 box
        create_atoms 1 box
        mass 1 1.0
        
        velocity all create 1.44 87287 loop geom
        
        pair_style lj/cut 2.5
        pair_coeff 1 1 1.0 1.0 2.5
        
        neighbor 0.3 bin
        neigh_modify delay 0 every 20 check no
        
        fix 1 all nve
        
        thermo 100
        run 10000
        """
        
        result = self.call_tool("lammps_run_script", {"script": script})
        print(f"   Result: {result['content'][0]['text']}")
        
        print("\nAI Agent: Extracting thermodynamic data...")
        result = self.call_tool("lammps_get_thermo")
        print(f"   Result: {result['content'][0]['text']}")
        
        print("\nAI Agent: Getting atom positions...")
        result = self.call_tool("lammps_get_positions")
        data = json.loads(result['content'][0]['text'])
        print(f"   Number of atoms: {data.get('natoms', 'N/A')}")
        
        print("\nAI Agent: Simulation complete!")
    
    def close(self):
        self.mcp.terminate()

if __name__ == "__main__":
    agent = LAMMPSAgent()
    try:
        agent.run_simple_md()
    finally:
        agent.close()
