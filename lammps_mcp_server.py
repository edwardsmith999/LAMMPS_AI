#!/usr/bin/env python3
"""
LAMMPS MCP (Model Context Protocol) Server
This allows Claude or other AI agents to interact with LAMMPS via MCP
"""

import json
import sys
import asyncio
from typing import Any, Dict, List
import os
import tempfile
import logging
from datetime import datetime

# Configure logging to stderr (not stdout, which is used for MCP)
#logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(asctime)s - %(levelname)s - %(message)s',
#    stream=sys.stderr
#)
#logger = logging.getLogger(__name__)

log_file = "mcp_server.log"

# Create format
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create handlers
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(log_format)

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_format)

# Configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(stderr_handler)
logger.addHandler(file_handler)

logger.info(f"Logging MCP output to: {log_file}")


# MCP Server Implementation
class LAMMPSMCPServer:
    def __init__(self):
        self.lmp = None
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Temp directory created: {self.temp_dir}")
        
    def initialize_lammps(self):
        """Initialize LAMMPS instance"""
        try:
            # Import here to catch errors properly
            from lammps import lammps

            # Initialize with no screen output
            self.lmp = lammps(cmdargs=['-screen', 'none', '-log', str(datetime.now())+'lammps.log'])
            logger.info("LAMMPS initialized successfully")
            return {"status": "success", "message": "LAMMPS initialized"}
        except ImportError as e:
            logger.error(f"LAMMPS import failed: {e}")
            return {"status": "error", "message": f"LAMMPS not installed or not in Python path: {str(e)}"}
        except Exception as e:
            logger.error(f"LAMMPS initialization failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_lammps_commands(self, commands: List[str]) -> Dict[str, Any]:
        """Execute a list of LAMMPS commands"""

        if not self.lmp:
            init_result = self.initialize_lammps()
            if init_result["status"] == "error":
                return init_result
        
        results = []
        try:
            for cmd in commands:
                logger.debug(f"Executing command: {cmd}")
                self.lmp.command(cmd)
                results.append({"command": cmd, "status": "executed"})
            return {"status": "success", "results": results}
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"status": "error", "message": str(e), "results": results}
    
    def run_lammps_script(self, script: str) -> Dict[str, Any]:
        """Execute a complete LAMMPS input script"""
        if not self.lmp:
            init_result = self.initialize_lammps()
            if init_result["status"] == "error":
                return init_result
       
        # Write script to temporary file
        script_path = os.path.join(self.temp_dir, "input.lammps")
        with open(script_path, 'w') as f:
            f.write(script)
        
        logger.info(f"Executing script from {script_path}")
        
        try:
            self.lmp.file(script_path)
            return {"status": "success", "message": "Script executed successfully"}
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_thermo_data(self) -> Dict[str, Any]:
        """Extract thermodynamic data from LAMMPS"""
        if not self.lmp:
            return {"status": "error", "message": "LAMMPS not initialized"}
        
        try:
            # Get basic system info
            natoms = self.lmp.get_natoms()
            
            # Try to get thermo data (may not be available depending on simulation state)
            try:
                temp = self.lmp.extract_compute("thermo_temp", 0, 0)
                pe = self.lmp.extract_compute("thermo_pe", 0, 0)
                press = self.lmp.extract_compute("thermo_press", 0, 0)
            except:
                temp = pe = press = None
            
            return {
                "status": "success",
                "natoms": natoms,
                "temperature": temp,
                "potential_energy": pe,
                "pressure": press
            }
        except Exception as e:
            logger.error(f"Failed to get thermo data: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_atom_positions(self) -> Dict[str, Any]:
        """Get positions of all atoms"""
        if not self.lmp:
            return {"status": "error", "message": "LAMMPS not initialized"}
        
        try:
            natoms = self.lmp.get_natoms()
            
            if natoms == 0:
                return {"status": "success", "natoms": 0, "positions": []}
            
            # Gather atom positions
            x = self.lmp.gather_atoms("x", 1, 3)
            
            positions = []
            # Only return first 100 atoms to avoid huge responses
            max_atoms = min(natoms, 100)
            for i in range(max_atoms):
                positions.append({
                    "id": i,
                    "x": float(x[i*3]),
                    "y": float(x[i*3+1]),
                    "z": float(x[i*3+2])
                })
            
            return {
                "status": "success", 
                "natoms": natoms, 
                "positions": positions,
                "note": f"Showing first {max_atoms} of {natoms} atoms"
            }
        except Exception as e:
            logger.error(f"Failed to get atom positions: {e}")
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """Clean up LAMMPS instance"""
        if self.lmp:
            try:
                self.lmp.close()
                logger.info("LAMMPS closed")
            except:
                pass
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        logger.info("Temp directory cleaned up")

# MCP Protocol Handler
async def handle_mcp_request(server: LAMMPSMCPServer, request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle incoming MCP requests"""
    method = request.get("method")
    params = request.get("params", {})
    
    logger.debug(f"Handling request: {method}")
    
    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "lammps_initialize",
                    "description": "Initialize a new LAMMPS simulation instance",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "lammps_run_commands",
                    "description": "Execute a list of LAMMPS commands",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "commands": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of LAMMPS commands to execute"
                            }
                        },
                        "required": ["commands"]
                    }
                },
                {
                    "name": "lammps_run_script",
                    "description": "Execute a complete LAMMPS input script",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "script": {
                                "type": "string",
                                "description": "Complete LAMMPS input script"
                            }
                        },
                        "required": ["script"]
                    }
                },
                {
                    "name": "lammps_get_thermo",
                    "description": "Get current thermodynamic properties",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "lammps_get_positions",
                    "description": "Get positions of all atoms",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }
    
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        
        if tool_name == "lammps_initialize":
            result = server.initialize_lammps()
        elif tool_name == "lammps_run_commands":
            result = server.run_lammps_commands(tool_args.get("commands", []))
        elif tool_name == "lammps_run_script":
            result = server.run_lammps_script(tool_args.get("script", ""))
        elif tool_name == "lammps_get_thermo":
            result = server.get_thermo_data()
        elif tool_name == "lammps_get_positions":
            result = server.get_atom_positions()
        else:
            result = {"status": "error", "message": f"Unknown tool: {tool_name}"}
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }
            ]
        }
    
    return {"error": f"Unknown method: {method}"}

# Main MCP Server Loop
async def main():
    server = LAMMPSMCPServer()
    logger.info("LAMMPS MCP Server starting...")

    try:
        # Read from stdin and write to stdout (MCP protocol)
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                
                if not line:
                    logger.info("EOF received, shutting down")
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                logger.debug(f"Received request: {line[:100]}...")
                
                request = json.loads(line)
                response = await handle_mcp_request(server, request)
                
                # Write response to stdout (JSON must be on single line)
                response_json = json.dumps(response)
                print(response_json, flush=True)
                logger.debug(f"Sent response: {response_json[:100]}...")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                error_response = {"error": f"Invalid JSON: {str(e)}"}
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                logger.error(f"Server error: {e}", exc_info=True)
                error_response = {"error": f"Server error: {str(e)}"}
                print(json.dumps(error_response), flush=True)
    
    finally:
        logger.info("Cleaning up...")
        server.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
