#!/usr/bin/env python3
"""
Test script for LAMMPS MCP Server
"""
import json
import subprocess
import sys
import time

def test_lammps_mcp():
    print("Starting LAMMPS MCP Server test...\n")
    
    # Start the MCP server
    proc = subprocess.Popen(
        ['python3', 'lammps_mcp_server.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Give server time to start
    time.sleep(1)
    
    def send_request(request_dict):
        """Send a request and get response"""
        request_json = json.dumps(request_dict)
        print(f"→ Sending: {request_dict['method']}")
        
        proc.stdin.write(request_json + '\n')
        proc.stdin.flush()
        
        # Read response with timeout
        try:
            response_line = proc.stdout.readline()
            
            if not response_line:
                print("✗ No response received (empty line)")
                return None
            
            response_line = response_line.strip()
            
            if not response_line:
                print("✗ Empty response")
                return None
            
            try:
                response = json.loads(response_line)
                print(f"✓ Response received\n")
                return response
            except json.JSONDecodeError as e:
                print(f"✗ Invalid JSON response: {e}")
                print(f"   Raw response: {response_line[:200]}")
                return None
                
        except Exception as e:
            print(f"✗ Error reading response: {e}")
            return None
    
    try:

        # Test 1: List available tools
        print("=" * 60)
        print("TEST 1: List Available Tools")
        print("=" * 60)
        request = {"method": "tools/list", "params": {}}
        response = send_request(request)
        
        if response and "tools" in response:
            print(f"Available tools ({len(response['tools'])}):")
            for tool in response['tools']:
                print(f"  - {tool['name']}: {tool['description']}")
        else:
            print("Failed to get tools list")
            return
        
        # Test 2: Initialize LAMMPS
        print("\n" + "=" * 60)
        print("TEST 2: Initialize LAMMPS")
        print("=" * 60)
        request = {
            "method": "tools/call",
            "params": {
                "name": "lammps_initialize",
                "arguments": {}
            }
        }
        response = send_request(request)
        
        if response and "content" in response:
            content = json.loads(response["content"][0]["text"])
            print(f"Status: {content['status']}")
            print(f"Message: {content['message']}")
            
            if content['status'] != 'success':
                print("\n LAMMPS initialization failed!")
                print("Make sure LAMMPS is installed with Python interface:")
                print("  pip install lammps")
                print("  OR build from source with -DPKG_PYTHON=ON")
                return
        else:
            print("Failed to initialize LAMMPS")
            return
        
        # Test 3: Run simple LAMMPS commands
        print("\n" + "=" * 60)
        print("TEST 3: Run Simple LAMMPS Commands")
        print("=" * 60)
        request = {
            "method": "tools/call",
            "params": {
                "name": "lammps_run_commands",
                "arguments": {
                    "commands": [
                        "units lj",
                        "atom_style atomic",
                        "lattice fcc 0.8442",
                        "region box block 0 4 0 4 0 4",
                        "create_box 1 box",
                        "create_atoms 1 box",
                        "mass            1 1.0",
                        "velocity        all create 1.44 87287 loop geom",
                        "pair_style      lj/cut 2.5",
                        "pair_coeff      1 1 1.0 1.0 2.5",
                        "fix             1 all nve",
                        "run             10"
                    ]
                }
            }
        }
        response = send_request(request)
        
        if response and "content" in response:
            content = json.loads(response["content"][0]["text"])
            print(f"Status: {content['status']}")
            if content['status'] == 'success':
                print(f"Commands executed: {len(content['results'])}")
                for result in content['results'][:3]:  # Show first 3
                    print(f"{result['command']}")
        else:
            print("Failed to run commands")
            return
        
        # Test 4: Get system info
        print("\n" + "=" * 60)
        print("TEST 4: Get Thermodynamic Data")
        print("=" * 60)
        request = {
            "method": "tools/call",
            "params": {
                "name": "lammps_get_thermo",
                "arguments": {}
            }
        }
        response = send_request(request)
        
        if response and "content" in response:
            content = json.loads(response["content"][0]["text"])
            print(f"Status: {content['status']}")
            if content['status'] == 'success':
                print(f"Number of atoms: {content.get('natoms', 'N/A')}")
                print(f"Temperature: {content.get('temperature', 'N/A')}")
                print(f"Potential Energy: {content.get('potential_energy', 'N/A')}")
                print(f"Pressure: {content.get('pressure', 'N/A')}")
        
        # Test 5: Get atom positions
        print("\n" + "=" * 60)
        print("TEST 5: Get Atom Positions")
        print("=" * 60)
        request = {
            "method": "tools/call",
            "params": {
                "name": "lammps_get_positions",
                "arguments": {}
            }
        }
        response = send_request(request)
        
        if response and "content" in response:
            content = json.loads(response["content"][0]["text"])
            print(f"Status: {content['status']}")
            if content['status'] == 'success':
                print(f"Total atoms: {content['natoms']}")
                print(f"Positions retrieved: {len(content['positions'])}")
                if content['positions']:
                    print("First 10 atom positions:")
                    for i in range(10):
                        pos = content['positions'][i]
                        print(f"  Atom {pos['id']}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down MCP server...")
        proc.terminate()
        proc.wait(timeout=5)

if __name__ == "__main__":
    test_lammps_mcp()
