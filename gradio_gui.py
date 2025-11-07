#!/usr/bin/env python3
"""
Gradio Chat Interface for LAMMPS AI Agent with Visualization
Integrates ExLlamaV2 + MCP + Gradio for interactive molecular dynamics
"""

import gradio as gr
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
import io
import base64
from PIL import Image
from typing import List, Tuple, Optional
import re

# Import from the exllama_mcp_client
from exllama_mcp_client import LAMMPSAgent, MCPClient

class LAMMPSChatInterface:
    """Gradio interface for LAMMPS AI Agent with visualization"""
    
    def __init__(self, model_dir: str, mcp_server_script: str):
        print("Initializing LAMMPS Chat Interface...")
        self.agent = LAMMPSAgent(model_dir, mcp_server_script)
        self.conversation_history = []
        self.current_positions = None
        self.current_thermo = None
        self.lammps_logs = []
        
    def parse_lammps_output(self, response: str) -> dict:
        """Parse LAMMPS tool results from agent response"""
        results = {
            'positions': None,
            'thermo': None,
            'logs': []
        }
        
        # Extract tool results
        tool_results = re.findall(r'<tool_result tool=\'(\w+)\'>\s*(.*?)\s*</tool_result>', 
                                 response, re.DOTALL)
        
        for tool_name, result_text in tool_results:
            try:
                result_data = json.loads(result_text)
                
                if tool_name == 'lammps_get_positions' and result_data.get('status') == 'success':
                    results['positions'] = result_data.get('positions', [])
                    results['logs'].append(f"Retrieved {len(results['positions'])} atom positions")
                
                elif tool_name == 'lammps_get_thermo' and result_data.get('status') == 'success':
                    results['thermo'] = result_data
                    results['logs'].append(f"Thermodynamics: T={result_data.get('temperature')}, PE={result_data.get('potential_energy')}")
                
                elif tool_name == 'lammps_run_commands' and result_data.get('status') == 'success':
                    num_commands = len(result_data.get('results', []))
                    results['logs'].append(f"Executed {num_commands} LAMMPS commands")
                
                elif tool_name == 'lammps_run_script' and result_data.get('status') == 'success':
                    results['logs'].append("LAMMPS script executed successfully")
                    
            except json.JSONDecodeError:
                continue
        
        return results
    
    def create_2d_visualization(self, positions: List[dict]) -> Optional[Figure]:
        """Create 2D matplotlib visualization of atom positions"""
        if not positions:
            return None
        
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        x = [p['x'] for p in positions]
        y = [p['y'] for p in positions]
        
        ax.scatter(x, y, c='blue', alpha=0.6, s=50, edgecolors='darkblue', linewidth=0.5)
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        ax.set_title(f'Atom Positions (2D projection, {len(positions)} atoms)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return fig
    
    def create_3d_visualization(self, positions: List[dict]) -> Optional[go.Figure]:
        """Create interactive 3D plotly visualization"""
        if not positions:
            return None
        
        x = [p['x'] for p in positions]
        y = [p['y'] for p in positions]
        z = [p['z'] for p in positions]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Z position"),
                line=dict(color='darkblue', width=0.5)
            ),
            text=[f"Atom {p['id']}<br>x: {p['x']:.2f}<br>y: {p['y']:.2f}<br>z: {p['z']:.2f}" 
                  for p in positions],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f'3D Atom Positions ({len(positions)} atoms)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=800,
            template='plotly_dark'
        )
        
        return fig
    
    def format_thermo_data(self, thermo: dict) -> str:
        """Format thermodynamic data as text"""
        if not thermo:
            return "No thermodynamic data available"
        
        lines = ["### Thermodynamic Properties\n"]
        
        if 'natoms' in thermo:
            lines.append(f"**Number of atoms:** {thermo['natoms']}")
        if thermo.get('temperature') is not None:
            lines.append(f"**Temperature:** {thermo['temperature']:.4f} K")
        if thermo.get('potential_energy') is not None:
            lines.append(f"**Potential Energy:** {thermo['potential_energy']:.4f}")
        if thermo.get('pressure') is not None:
            lines.append(f"**Pressure:** {thermo['pressure']:.4f}")
        
        return "\n".join(lines)
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, Optional[Figure], Optional[go.Figure], str]:
        """
        Process chat message and return updated UI components
        
        Returns:
            - Updated chat history
            - LAMMPS logs
            - 2D plot
            - 3D plot  
            - Thermodynamic data
        """
        if not message.strip():
            return history, "", None, None, ""
        
        # Get response from agent
        try:
            response = self.agent.chat(message, max_turns=5)
            
            # Parse LAMMPS outputs
            parsed = self.parse_lammps_output(response)
            
            # Update internal state
            if parsed['positions']:
                self.current_positions = parsed['positions']
            if parsed['thermo']:
                self.current_thermo = parsed['thermo']
            self.lammps_logs.extend(parsed['logs'])
            
            # Clean response for display (remove tool XML tags)
            clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
            clean_response = re.sub(r'<tool_result.*?</tool_result>', '', clean_response, flags=re.DOTALL)
            clean_response = clean_response.strip()
            
            # Update history
            history.append((message, clean_response))
            
            # Create visualizations
            plot_2d = self.create_2d_visualization(self.current_positions)
            plot_3d = self.create_3d_visualization(self.current_positions)
            
            # Format outputs
            logs_text = "\n".join([f"• {log}" for log in self.lammps_logs[-10:]])  # Last 10 logs
            thermo_text = self.format_thermo_data(self.current_thermo)
            
            return history, logs_text, plot_2d, plot_3d, thermo_text
            
        except Exception as e:
            error_msg = f" Error: {str(e)}"
            history.append((message, error_msg))
            return history, error_msg, None, None, ""
    
    def clear_history(self):
        """Clear conversation and reset state"""
        self.conversation_history = []
        self.current_positions = None
        self.current_thermo = None
        self.lammps_logs = []
        return [], "", None, None, ""
    
    def get_example_queries(self) -> List[List[str]]:
        """Get example queries for the interface"""
        return [
            ["Initialize LAMMPS and create a simple FCC lattice with 500 atoms in a 8x8x8 box"],
            ["Run a molecular dynamics simulation with Lennard-Jones potential for 1000 steps"],
            ["Show me the current atom positions and thermodynamic properties"],
            ["Create a 2D square lattice with 100 atoms and visualize it"],
            ["Run an NVT simulation at temperature 1.5 for 2000 timesteps and analyze the results"],
        ]


def create_gradio_interface(model_dir: str, mcp_server_script: str):
    """Create and launch Gradio interface"""
    
    interface = LAMMPSChatInterface(model_dir, mcp_server_script)
    
    with gr.Blocks(title="LAMMPS AI Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # LAMMPS AI Agent Chat Interface
        
        Interact with LAMMPS molecular dynamics simulations using natural language.
        The AI agent will execute LAMMPS commands and visualize results in real-time.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=True,
                    avatar_images=(None, "🤖")
                )
                
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask me to run LAMMPS simulations...",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)
                
                # Example queries
                gr.Examples(
                    examples=interface.get_example_queries(),
                    inputs=msg,
                    label="Example Queries"
                )
                
                # LAMMPS logs
                logs = gr.Textbox(
                    label="LAMMPS Activity Log",
                    lines=5,
                    max_lines=10,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                # Thermodynamic data
                thermo = gr.Markdown(
                    label="Thermodynamic Properties",
                    value="*No data yet*"
                )
                
                # Visualizations in tabs
                with gr.Tabs():
                    with gr.Tab("2D View"):
                        plot_2d = gr.Plot(label="2D Projection")
                    
                    with gr.Tab("3D View"):
                        plot_3d = gr.Plot(label="Interactive 3D View")
        
        # Event handlers
        def submit_message(message, history):
            return interface.chat(message, history)
        
        submit_btn.click(
            fn=submit_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, logs, plot_2d, plot_3d, thermo]
        ).then(
            fn=lambda: "",
            outputs=msg
        )
        
        msg.submit(
            fn=submit_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, logs, plot_2d, plot_3d, thermo]
        ).then(
            fn=lambda: "",
            outputs=msg
        )
        
        clear_btn.click(
            fn=interface.clear_history,
            outputs=[chatbot, logs, plot_2d, plot_3d, thermo]
        )
        
        gr.Markdown("""
        ---
        ### Tips:
        - Ask the agent to initialize LAMMPS first
        - Request specific simulations (NVE, NVT, etc.)
        - Ask for visualizations and thermodynamic data
        - Be specific about lattice types, number of atoms, and simulation parameters
        """)
    
    return demo


if __name__ == "__main__":
    # Configuration
    MODEL_DIR = "/media/es205/NVMe/LLM/Qwen2.5-14B-Instruct-exl2-6_5/"
    MCP_SERVER = "lammps_mcp_server.py"
    
    print("Starting Gradio interface...")
    print("This will load the model and may take a few minutes...\n")
    
    try:
        demo = create_gradio_interface(MODEL_DIR, MCP_SERVER)
        
        # Launch with sharing options
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,
            share=False,  # Set to True for public URL
            inbrowser=True  # Auto-open browser
        )
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
