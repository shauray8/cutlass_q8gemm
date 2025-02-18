import subprocess
import xml.etree.ElementTree as ET
import json
import time
from datetime import datetime
import argparse
import os
import psutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from threading import Thread
import logging
from typing import Dict, List, Any

class EnhancedSystemMonitor:
    def __init__(self, output_file="system_metrics.json", gpu_stats_file="gpu_stats.json"):
        self.output_file = output_file
        self.gpu_stats_file = gpu_stats_file
        self.metrics = []
        self.running = False
        self.monitor_thread = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """Start the monitoring process"""
        self.logger.info("Starting monitoring...")
        self.running = True
        self.monitor_thread = Thread(target=self.collect_metrics)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.logger.info("Stopping monitoring...")
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()

    def collect_metrics(self):
        """Collect all system metrics"""
        while self.running:
            try:
                gpu_root = self.get_gpu_info()
                gpu_stats = []
                if gpu_root:
                    for i, gpu in enumerate(gpu_root.findall('gpu')):
                        stats = self.parse_gpu_stats(gpu)
                        stats['gpu_id'] = i
                        gpu_stats.append(stats)

                current_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu': self.get_cpu_info(),
                    'gpu': gpu_stats,
                    'memory': dict(psutil.virtual_memory()._asdict()),
                    'network': dict(psutil.net_io_counters()._asdict()),
                    'disk': {disk: dict(psutil.disk_io_counters(perdisk=True)[disk]._asdict()) 
                            for disk in psutil.disk_io_counters(perdisk=True)}
                }
                
                self.metrics.append(current_metrics)
                self.save_metrics()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                time.sleep(1)

    def get_gpu_info(self):
        """Fetch NVIDIA GPU details in XML format."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '-q', '-x'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return ET.fromstring(result.stdout)
        except Exception as e:
            self.logger.error(f"Error getting GPU info: {str(e)}")
            return None

    def parse_gpu_stats(self, gpu):
        """Parse GPU statistics from NVIDIA SMI XML output."""
        try:
            stats = {}
            stats['utilization'] = {
                'gpu_util': self.safe_get(gpu, 'utilization/gpu_util', "0 %"),
                'memory_util': self.safe_get(gpu, 'utilization/memory_util', "0 %")
            }
            stats['temperature'] = {
                'gpu_temp': {
                    'value': self.safe_get(gpu, 'temperature/gpu_temp', "0 C")
                }
            }
            stats['clocks'] = {
                'graphics_clock': {
                    'current_mhz': self.safe_get(gpu, 'clocks/graphics_clock', "0 MHz")
                },
                'memory_clock': {
                    'current_mhz': self.safe_get(gpu, 'clocks/mem_clock', "0 MHz")
                }
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error parsing GPU stats: {str(e)}")
            return {}

    def safe_get(self, element, path, default="Unknown"):
        """Safely extract text from an XML element using a path."""
        try:
            for part in path.split('/'):
                element = element.find(part)
                if element is None:
                    return default
            return element.text.strip() if element is not None and element.text else default
        except Exception:
            return default

    def get_cpu_info(self):
        """Get detailed CPU information and performance metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1, percpu=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'cpu_stats': psutil.cpu_stats()._asdict(),
                'load_avg': psutil.getloadavg()
            }
        except Exception as e:
            self.logger.error(f"Error getting CPU info: {str(e)}")
            return {}

    def save_metrics(self):
        """Save metrics to JSON file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")

    def safe_float_convert(self, value_str):
        """Safely convert string values to float, handling 'Unknown' and other non-numeric values"""
        try:
            if isinstance(value_str, str):
                numeric_part = value_str.split()[0]
                return float(numeric_part)
            return float(value_str)
        except (ValueError, TypeError, AttributeError):
            return 0.0

    def create_interactive_dashboard(self):
        """Create an interactive Plotly dashboard from collected metrics"""
        try:
            if not self.metrics:
                self.logger.error("No metrics collected to create dashboard")
                return

            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'CPU Usage', 'GPU Usage',
                    'Memory Usage', 'GPU Temperature',
                    'Network I/O', 'Disk I/O'
                ),
                vertical_spacing=0.1
            )

            # Time axis for all plots
            time_axis = [datetime.fromisoformat(m['timestamp']) for m in self.metrics]

            # CPU Usage
            cpu_data = [np.mean(m['cpu']['cpu_percent']) for m in self.metrics]
            fig.add_trace(
                go.Scatter(x=time_axis, y=cpu_data, name="CPU Usage %"),
                row=1, col=1
            )

            # GPU Usage
            if self.metrics[0].get('gpu') and self.metrics[0]['gpu']:
                gpu_util = [self.safe_float_convert(m['gpu'][0]['utilization']['gpu_util'])
                           for m in self.metrics]
                fig.add_trace(
                    go.Scatter(x=time_axis, y=gpu_util, name="GPU Usage %"),
                    row=1, col=2
                )

            # Memory Usage
            memory_data = [m['memory']['percent'] for m in self.metrics]
            fig.add_trace(
                go.Scatter(x=time_axis, y=memory_data, name="Memory Usage %"),
                row=2, col=1
            )

            # GPU Temperature
            if self.metrics[0].get('gpu') and self.metrics[0]['gpu']:
                temp_data = [self.safe_float_convert(m['gpu'][0]['temperature']['gpu_temp']['value'])
                            for m in self.metrics]
                fig.add_trace(
                    go.Scatter(x=time_axis, y=temp_data, name="GPU Temperature °C"),
                    row=2, col=2
                )

            # Network I/O
            net_bytes_sent = [m['network']['bytes_sent'] / 1e6 for m in self.metrics]  # Convert to MB
            net_bytes_recv = [m['network']['bytes_recv'] / 1e6 for m in self.metrics]  # Convert to MB
            fig.add_trace(
                go.Scatter(x=time_axis, y=net_bytes_sent, name="Network MB Sent"),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_axis, y=net_bytes_recv, name="Network MB Received"),
                row=3, col=1
            )

            # Disk I/O
            if self.metrics[0]['disk']:
                disk_name = list(self.metrics[0]['disk'].keys())[0]
                disk_read = [m['disk'][disk_name]['read_bytes'] / 1e6 for m in self.metrics]  # Convert to MB
                disk_write = [m['disk'][disk_name]['write_bytes'] / 1e6 for m in self.metrics]  # Convert to MB
                fig.add_trace(
                    go.Scatter(x=time_axis, y=disk_read, name="Disk MB Read"),
                    row=3, col=2
                )
                fig.add_trace(
                    go.Scatter(x=time_axis, y=disk_write, name="Disk MB Written"),
                    row=3, col=2
                )

            # Update layout
            fig.update_layout(
                height=1200,
                title_text="System Performance Dashboard",
                showlegend=True,
                template="plotly_dark"
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Percentage", row=1, col=1)
            fig.update_yaxes(title_text="Percentage", row=1, col=2)
            fig.update_yaxes(title_text="Percentage", row=2, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
            fig.update_yaxes(title_text="MB", row=3, col=1)
            fig.update_yaxes(title_text="MB", row=3, col=2)

            # Save to HTML file
            fig.write_html("system_dashboard.html")
            self.logger.info("Dashboard created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Enhanced System Monitoring Suite')
    parser.add_argument('--output', type=str, default='system_metrics.json',
                       help='Output JSON file name (default: system_metrics.json)')
    parser.add_argument('--duration', type=int, default=300,
                       help='Monitoring duration in seconds (default: 300)')
    
    args = parser.parse_args()
    
    monitor = EnhancedSystemMonitor(output_file=args.output)
    
    try:
        print(f"Starting system monitoring for {args.duration} seconds...")
        monitor.start_monitoring()
        
        # Run for specified duration
        time.sleep(args.duration)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Create dashboard
        print("Creating interactive dashboard...")
        monitor.create_interactive_dashboard()
        print("Dashboard saved as system_dashboard.html")
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        monitor.logger.error(f"Error details: {str(e)}", exc_info=True)
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
