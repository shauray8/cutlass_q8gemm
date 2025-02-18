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
                time.sleep(1)  # Wait before retrying

    def safe_get(self, element, tag, default="Unknown"):
        """Safely extract text from an XML element, avoiding NoneType errors."""
        if element is None:
            return default
        elem = element.find(tag)
        return elem.text.strip() if elem is not None and elem.text else default

    def get_nvidia_driver_version(self):
        """Retrieve the NVIDIA driver version."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout.strip() if result.returncode == 0 else "Unknown"
        except Exception as e:
            self.logger.error(f"Error getting NVIDIA driver version: {str(e)}")
            return "Unknown"

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
            
            # Basic information
            stats['timestamp'] = datetime.now().isoformat()
            stats['product_name'] = self.safe_get(gpu, 'product_name')
            stats['serial'] = self.safe_get(gpu, 'serial')
            stats['driver_version'] = self.safe_get(gpu, 'driver_version', self.get_nvidia_driver_version())
            stats['vbios_version'] = self.safe_get(gpu, 'vbios_version')
            
            # Temperature stats
            temp = gpu.find('temperature')
            if temp is not None:
                stats['temperature'] = {
                    'gpu_temp': {
                        'value': self.safe_get(temp, 'gpu_temp'),
                        'unit': 'C',
                        'max_threshold': self.safe_get(temp, 'gpu_temp_max_threshold')
                    }
                }
            
            # Utilization
            util_section = gpu.find('utilization')
            if util_section:
                stats['utilization'] = {
                    'gpu_util': self.safe_get(util_section, 'gpu_util'),
                    'memory_util': self.safe_get(util_section, 'memory_util'),
                    'encoder_util': self.safe_get(util_section, 'encoder_util'),
                    'decoder_util': self.safe_get(util_section, 'decoder_util')
                }
            
            # Clock speeds
            stats['clocks'] = {}
            for clock_type in ['graphics', 'sm', 'memory']:
                clock = gpu.find(f'clocks/{clock_type}_clock')
                stats['clocks'][f'{clock_type}_clock'] = {
                    'current_mhz': self.safe_get(clock, '.'),
                    'unit': 'MHz'
                }
            
            # PCIe Information
            pci_info = gpu.find('pci')
            if pci_info:
                stats['pcie'] = {
                    'link_gen': self.safe_get(pci_info, 'pci_gpu_link_info/link_gen'),
                    'current_link_width': self.safe_get(pci_info, 'pci_gpu_link_info/link_widths/current_link_width'),
                    'tx_util': self.safe_get(pci_info, 'tx_util'),
                    'rx_util': self.safe_get(pci_info, 'rx_util')
                }
            
            return stats
        except Exception as e:
            self.logger.error(f"Error parsing GPU stats: {str(e)}")
            return {}

    def get_cpu_info(self):
        """Get detailed CPU information and performance metrics"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1, percpu=True),
                'cpu_freq': psutil.cpu_freq()._asdict(),
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

    # [Rest of the methods including create_interactive_dashboard remain the same as in the previous version]

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
