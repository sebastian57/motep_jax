#!/usr/bin/env python3
"""
Performance monitoring script for MTP-LAMMPS simulations
Run alongside your simulation to track resource usage
"""

import psutil
import GPUtil
import time
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import threading
import subprocess

class PerformanceMonitor:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        
        self.monitoring = True
        
    def monitor_resources(self):
        """Monitor CPU, memory, and GPU usage"""
        while self.monitoring:
            timestamp = time.time()
            
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # GPU monitoring (ROCm)
            gpu_stats = self._get_rocm_stats()
            
            self.timestamps.append(timestamp)
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_percent)
            self.gpu_usage.append(gpu_stats['usage'])
            self.gpu_memory.append(gpu_stats['memory'])
            
            if self.rank == 0:
                print(f"[Monitor] CPU: {cpu_percent:.1f}%, RAM: {memory_percent:.1f}%, "
                      f"GPU: {gpu_stats['usage']:.1f}%, GPU_MEM: {gpu_stats['memory']:.1f}%")
            
            time.sleep(self.log_interval)
    
    def _get_rocm_stats(self):
        """Get ROCm GPU statistics"""
        try:
            # Use rocm-smi to get GPU stats
            result = subprocess.run(['rocm-smi', '--showuse', '--showmemuse'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                # Parse the output (this is simplified - adjust based on your rocm-smi output)
                gpu_usage = 0.0
                gpu_memory = 0.0
                
                for line in lines:
                    if 'GPU use' in line:
                        gpu_usage = float(line.split()[-1].replace('%', ''))
                    elif 'GPU memory use' in line:
                        gpu_memory = float(line.split()[-1].replace('%', ''))
                
                return {'usage': gpu_usage, 'memory': gpu_memory}
            else:
                return {'usage': 0.0, 'memory': 0.0}
                
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            return {'usage': 0.0, 'memory': 0.0}
    
    def start_monitoring(self):
        """Start monitoring in a separate thread"""
        self.monitor_thread = threading.Thread(target=self.monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Rank {self.rank}: Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring and save results"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        
        self._save_results()
        if self.rank == 0:
            self._create_plots()
    
    def _save_results(self):
        """Save monitoring results to file"""
        data = {
            'timestamps': self.timestamps,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'gpu_memory': self.gpu_memory
        }
        
        import pickle
        with open(f'performance_rank_{self.rank}.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Rank {self.rank}: Performance data saved")
    
    def _create_plots(self):
        """Create performance plots (rank 0 only)"""
        if not self.timestamps:
            return
            
        # Collect data from all ranks
        all_data = self.comm.gather({
            'timestamps': self.timestamps,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'gpu_memory': self.gpu_memory
        }, root=0)
        
        if self.rank != 0:
            return
        
        plt.figure(figsize=(15, 10))
        
        # CPU Usage
        plt.subplot(2, 2, 1)
        for i, data in enumerate(all_data):
            if data['timestamps']:
                times = np.array(data['timestamps']) - data['timestamps'][0]
                plt.plot(times, data['cpu_usage'], label=f'Rank {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('CPU Usage (%)')
        plt.title('CPU Usage Over Time')
        plt.legend()
        plt.grid(True)
        
        # Memory Usage
        plt.subplot(2, 2, 2)
        for i, data in enumerate(all_data):
            if data['timestamps']:
                times = np.array(data['timestamps']) - data['timestamps'][0]
                plt.plot(times, data['memory_usage'], label=f'Rank {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory Usage (%)')
        plt.title('Memory Usage Over Time')
        plt.legend()
        plt.grid(True)
        
        # GPU Usage
        plt.subplot(2, 2, 3)
        for i, data in enumerate(all_data):
            if data['timestamps']:
                times = np.array(data['timestamps']) - data['timestamps'][0]
                plt.plot(times, data['gpu_usage'], label=f'Rank {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('GPU Usage (%)')
        plt.title('GPU Usage Over Time')
        plt.legend()
        plt.grid(True)
        
        # GPU Memory
        plt.subplot(2, 2, 4)
        for i, data in enumerate(all_data):
            if data['timestamps']:
                times = np.array(data['timestamps']) - data['timestamps'][0]
                plt.plot(times, data['gpu_memory'], label=f'Rank {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('GPU Memory (%)')
        plt.title('GPU Memory Usage Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Performance plots saved as 'performance_analysis.png'")


# Usage example:
if __name__ == "__main__":
    monitor = PerformanceMonitor(log_interval=5)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Your simulation would run here
    # For testing, just sleep
    time.sleep(60)
    
    # Stop monitoring
    monitor.stop_monitoring()
