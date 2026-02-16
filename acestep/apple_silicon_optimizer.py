"""
Apple Silicon ARM64 Optimization Configuration
Provides device detection and MPS optimization for Apple Silicon Macs
"""

import os
import platform
import subprocess
import sys
from typing import Dict, Optional, Tuple


class AppleSiliconOptimizer:
    """Apple Silicon device and performance optimization utilities."""
    
    def __init__(self):
        self.is_apple_silicon = self._detect_apple_silicon()
        self.mps_available = self._check_mps_availability()
        
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon (M1, M2, M3, M4, etc.)"""
        if platform.system() != "Darwin":
            return False
        
        # Check processor architecture
        machine = platform.machine().lower()
        if machine == "arm64":
            return True
            
        # Additional check for Apple Silicon detection
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            brand_string = result.stdout.strip().lower()
            return "apple" in brand_string
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_mps_availability(self) -> bool:
        """Check if Metal Performance Shaders (MPS) backend is available."""
        if not self.is_apple_silicon:
            return False
            
        try:
            import torch
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except ImportError:
            return False
    
    def get_optimal_device(self, prefer_mps: bool = True) -> str:
        """
        Get the optimal device for PyTorch operations on Apple Silicon.
        
        Args:
            prefer_mps: Whether to prefer MPS over CPU when available
            
        Returns:
            Device string: 'mps', 'cpu', or 'cuda' (fallback)
        """
        if self.is_apple_silicon and prefer_mps and self.mps_available:
            return "mps"
        elif self._check_cuda_availability():
            return "cuda"
        else:
            return "cpu"
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available (unlikely on Apple Silicon but possible via cloud)."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def apply_apple_silicon_optimizations(self) -> None:
        """Apply Apple Silicon specific environment optimizations."""
        if not self.is_apple_silicon:
            return
        
        # Set environment variables for optimal Apple Silicon performance
        optimizations = {
            # PyTorch MPS optimizations
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            
            # Memory management
            "MALLOC_ARENA_MAX": "4",
            
            # Threading optimizations for ARM64
            "OMP_NUM_THREADS": str(min(8, os.cpu_count() or 4)),
            "MKL_NUM_THREADS": str(min(8, os.cpu_count() or 4)),
            "OPENBLAS_NUM_THREADS": str(min(8, os.cpu_count() or 4)),
            "VECLIB_MAXIMUM_THREADS": str(min(8, os.cpu_count() or 4)),
            
            # Disable problematic optimizations on ARM64
            "TOKENIZERS_PARALLELISM": "false",
        }
        
        for key, value in optimizations.items():
            os.environ[key] = value
        
        print("✅ Applied Apple Silicon ARM64 optimizations")
    
    def get_memory_info(self) -> Dict[str, Optional[int]]:
        """Get system memory information."""
        memory_info = {}
        
        try:
            # Get system memory on macOS
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                total_bytes = int(result.stdout.strip())
                memory_info["total_gb"] = total_bytes // (1024**3)
            
            # Get available memory
            import psutil
            memory_info["available_gb"] = psutil.virtual_memory().available // (1024**3)
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, 
                FileNotFoundError, ImportError, ValueError):
            memory_info["total_gb"] = None
            memory_info["available_gb"] = None
        
        return memory_info
    
    def get_recommended_settings(self) -> Dict[str, any]:
        """Get recommended settings for Apple Silicon deployment."""
        memory_info = self.get_memory_info()
        total_gb = memory_info.get("total_gb", 8)
        
        # Recommend offload settings based on memory
        offload_to_cpu = total_gb < 32  # Offload on systems with <32GB RAM
        
        return {
            "device": self.get_optimal_device(),
            "offload_to_cpu": offload_to_cpu,
            "offload_dit_to_cpu": total_gb < 16,  # More aggressive offload for <16GB
            "use_flash_attention": False,  # Limited ARM64 support
            "backend": "pt",  # PyTorch backend preferred over vLLM on Apple Silicon
            "batch_size": min(4, max(1, (total_gb // 8))),  # Dynamic batch size
            "precision": "float16" if self.mps_available else "float32",
        }
    
    def print_system_info(self) -> None:
        """Print detailed system information for Apple Silicon."""
        print("🍎 Apple Silicon ARM64 System Information")
        print("=" * 50)
        print(f"Platform: {platform.platform()}")
        print(f"Architecture: {platform.machine()}")
        print(f"Apple Silicon Detected: {self.is_apple_silicon}")
        print(f"MPS Available: {self.mps_available}")
        print(f"Optimal Device: {self.get_optimal_device()}")
        
        memory_info = self.get_memory_info()
        if memory_info["total_gb"]:
            print(f"Total Memory: {memory_info['total_gb']} GB")
        if memory_info["available_gb"]:
            print(f"Available Memory: {memory_info['available_gb']} GB")
        
        recommendations = self.get_recommended_settings()
        print("\n🚀 Recommended Settings:")
        for key, value in recommendations.items():
            print(f"  {key}: {value}")
        print("=" * 50)


def optimize_for_apple_silicon() -> AppleSiliconOptimizer:
    """
    Initialize and apply Apple Silicon optimizations.
    
    Returns:
        AppleSiliconOptimizer instance with system information
    """
    optimizer = AppleSiliconOptimizer()
    optimizer.apply_apple_silicon_optimizations()
    
    if optimizer.is_apple_silicon:
        optimizer.print_system_info()
    
    return optimizer


# Auto-apply optimizations when module is imported
if __name__ == "__main__":
    # Only apply when run directly, not when imported
    optimizer = optimize_for_apple_silicon()
else:
    # When imported, create optimizer but don't auto-apply (let the user decide)
    __optimizer = AppleSiliconOptimizer()
    
    # Export commonly used functions
    def get_optimal_device() -> str:
        return __optimizer.get_optimal_device()
    
    def is_apple_silicon() -> bool:
        return __optimizer.is_apple_silicon
    
    def get_recommended_settings() -> Dict[str, any]:
        return __optimizer.get_recommended_settings()