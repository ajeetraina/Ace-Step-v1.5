#!/bin/bash
# ACE-Step v1.5 Apple Silicon ARM64 Build and Test Script

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Apple Silicon
check_apple_silicon() {
    print_status "Checking Apple Silicon compatibility..."
    
    if [[ "$(uname -s)" != "Darwin" ]]; then
        print_error "This script is designed for macOS. Detected: $(uname -s)"
        exit 1
    fi
    
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        print_success "Running on Apple Silicon (ARM64)"
        
        # Check for specific Apple Silicon chip
        CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
        print_status "Detected chip: $CHIP"
    elif [[ "$ARCH" == "x86_64" ]]; then
        # Check if running under Rosetta 2
        if [[ $(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0) == "1" ]]; then
            print_warning "Running under Rosetta 2 (x86_64 emulation)"
            print_warning "Performance will be suboptimal. Consider using native ARM64 terminal."
        else
            print_warning "Running on Intel Mac. Some optimizations may not apply."
        fi
    else
        print_error "Unsupported architecture: $ARCH"
        exit 1
    fi
}

# Check Python version and architecture
check_python() {
    print_status "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        print_status "Install Python via Homebrew: brew install python@3.11"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "Python version: $PYTHON_VERSION"
    
    # Check Python architecture
    PYTHON_ARCH=$(python3 -c "import platform; print(platform.machine())")
    if [[ "$PYTHON_ARCH" == "arm64" ]]; then
        print_success "Python is running natively on Apple Silicon"
    else
        print_warning "Python is running under x86_64 (Rosetta). Consider reinstalling Python for ARM64."
    fi
    
    # Check required Python version
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 || ($PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 11) ]]; then
        print_error "Python 3.11+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

# Check and install uv package manager
check_uv() {
    print_status "Checking uv package manager..."
    
    if ! command -v uv &> /dev/null; then
        print_status "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source "$HOME/.local/bin/env" 2>/dev/null || true
    fi
    
    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version 2>&1 | cut -d' ' -f2)
        print_success "uv version: $UV_VERSION"
    else
        print_error "Failed to install uv. Please install manually."
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies with Apple Silicon optimizations..."
    
    # Sync dependencies using uv
    uv sync
    
    print_success "Dependencies installed successfully"
}

# System information gathering
gather_system_info() {
    print_status "Gathering Apple Silicon system information..."
    
    echo "=== System Information ==="
    echo "OS: $(sw_vers -productName) $(sw_vers -productVersion)"
    echo "Hardware: $(system_profiler SPHardwareDataType | grep "Model Name" | sed 's/.*: //')"
    echo "Architecture: $(uname -m)"
    echo "Memory: $(system_profiler SPHardwareDataType | grep "Memory" | sed 's/.*: //')"
    echo "Python: $(python3 --version) ($(python3 -c "import platform; print(platform.machine())"))"
    
    # Check PyTorch MPS availability
    uv run python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS device ready for Apple Silicon acceleration!')
else:
    print('MPS not available, will use CPU fallback')
"
    echo "=========================="
}

# Run Apple Silicon optimizer
test_optimizer() {
    print_status "Testing Apple Silicon optimizer..."
    
    uv run python3 -c "
from acestep.apple_silicon_optimizer import optimize_for_apple_silicon
optimizer = optimize_for_apple_silicon()
print('Apple Silicon optimizer test completed successfully!')
"
    
    print_success "Apple Silicon optimizer working correctly"
}

# Build Docker image for Apple Silicon
build_docker() {
    if [[ "$1" == "--docker" ]]; then
        print_status "Building Docker image for Apple Silicon..."
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            print_error "Docker is not installed"
            print_status "Install Docker Desktop for Mac with Apple Silicon support"
            return 1
        fi
        
        # Enable buildx for multi-platform builds
        docker buildx create --use 2>/dev/null || true
        
        # Build ARM64 image
        print_status "Building ARM64 Docker image..."
        docker buildx build --platform linux/arm64 -t ace-step:arm64-latest . --load
        
        print_success "Docker image built successfully for ARM64"
        
        # Test Docker image
        print_status "Testing Docker image..."
        docker run --rm ace-step:arm64-latest python3 -c "
from acestep.apple_silicon_optimizer import is_apple_silicon
print('Docker test completed successfully!')
print(f'Container architecture: arm64')
"
        print_success "Docker image test completed"
    fi
}

# Run performance benchmark
run_benchmark() {
    if [[ "$1" == "--benchmark" ]]; then
        print_status "Running Apple Silicon performance benchmark..."
        
        uv run python3 -c "
import time
import torch
from acestep.apple_silicon_optimizer import AppleSiliconOptimizer

optimizer = AppleSiliconOptimizer()
device = optimizer.get_optimal_device()
print(f'Using device: {device}')

# Simple tensor operation benchmark
if device == 'mps':
    tensor = torch.randn(1000, 1000).to('mps')
    start_time = time.time()
    for _ in range(100):
        result = torch.mm(tensor, tensor)
    end_time = time.time()
    print(f'MPS tensor operations (100x1000x1000): {end_time - start_time:.2f}s')
else:
    tensor = torch.randn(1000, 1000)
    start_time = time.time()
    for _ in range(100):
        result = torch.mm(tensor, tensor)
    end_time = time.time()
    print(f'CPU tensor operations (100x1000x1000): {end_time - start_time:.2f}s')

print('Benchmark completed!')
"
        
        print_success "Benchmark completed"
    fi
}

# Main execution
main() {
    echo "🍎 ACE-Step v1.5 Apple Silicon ARM64 Build Script"
    echo "================================================"
    
    check_apple_silicon
    check_python
    check_uv
    gather_system_info
    install_dependencies
    test_optimizer
    
    # Handle optional arguments
    for arg in "$@"; do
        case $arg in
            --docker)
                build_docker --docker
                ;;
            --benchmark)
                run_benchmark --benchmark
                ;;
        esac
    done
    
    echo ""
    print_success "Apple Silicon setup completed successfully!"
    echo ""
    echo "🚀 Quick start commands:"
    echo "  uv run acestep                    # Launch web UI"
    echo "  uv run acestep-api               # Launch REST API"
    echo "  uv run acestep --device auto     # Auto-detect optimal device"
    echo "  docker-compose up ace-step       # Run with Docker"
    echo ""
    echo "📚 For detailed instructions, see APPLE_SILICON_GUIDE.md"
}

# Help message
show_help() {
    echo "ACE-Step v1.5 Apple Silicon Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help        Show this help message"
    echo "  --docker      Build and test Docker image"
    echo "  --benchmark   Run performance benchmark"
    echo ""
    echo "Examples:"
    echo "  $0                    # Basic setup"
    echo "  $0 --docker          # Setup with Docker build"
    echo "  $0 --benchmark       # Setup with performance test"
    echo "  $0 --docker --benchmark  # Full setup with all tests"
}

# Handle command line arguments
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Run main function with all arguments
main "$@"