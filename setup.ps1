# Installs correct PyTorch version depending on system CUDA capability
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $nvidiaOutput = & nvidia-smi 2>$null
    $cudaVersion = $null
    if ($nvidiaOutput) {
        $m = [regex]::Match($nvidiaOutput, 'CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)')
        if ($m.Success) {
            $cudaVersion = $m.Groups[1].Value.Trim()
        }
    }
    Write-Host "Detected NVIDIA GPU with CUDA version $cudaVersion"
        try {
            $ver = [version]$cudaVersion
        } catch {
            Write-Host "Unable to parse CUDA version string '$cudaVersion'; defaulting to CPU-only install"
            $ver = [version]"0.0"
        }

        if ($ver -ge [version]"13.0") {
            Write-Host "Installing PyTorch with CUDA 13.0 support"
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
        } elseif ($ver -ge [version]"12.8") {
            Write-Host "Installing PyTorch with CUDA 12.8 support"
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        } elseif ($ver -ge [version]"12.6") {
            Write-Host "Installing PyTorch with CUDA 12.6 support"
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
        } else {
            Write-Host "Installing CPU-only PyTorch version"
            uv pip install torch torchvision
        }
} else {
    Write-Host "No NVIDIA GPU detected. Installing CPU-only PyTorch version"
    uv pip install torch torchvision
}
