# RAG Tool Installer Script

param(
    [string]$PythonVersion = "3.12.0",
    [string]$InstallPath   = ""
)

$global:installStep = 0
$global:totalSteps  = 4
$global:startTime   = Get-Date

# ---------------------------------------------------------------------------
# Setup Configuration
# ---------------------------------------------------------------------------

function Get-InstallPath {
    if ($InstallPath -and (Test-Path $InstallPath)) {
        return $InstallPath
    }

    # Default to script directory itself (where setup.bat is running from)
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    $defaultPath = $scriptDir
    
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Yellow
    Write-Host "  RAG Tool Installation Path Setup" -ForegroundColor Yellow
    Write-Host ("=" * 60) -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Default path: $defaultPath" -ForegroundColor Cyan
    Write-Host "  (Current directory: $scriptDir)" -ForegroundColor DarkGray
    Write-Host ""
    
    $response = Read-Host "  Use default path? (Y/n) or enter custom path"
    
    if ($response -eq "" -or $response.ToUpper() -eq "Y") {
        $selectedPath = $defaultPath
    } elseif ($response -eq "n" -or $response.ToUpper() -eq "N") {
        $selectedPath = Read-Host "  Enter custom install path"
    } else {
        $selectedPath = $response
    }

    # Create directory if it doesn't exist
    if (-not (Test-Path $selectedPath)) {
        try {
            New-Item -ItemType Directory -Path $selectedPath -Force | Out-Null
            Write-Host "  Created directory: $selectedPath" -ForegroundColor Green
        } catch {
            Write-Host "  Error creating directory: $_" -ForegroundColor Red
            Write-Host "  Using default path instead..." -ForegroundColor Yellow
            $selectedPath = $defaultPath
            New-Item -ItemType Directory -Path $selectedPath -Force | Out-Null
        }
    }

    Write-Host ""
    Write-Host "  Install path: $selectedPath" -ForegroundColor Green
    Write-Host ""
    
    # Save config
    $configPath = Join-Path $selectedPath "install_config.txt"
    @{
        "InstallPath" = $selectedPath
        "InstallDate" = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        "PythonVersion" = $PythonVersion
    } | ConvertTo-Json | Set-Content $configPath

    return $selectedPath
}

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

function Write-Banner {
    param([string]$Message, [string]$Color = "Yellow")
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor $Color
    Write-Host "  $Message" -ForegroundColor $Color
    Write-Host ("=" * 60) -ForegroundColor $Color
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    $global:installStep++
    $pct = [int](($global:installStep / $global:totalSteps) * 100)
    Write-Host ""
    Write-Host "[Step $global:installStep/$global:totalSteps] $Message" -ForegroundColor White
    Write-Progress -Activity "RAG Tool Installation" `
        -Status "Step $global:installStep of $global:totalSteps : $Message" `
        -PercentComplete $pct
}

function Write-Success {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "  [ERROR] $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "  [*] $Message" -ForegroundColor Cyan
}

function Get-Elapsed {
    $e = (Get-Date) - $global:startTime
    return "{0:mm}m {0:ss}s" -f $e
}

# ---------------------------------------------------------------------------
# Download with progress
# ---------------------------------------------------------------------------

function Download-FileWithProgress {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$DisplayName,
        [int]$ItemIndex = 1,
        [int]$TotalItems = 1
    )

    Write-Info "[$ItemIndex/$TotalItems] Downloading $DisplayName..."

    # BITS Transfer shows a native Windows progress bar
    $bitsOk = $false
    try {
        Import-Module BitsTransfer -ErrorAction Stop
        $bitsOk = $true
    } catch {}

    if ($bitsOk) {
        try {
            Write-Progress -Activity "RAG Tool Setup" `
                -CurrentOperation "[DOWNLOAD] $DisplayName" `
                -Status "Initializing download..." `
                -PercentComplete 0
            
            Start-BitsTransfer `
                -Source      $Url `
                -Destination $Destination `
                -DisplayName "Downloading $DisplayName" `
                -Description "RAG Tool Setup"
            
            $mb = [math]::Round((Get-Item $Destination).Length / 1MB, 2)
            Write-Success "$DisplayName downloaded ($mb MB)"
            return
        } catch {
            Write-Info "BITS unavailable - switching to fallback download..."
        }
    }

    # Fallback: WebClient with size callback
    try {
        Write-Progress -Activity "RAG Tool Setup" `
            -CurrentOperation "[DOWNLOAD] $DisplayName" `
            -Status "Please wait..." `
            -PercentComplete 10
        
        Write-Host "  Downloading... (this may take a few minutes)" -ForegroundColor DarkGray
        
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($Url, $Destination)
        
        $mb = [math]::Round((Get-Item $Destination).Length / 1MB, 2)
        Write-Success "$DisplayName downloaded ($mb MB)"
    } catch {
        Write-ErrorMsg "Download failed: $_"
        throw
    }
}

# ---------------------------------------------------------------------------
# Python check / install
# ---------------------------------------------------------------------------

function Test-PythonInstalled {
    Write-Info "Checking for Python..."
    try {
        $ver = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Found: $ver"
            return $true
        }
    } catch {}
    return $false
}

function Install-Python {
    Write-Info "Python not found - downloading Python $PythonVersion..."

    $url       = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
    $installer = "$env:TEMP\python-$PythonVersion-installer.exe"

    try {
        Download-FileWithProgress -Url $url -Destination $installer -DisplayName "Python $PythonVersion"

        Write-Info "Running Python installer (approximately 1 to 3 minutes)..."
        Write-Progress -Activity "Installing Python $PythonVersion" `
            -Status "Please wait - do not close this window..." `
            -PercentComplete 0

        $proc = Start-Process -FilePath $installer `
            -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" `
            -PassThru -Wait

        Write-Progress -Activity "Installing Python $PythonVersion" -Completed

        if ($proc.ExitCode -ne 0) {
            Write-ErrorMsg "Installer exited with code $($proc.ExitCode)"
            return $false
        }

        Write-Success "Python installed"

        # Refresh PATH so python is immediately accessible
        $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        $userPath    = [System.Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path    = $machinePath + ";" + $userPath

        return (Test-PythonInstalled)
    } catch {
        Write-ErrorMsg "Python installation failed: $_"
        return $false
    }
}

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------

function Create-VirtualEnv {
    param([string]$VenvPath)

    Write-Info "Creating virtual environment at: $VenvPath"
    try {
        python -m venv $VenvPath 2>&1 | Out-Null
        Write-Success "Virtual environment created"
        return $true
    } catch {
        Write-ErrorMsg "Failed to create venv: $_"
        return $false
    }
}

# ---------------------------------------------------------------------------
# Requirements install
# ---------------------------------------------------------------------------

function Install-Requirements {
    param(
        [string]$VenvPath,
        [string]$RequirementsPath,
        [string]$Label = "packages"
    )

    $pythonExe = Join-Path $VenvPath "Scripts\python.exe"

    Write-Info "Upgrading pip..."
    Write-Progress -Activity "RAG Tool Setup" `
        -CurrentOperation "Upgrading pip" `
        -PercentComplete 25

    & "$pythonExe" -m pip install --upgrade pip --quiet 2>&1 | Out-Null
    Write-Success "pip upgraded"

    Write-Info "Installing $Label (streaming output below)..."
    Write-Host ""
    Write-Progress -Activity "RAG Tool Setup" `
        -CurrentOperation "[INSTALL] $Label" `
        -Status "Processing..." `
        -PercentComplete 30

    $captured = [System.Collections.Generic.List[string]]::new()
    $packageCount = 0

    & "$pythonExe" -m pip install -r "$RequirementsPath" 2>&1 | ForEach-Object {
        $line    = $_.ToString()
        $trimmed = $line.Trim()
        $captured.Add($line)

        if ($trimmed -eq "") { return }

        if ($trimmed -match "^Collecting\s+(\S+)") { 
            $packageCount++
            Write-Host "  Collecting $($Matches[1])..." -ForegroundColor Cyan 
            Write-Progress -Activity "RAG Tool Setup" `
                -CurrentOperation "[INSTALL] $Label" `
                -Status "Collecting: $($Matches[1])" `
                -PercentComplete 30
        }
        elseif ($trimmed -match "^Downloading\s+(\S+)") { 
            Write-Host "    Downloading $($Matches[1])" -ForegroundColor DarkCyan 
        }
        elseif ($trimmed -match "^Installing collected") { 
            Write-Host ""
            Write-Host "  Installing $packageCount packages..." -ForegroundColor Yellow 
            Write-Progress -Activity "RAG Tool Setup" `
                -CurrentOperation "[INSTALL] $Label" `
                -Status "Installing $packageCount packages" `
                -PercentComplete 50
        }
        elseif ($trimmed -match "^Successfully installed") { 
            Write-Host "  [OK] $trimmed" -ForegroundColor Green 
            Write-Progress -Activity "RAG Tool Setup" `
                -CurrentOperation "[INSTALL] $Label" `
                -Status "$trimmed" `
                -PercentComplete 75
        }
        elseif ($trimmed -match "^(ERROR|error)") { 
            Write-Host "  [ERROR] $trimmed" -ForegroundColor Red 
        }
        elseif ($trimmed -match "^(WARNING|warning)") { 
            Write-Host "  [WARN] $trimmed" -ForegroundColor DarkYellow 
        }
        else { 
            Write-Host "    $trimmed" -ForegroundColor Gray 
        }
    }

    $exitCode  = $LASTEXITCODE
    $outputStr = $captured -join "`n"
    Write-Host ""

    if ($exitCode -eq 0) {
        Write-Success "All $Label installed successfully ($packageCount packages)"
        return $true
    }

    # Handle missing local wheel files
    if ($outputStr -match "No such file or directory") {
        Write-Host ""
        Write-Host ("=" * 52) -ForegroundColor Yellow
        Write-Info "Local wheel files not found in requirements"
        Write-Host ("=" * 52) -ForegroundColor Yellow
        Write-Host ""

        $useLocal = Read-Host "  Do you have local torch wheels? (yes/no)"

        if ($useLocal -eq "yes") {
            $wheelsPath = Read-Host "  Enter the path to your wheels directory"

            if (Test-Path $wheelsPath) {
                Write-Info "Installing from local wheels: $wheelsPath"
                Write-Host ""

                & "$pythonExe" -m pip install -r "$RequirementsPath" --find-links "$wheelsPath" 2>&1 | ForEach-Object {
                    $t = $_.ToString().Trim()
                    if ($t -eq "") { return }
                    if ($t -match "^Collecting") { Write-Host "  $t" -ForegroundColor Cyan }
                    elseif ($t -match "^Installing") { Write-Host "  $t" -ForegroundColor Yellow }
                    elseif ($t -match "^Successfully") { Write-Host "  [OK] $t" -ForegroundColor Green }
                    elseif ($t -match "^(ERROR|error)") { Write-Host "  [ERROR] $t" -ForegroundColor Red }
                    else { Write-Host "  $t" -ForegroundColor Gray }
                }

                if ($LASTEXITCODE -eq 0) {
                    Write-Success "$Label installed successfully"
                    return $true
                }
            } else {
                Write-ErrorMsg "Path not found: $wheelsPath"
            }
        } else {
            Write-Info "Falling back to PyPI (removing local file references)..."

            $tempReq  = "$env:TEMP\requirements_pypi.txt"
            $filtered = (Get-Content $RequirementsPath) |
                Where-Object { $_ -notmatch 'file://|@\s*file' -and $_.Trim() -ne "" }
            $filtered += ""
            $filtered += "# PyPI torch packages"
            $filtered += "torch>=2.0.0"
            $filtered += "torchaudio>=2.0.0"
            $filtered += "torchvision>=0.15.0"
            $filtered | Set-Content $tempReq

            Write-Host ""

            & "$pythonExe" -m pip install -r "$tempReq" 2>&1 | ForEach-Object {
                $t = $_.ToString().Trim()
                if ($t -eq "") { return }
                if ($t -match "^Collecting") { Write-Host "  $t" -ForegroundColor Cyan }
                elseif ($t -match "^Installing") { Write-Host "  $t" -ForegroundColor Yellow }
                elseif ($t -match "^Successfully") { Write-Host "  [OK] $t" -ForegroundColor Green }
                elseif ($t -match "^(ERROR|error)") { Write-Host "  [ERROR] $t" -ForegroundColor Red }
                else { Write-Host "  $t" -ForegroundColor Gray }
            }

            if ($LASTEXITCODE -eq 0) {
                Write-Success "$Label installed successfully"
                Remove-Item $tempReq -Force -ErrorAction SilentlyContinue
                return $true
            }
        }
    }

    Write-ErrorMsg "$Label installation failed"
    return $false
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function Main {
    Write-Banner "RAG Tool Installer"

    # Get installation path from user
    $installPath = Get-InstallPath

    # Step 1 - Python
    Write-Step "Checking Python"
    if (-not (Test-PythonInstalled)) {
        if (-not (Install-Python)) {
            Write-ErrorMsg "Setup failed at Python installation"
            exit 1
        }
    }

    $scriptDir      = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    $venvPath       = Join-Path $installPath "RagTool"
    $reqFile        = Join-Path $scriptDir "requirements.txt"
    $desktopReqFile = Join-Path $scriptDir "desktop\requirements.txt"

    # Step 2 - Virtual environment
    Write-Step "Creating virtual environment"
    if (-not (Create-VirtualEnv $venvPath)) {
        Write-ErrorMsg "Setup failed at venv creation"
        exit 1
    }

    # Step 3 - Install packages
    Write-Step "Installing Python packages"
    if (Test-Path $reqFile) {
        if (-not (Install-Requirements $venvPath $reqFile "main packages")) {
            Write-ErrorMsg "Setup failed at package installation"
            exit 1
        }
    } else {
        Write-Info "requirements.txt not found - skipping"
    }

    if (Test-Path $desktopReqFile) {
        Write-Info "Installing desktop requirements..."
        if (-not (Install-Requirements $venvPath $desktopReqFile "desktop packages")) {
            Write-ErrorMsg "Setup failed at desktop package installation"
            exit 1
        }
    }

    # Step 4 - Done
    Write-Step "Finalizing"
    Write-Progress -Activity "RAG Tool Installation" -Completed

    $elapsed = Get-Elapsed
    Write-Banner "Installation Complete! (took $elapsed)" "Green"

    Write-Info "Installation path:"
    Write-Host "  $installPath" -ForegroundColor Yellow
    Write-Host ""
    Write-Info "To activate the virtual environment:"
    Write-Host "  $venvPath\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Info "To launch RAG Tool:"
    Write-Host "  .\RAGTOOL.exe" -ForegroundColor Yellow
    Write-Host ""

    Read-Host "Press Enter to exit"
}

Main
