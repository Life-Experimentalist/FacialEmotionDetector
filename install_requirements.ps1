# install_requirements.ps1

# Colorful Choices
$freezeColor = "Cyan"
$installColor = "Green"

# Function to display a colorful choice
function Write-ColorfulChoice {
    param (
        [string]$Text,
        [string]$Color
    )
    Write-Host -ForegroundColor $Color $Text
}

# Ask the user what they want to do
Write-Host "Choose an option:"
Write-ColorfulChoice "(1) Freeze requirements to requirements.txt" $freezeColor
Write-ColorfulChoice "(2) Install requirements from a file" $installColor
$choice = Read-Host "Enter your choice (1/2)"

if ($choice -eq "1") {
    # Ask if versions are needed
    $versionChoice = Read-Host "Do you want to include versions in the requirements file? (y/n)"

    if ($versionChoice -eq "y") {
        # Freeze requirements with versions
        pip freeze > requirements.txt
    } else {
        # Freeze requirements without versions
        pip freeze | ForEach-Object {
            if ($_ -match "^([^=]+)==") {
                $packageName = $Matches[1]
                Write-Output $packageName
            }
        } > requirements.txt
    }

    Write-Host "Requirements list frozen to requirements.txt"
    exit 0
} elseif ($choice -eq "2") {
    $requirementsFile = "requirements.txt"

    # Check if requirements.txt exists
    if (-Not (Test-Path $requirementsFile)) {
        # List available .txt files
        Write-Host "requirements.txt not found. Available .txt files in the current directory:"
        Get-ChildItem -Path "." -Filter "*.txt" | ForEach-Object {
            Write-Host $_.Name
        }

        # Ask for a different file
        $requirementsFile = Read-Host "Enter the name of the requirements file (including extension):"
        if (-Not (Test-Path $requirementsFile)) {
            Write-Host "File not found!"
            exit 1
        }
    }

    # Install requirements with skipping
    $failedPackages = @()

    try {
        pip install -r $requirementsFile --upgrade
    }
    catch {
        Write-Warning "An error occurred during initial installation. Attempting to continue with individual packages..."

        foreach ($package in (Get-Content $requirementsFile)) {
            try {
                pip install $package --upgrade
            }
            catch {
                Write-Error "Failed to install package: $package"
                $failedPackages += $package
            }
        }
    }

    if ($failedPackages) {
        Write-Host "`n--- Failed Packages ---" -ForegroundColor Red
        foreach ($failedPackage in $failedPackages) {
            Write-Host $failedPackage -ForegroundColor Red
        }
    }

    Write-Host "Requirements installation completed."
} else {
    Write-Host "Invalid choice."
    exit 1
}

# PowerShell script to install required packages

Write-Host "Installing required Python packages for Facial Emotion Detection..." -ForegroundColor Green

# Create and activate virtual environment (optional)
if (-Not (Test-Path -Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

# Activate virtual environment
if (Test-Path -Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Virtual environment not found. Installing globally..." -ForegroundColor Yellow
}

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "You can now run the following commands:" -ForegroundColor Cyan
Write-Host "  python prepare_data.py  - to download and prepare the dataset" -ForegroundColor White
Write-Host "  python train_model.py   - to train the emotion detection model" -ForegroundColor White
Write-Host "  python test_model.py    - to test the model with webcam" -ForegroundColor White
