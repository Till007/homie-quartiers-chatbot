
# start_dev_stack.ps1

# Projektverzeichnis automatisch ermitteln
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

# Prüfen, ob Python existiert
if (-not (Test-Path $VenvPython)) {
    Write-Host "FEHLER: Python nicht gefunden: $VenvPython" -ForegroundColor Red
    exit 1
}

Write-Host "Python gefunden: $VenvPython" -ForegroundColor Green

# Arbeitsverzeichnis auf Projektroot setzen
Set-Location $ProjectRoot

Write-Host "Prüfe, ob Ollama Server (Port 11434) läuft..." -ForegroundColor Yellow

$OllamaRunning = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue

if (-not $OllamaRunning) {
    Write-Host "Starte lokalen Ollama Server im Hintergrund..." -ForegroundColor Cyan

    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$ProjectRoot'; ollama serve"
    )

    Start-Sleep -Seconds 3
}

Write-Host "Starte HOMIE Microservice-Flotte..." -ForegroundColor Cyan

$Services = @(
    @{ Name = "Quality Gate"; Module = "services.quality_gate_service.main:app"; Port = 8001 },
    @{ Name = "ETL Ingest"; Module = "services.etl_ingest_service.main:app"; Port = 8002 },
    @{ Name = "RAG Retrieval"; Module = "services.rag_retrieval_service.main:app"; Port = 8003 },
    @{ Name = "Orchestrator"; Module = "services.orchestrator_service.main:app"; Port = 8004 },
    @{ Name = "Speech"; Module = "services.speech_service.main:app"; Port = 8005 }
)

foreach ($Service in $Services) {

    Write-Host "Starte $($Service.Name) auf Port $($Service.Port)..." -ForegroundColor Cyan

    $Command = "Set-Location '$ProjectRoot'; & '$VenvPython' -m uvicorn $($Service.Module) --host 127.0.0.1 --port $($Service.Port) --reload"

    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        $Command
    )
}

Start-Sleep -Seconds 4

Write-Host "Starte Gradio Debug- & Kiosk-Oberfläche..." -ForegroundColor Green

& $VenvPython -m services.ui_debug_client.app
