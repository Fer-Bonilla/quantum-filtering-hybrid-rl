# tasks.ps1 - Wrapper PowerShell para comandos comunes del proyecto.
# Uso: .\tasks.ps1 <comando> [args...]

param(
    [Parameter(Position=0)]
    [string]$Command = "help",

    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host "Comandos disponibles:" -ForegroundColor Cyan
    Write-Host "  test              Ejecutar suite completa de tests"
    Write-Host "  test-unit         Solo tests unitarios"
    Write-Host "  test-data         Tests de la capa de datos"
    Write-Host "  test-quantum      Tests del modulo cuantico"
    Write-Host "  lint              Ejecutar ruff check"
    Write-Host "  format            Ejecutar ruff format"
    Write-Host "  typecheck         Ejecutar mypy"
    Write-Host "  download-data     Descargar datos OHLCV (Nivel 1)"
    Write-Host "  train-a           Entrenar Modelo A (baseline PPO)"
    Write-Host "  train-b           Entrenar Modelo B (PPO + rasgos relacionales)"
    Write-Host "  train-c           Entrenar Modelo C (PPO + caminata clasica)"
    Write-Host "  train-d           Entrenar Modelo D (PPO + DTQW)"
    Write-Host "  campaign          Ejecutar campania experimental A/B/C/D x N semillas"
    Write-Host "  ablation          Ejecutar ablaciones (init, noise, M, k, m)"
    Write-Host "  aggregate         Generar tabla comparativa y bootstrap pareado C vs D"
    Write-Host "  figures           Generar figuras PNG para la tesis"
    Write-Host "  clean             Limpiar caches y outputs intermedios"
}

switch ($Command) {
    "help"      { Show-Help }
    "test"      { uv run pytest tests\ $Args }
    "test-unit" { uv run pytest tests\unit\ $Args }
    "test-data" { uv run pytest tests\unit\test_data_* tests\unit\test_features_* $Args }
    "test-quantum" { uv run pytest tests\unit\ -m quantum $Args }
    "lint"      { uv run ruff check src tests scripts }
    "format"    { uv run ruff format src tests scripts }
    "typecheck" { uv run mypy src }
    "download-data" {
        $seed = if ($Args.Count -gt 0) { $Args[0] } else { "42" }
        uv run python -m src.data.download --config configs\data\nivel1.yaml
    }
    "train-a"   { uv run python -m src.main train --config configs\experiment\model_a.yaml $Args }
    "train-b"   { uv run python -m src.main train --config configs\experiment\model_b.yaml $Args }
    "train-c"   { uv run python -m src.main train --config configs\experiment\model_c.yaml $Args }
    "train-d"   { uv run python -m src.main train --config configs\experiment\model_d.yaml $Args }
    "campaign"  { uv run python scripts\run_campaign.py $Args }
    "ablation"  { uv run python scripts\run_ablation.py $Args }
    "aggregate" { uv run python scripts\aggregate_results.py $Args }
    "figures"   { uv run python scripts\generate_figures.py $Args }
    "clean" {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
            .pytest_cache, .mypy_cache, .ruff_cache, .coverage, htmlcov, `
            data\interim, data\processed
        Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force
        Write-Host "Limpieza completa." -ForegroundColor Green
    }
    default     {
        Write-Host "Comando desconocido: $Command" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
