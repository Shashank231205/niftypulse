<#
.SYNOPSIS
  Start, stop and inspect the local NiftyPulse stack (infrastructure and services).

.EXAMPLE
  ./ops/devstack.ps1 up infra
  ./ops/devstack.ps1 up recorder archiver
  ./ops/devstack.ps1 status
  ./ops/devstack.ps1 down all
  ./ops/devstack.ps1 exec nats      # run one component in the foreground

  `up` starts each component as a hidden background process running `exec`, which
  re-reads .env itself, so no secret is ever written to a launcher file. Every path,
  address and secret comes from .env.
#>
param(
    [Parameter(Mandatory)][ValidateSet('up', 'down', 'status', 'exec')][string]$Command,
    [Parameter(ValueFromRemainingArguments)][string[]]$Targets = @('all')
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

function Import-DotEnv([string]$Path) {
    if (-not (Test-Path $Path)) { throw ".env not found at $Path (copy .env.example)" }
    foreach ($line in [IO.File]::ReadAllLines($Path)) {
        $line = $line.Trim([char]0xFEFF).Trim()
        if (-not $line -or $line.StartsWith('#') -or -not $line.Contains('=')) { continue }
        $name, $value = $line -split '=', 2
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
    }
}

function Need([string]$Name) {
    $v = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($v)) { throw "required setting $Name is missing in .env" }
    return $v
}

Import-DotEnv (Join-Path $Root '.env')
$Data = Need 'STACK_DATA_DIR'
$RunDir = Join-Path $Data 'run'
$LogDir = Join-Path $Data 'logs'
New-Item -ItemType Directory -Force $RunDir, $LogDir | Out-Null

function Expand-Template([string]$Source, [string]$Target) {
    $text = [IO.File]::ReadAllText($Source)
    $text = [regex]::Replace($text, '\$\{([A-Z0-9_]+)\}', { param($m) Need $m.Groups[1].Value })
    [IO.File]::WriteAllText($Target, $text)
}

function Write-PrometheusTargets([string]$Path) {
    $targets = foreach ($var in (Get-ChildItem Env: | Where-Object Name -match '^([A-Z]+)_METRICS_PORT$')) {
        @{ targets = @("$(Need 'METRICS_HOST'):$($var.Value)"); labels = @{ service = $Matches[1].ToLower() } }
    }
    ConvertTo-Json -InputObject @($targets) -Depth 4 | Set-Content -Encoding utf8 $Path
}

# --- component definitions ---------------------------------------------------
# Each returns @{ Exe; Args; Env } and may prepare files first.

$Services = @('recorder', 'archiver', 'featurizer', 'predictor', 'strategy', 'monitor', 'gateway')
$InfraOrder = @('nats', 'postgres', 'jaeger', 'prometheus', 'grafana')

function Get-Spec([string]$Name, [bool]$Prepare) {
    switch ($Name) {
        'nats' {
            # NATS config strings treat backslashes as escapes.
            return @{ Exe = Need 'NATS_SERVER_BIN'; Args = @('-c', (Join-Path $Root 'ops/nats.conf'))
                Env = @{ NATS_STORE_DIR = (Join-Path $Data 'nats') -replace '\\', '/' } }
        }
        'postgres' {
            $bin = Need 'POSTGRES_BIN_DIR'
            $pgdata = Join-Path $Data 'postgres'
            if ($Prepare -and -not (Test-Path (Join-Path $pgdata 'PG_VERSION'))) {
                $pw = New-TemporaryFile
                try {
                    [IO.File]::WriteAllText($pw, (Need 'POSTGRES_SUPERUSER_PASSWORD'))
                    & (Join-Path $bin 'initdb.exe') -D $pgdata -U (Need 'POSTGRES_SUPERUSER') `
                        --pwfile=$pw --auth=scram-sha-256 -E UTF8 --locale=C | Out-Null
                } finally { Remove-Item $pw }
            }
            return @{ Exe = Join-Path $bin 'postgres.exe'
                Args = @('-D', $pgdata, '-p', (Need 'DB_PORT'), '-c', "listen_addresses=$(Need 'DB_HOST')"); Env = @{} }
        }
        'jaeger' {
            return @{ Exe = Need 'JAEGER_BIN'; Args = @('--config', (Join-Path $Root 'ops/jaeger.yaml')); Env = @{} }
        }
        'prometheus' {
            $cfg = Join-Path $Data 'prometheus.yml'
            $targets = (Join-Path $Data 'prometheus-targets.json') -replace '\\', '/'
            if ($Prepare) {
                $env:PROMETHEUS_TARGETS_FILE = $targets
                Write-PrometheusTargets $targets
                Expand-Template (Join-Path $Root 'ops/prometheus.yml.tmpl') $cfg
            }
            return @{ Exe = Need 'PROMETHEUS_BIN'; Env = @{}
                Args = @("--config.file=$cfg", "--storage.tsdb.path=$(Join-Path $Data 'prometheus')",
                    "--web.listen-address=$(Need 'PROMETHEUS_LISTEN')") }
        }
        'grafana' {
            $home_ = Need 'GRAFANA_HOME'
            New-Item -ItemType Directory -Force (Join-Path $Data 'grafana') | Out-Null
            return @{ Exe = Join-Path (Join-Path $home_ 'bin') 'grafana.exe'; Args = @('server', '--homepath', $home_)
                Env = @{
                    GF_SERVER_HTTP_ADDR            = Need 'GRAFANA_HTTP_ADDR'
                    GF_SERVER_HTTP_PORT            = Need 'GRAFANA_HTTP_PORT'
                    GF_SECURITY_ADMIN_PASSWORD     = Need 'GRAFANA_ADMIN_PASSWORD'
                    GF_PATHS_DATA                  = Join-Path $Data 'grafana'
                    GF_PATHS_LOGS                  = $LogDir
                    GF_PATHS_PROVISIONING          = Join-Path $Root 'ops/grafana/provisioning'
                    GF_ANALYTICS_REPORTING_ENABLED = 'false'
                    GF_ANALYTICS_CHECK_FOR_UPDATES = 'false'
                    GF_AUTH_ANONYMOUS_ENABLED      = 'false'
                } }
        }
        default {
            if ($Services -notcontains $Name) { throw "unknown target $Name" }
            $upper = $Name.ToUpper()
            return @{ Exe = Need 'UV_BIN'; Args = @('run', '--no-sync', 'niftypulse', 'run', $Name)
                Env = @{
                    NATS_USER         = $Name
                    NATS_PASSWORD     = Need "NATS_${upper}_PASSWORD"
                    OTEL_METRICS_PORT = Need "${upper}_METRICS_PORT"
                } }
        }
    }
}

# --- process control ---------------------------------------------------------

function Get-RunningPid([string]$Name) {
    $file = Join-Path $RunDir "$Name.pid"
    if (-not (Test-Path $file)) { return $null }
    $id = [int](Get-Content $file)
    if (Get-Process -Id $id -ErrorAction SilentlyContinue) { return $id }
    Remove-Item $file
    return $null
}

function Start-Background([string]$Name) {
    if (Get-RunningPid $Name) { Write-Host "$Name already running"; return }
    $p = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -PassThru -WorkingDirectory $Root `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", 'exec', $Name)
    Write-Host "$Name started (pid $($p.Id))"
}

function Stop-Background([string]$Name) {
    $id = Get-RunningPid $Name
    if (-not $id) { Write-Host "$Name not running"; return }
    if ($Name -eq 'postgres') {
        & (Join-Path (Need 'POSTGRES_BIN_DIR') 'pg_ctl.exe') stop -D (Join-Path $Data 'postgres') -m fast | Out-Null
    }
    & taskkill.exe /T /F /PID $id 2>&1 | Out-Null
    Remove-Item (Join-Path $RunDir "$Name.pid") -ErrorAction SilentlyContinue
    Write-Host "$Name stopped"
}

function Invoke-Exec([string]$Name) {
    $spec = Get-Spec $Name $true
    foreach ($k in $spec.Env.Keys) { [Environment]::SetEnvironmentVariable($k, $spec.Env[$k], 'Process') }
    Set-Content (Join-Path $RunDir "$Name.pid") $PID
    # Start-Process redirects at the OS level; `&` with 2> would turn every stderr log
    # line into a PowerShell error in Windows PowerShell 5.1.
    $argLine = ($spec.Args | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    try {
        $p = Start-Process -FilePath $spec.Exe -ArgumentList $argLine -NoNewWindow -Wait -PassThru `
            -WorkingDirectory $Root `
            -RedirectStandardOutput (Join-Path $LogDir "$Name.out.log") `
            -RedirectStandardError (Join-Path $LogDir "$Name.err.log")
    } finally {
        Remove-Item (Join-Path $RunDir "$Name.pid") -ErrorAction SilentlyContinue
    }
    exit $p.ExitCode
}

$selected = @(foreach ($t in $Targets) {
    switch ($t) {
        'all' { $InfraOrder + $Services }
        'infra' { $InfraOrder }
        'services' { $Services }
        default { $t }
    }
})

switch ($Command) {
    'exec' { Invoke-Exec $selected[0] }
    'up' { foreach ($t in $selected) { Start-Background $t } }
    'down' { [array]::Reverse($selected); foreach ($t in $selected) { Stop-Background $t } }
    'status' {
        foreach ($t in $selected) {
            $id = Get-RunningPid $t
            '{0,-12} {1}' -f $t, $(if ($id) { "running (pid $id)" } else { 'stopped' })
        }
    }
}
