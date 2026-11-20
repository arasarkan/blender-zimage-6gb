# Prints the Python executable of the ComfyUI server listening on the given port.
param([int]$Port = 8188)
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $conn) { exit 1 }
$proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)"
# A venv launcher shows up as the base interpreter in ExecutablePath; the command
# line's first token is the venv python that actually has ComfyUI's packages.
$cmd = $proc.CommandLine.Trim()
if ($cmd.StartsWith('"')) { $first = $cmd.Substring(1, $cmd.IndexOf('"', 1) - 1) }
else { $first = ($cmd -split '\s+')[0] }
if ([IO.Path]::IsPathRooted($first) -and (Test-Path $first)) { $first } else { $proc.ExecutablePath }
