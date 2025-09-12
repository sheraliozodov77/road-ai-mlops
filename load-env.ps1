# load-env.ps1
Get-Content .env | ForEach-Object {
    if ($_ -match "^\s*([^#][\w_]+)=(.*)$") {
        $key = $matches[1]
        $value = $matches[2].Trim('"')
        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}
