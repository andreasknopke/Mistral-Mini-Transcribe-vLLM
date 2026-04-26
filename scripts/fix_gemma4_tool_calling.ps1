[CmdletBinding()]
param(
    [string]$SparkHost = "192.168.188.173",
    [string]$SparkUser = "ksai0001_local",
    [string]$RemoteAppDir = "/home/ksai0001_local/correction-llm-vllm",
    [string]$ToolCallParser = "gemma4",
    [string]$SudoPass = ""
)

if (-not $SudoPass) {
    $secPass = Read-Host -AsSecureString "Sudo-Passwort fuer $SparkUser@$SparkHost"
    $SudoPass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secPass))
}

$sshTarget = "$SparkUser@$SparkHost"

$remoteScript = @'
set -euo pipefail

APP_DIR="$1"
TOOL_CALL_PARSER="$2"
SUDO_PASS="$3"
RUN_FILE="${APP_DIR}/run_gemma4.sh"

if [ ! -f "${RUN_FILE}" ]; then
  echo "FEHLER: ${RUN_FILE} nicht gefunden."
  exit 1
fi

python3 - "$RUN_FILE" "$TOOL_CALL_PARSER" <<'PY'
import re
import sys
from datetime import datetime
from pathlib import Path

run_file = Path(sys.argv[1])
parser = sys.argv[2]
original = run_file.read_text(encoding="utf-8")
updated = original

if "--enable-auto-tool-choice" not in updated:
    anchor = "  --moe-backend marlin \\\n"
    replacement = (
        "  --moe-backend marlin \\\n"
        "  --enable-auto-tool-choice \\\n"
        f"  --tool-call-parser {parser} \\\n"
    )
    if anchor not in updated:
        raise SystemExit("Konnte Einfügepunkt '--moe-backend marlin' in run_gemma4.sh nicht finden.")
    updated = updated.replace(anchor, replacement, 1)
elif "--tool-call-parser" not in updated:
    anchor = "  --enable-auto-tool-choice \\\n"
    replacement = (
        "  --enable-auto-tool-choice \\\n"
        f"  --tool-call-parser {parser} \\\n"
    )
    if anchor not in updated:
        raise SystemExit("Konnte Einfügepunkt '--enable-auto-tool-choice' in run_gemma4.sh nicht finden.")
    updated = updated.replace(anchor, replacement, 1)
else:
    updated = re.sub(
        r"(^\s*--tool-call-parser\s+)\S+(\s+\\$)",
        rf"\1{parser}\2",
        updated,
        count=1,
        flags=re.MULTILINE,
    )

if updated == original:
    print("Keine Änderung nötig; run_gemma4.sh enthält die gewünschten Flags bereits.")
    raise SystemExit(0)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
backup = run_file.with_name(f"{run_file.name}.bak-{timestamp}")
backup.write_text(original, encoding="utf-8")
run_file.write_text(updated, encoding="utf-8")
print(f"Patched {run_file} and created backup {backup}")
PY

echo "Starte correction-llm neu ..."
echo "$SUDO_PASS" | sudo -S systemctl restart correction-llm

echo "Warte auf http://127.0.0.1:9000/v1/models ..."
for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:9000/v1/models >/dev/null 2>&1; then
    echo "Gemma 4 ist wieder erreichbar."
    echo "$SUDO_PASS" | sudo -S systemctl --no-pager --full status correction-llm | sed -n '1,25p'
    exit 0
  fi
  sleep 2
done

echo "FEHLER: correction-llm ist nach dem Restart nicht erreichbar."
echo "$SUDO_PASS" | sudo -S systemctl --no-pager --full status correction-llm | sed -n '1,40p' || true
journalctl -u correction-llm -n 80 --no-pager || true
exit 1
'@

$unixScript = $remoteScript -replace "`r`n", "`n"
$unixScript | ssh $sshTarget "bash -s -- '$RemoteAppDir' '$ToolCallParser' '$SudoPass'"
