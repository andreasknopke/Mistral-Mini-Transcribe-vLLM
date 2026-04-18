#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/spark_admin"
INSTALL_DIR="${SPARK_ADMIN_INSTALL_DIR:-/opt/spark-admin}"
SERVICE_NAME="spark-admin"
HOST="${SPARK_ADMIN_HOST:-0.0.0.0}"
PORT="${SPARK_ADMIN_PORT:-7000}"
STACK_OWNER="${SPARK_STACK_OWNER:-${SUDO_USER:-${USER:-ksai0001_local}}}"
STACK_HOME="${SPARK_STACK_HOME:-/home/${STACK_OWNER}}"
ENV_FILE="/etc/${SERVICE_NAME}.env"

run_sudo() {
  if [ -n "${SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

echo "==============================================="
echo " Spark Admin Webserver installieren"
echo "==============================================="
echo ""
echo "Quelle:                  ${SOURCE_DIR}"
echo "Installationsordner:     ${INSTALL_DIR}"
echo "Service:                 ${SERVICE_NAME}"
echo "Port:                    ${PORT}"
echo "Stack Owner:             ${STACK_OWNER}"
echo "Stack Home:              ${STACK_HOME}"
echo ""

if [ ! -d "${SOURCE_DIR}" ]; then
  echo "spark_admin Quelle nicht gefunden: ${SOURCE_DIR}"
  echo "Bitte zuerst das Repo bzw. die Deploy-Dateien inklusive spark_admin/ kopieren."
  exit 1
fi

echo "[1/6] Systempakete prüfen"
run_sudo apt update
run_sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential libpam0g-dev

echo "[2/6] App-Dateien nach ${INSTALL_DIR} kopieren"
run_sudo mkdir -p "${INSTALL_DIR}"
run_sudo rm -rf "${INSTALL_DIR}/templates" "${INSTALL_DIR}/static"
run_sudo cp "${SOURCE_DIR}/app.py" "${INSTALL_DIR}/app.py"
run_sudo cp "${SOURCE_DIR}/requirements.txt" "${INSTALL_DIR}/requirements.txt"
run_sudo cp -r "${SOURCE_DIR}/templates" "${INSTALL_DIR}/templates"
run_sudo cp -r "${SOURCE_DIR}/static" "${INSTALL_DIR}/static"

echo "[3/6] Python Virtualenv einrichten"
run_sudo python3 -m venv "${INSTALL_DIR}/.venv"
run_sudo "${INSTALL_DIR}/.venv/bin/pip" install --upgrade pip
run_sudo "${INSTALL_DIR}/.venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"

echo "[4/6] Environment-Datei schreiben"
if run_sudo test -f "${ENV_FILE}"; then
  EXISTING_SECRET="$(run_sudo awk -F= '/^SPARK_ADMIN_SESSION_SECRET=/{print $2}' "${ENV_FILE}" | tail -n 1)"
else
  EXISTING_SECRET=""
fi
if [ -z "${EXISTING_SECRET}" ]; then
  EXISTING_SECRET="$(python3 - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"
fi

TMP_ENV="$(mktemp)"
cat > "${TMP_ENV}" <<EOF
SPARK_ADMIN_HOST=${HOST}
SPARK_ADMIN_PORT=${PORT}
SPARK_STACK_OWNER=${STACK_OWNER}
SPARK_STACK_HOME=${STACK_HOME}
SPARK_ADMIN_SESSION_SECRET=${EXISTING_SECRET}
SPARK_ADMIN_PAM_SERVICE=login
EOF
run_sudo mv "${TMP_ENV}" "${ENV_FILE}"
run_sudo chmod 600 "${ENV_FILE}"

echo "[5/6] systemd Service einrichten"
SERVICE_FILE="$(mktemp)"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Spark Admin Control Center
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${INSTALL_DIR}/.venv/bin/python ${INSTALL_DIR}/app.py
Restart=always
RestartSec=5
TimeoutStartSec=30

[Install]
WantedBy=multi-user.target
EOF
run_sudo mv "${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"
run_sudo systemctl daemon-reload
run_sudo systemctl enable "${SERVICE_NAME}"

echo "[6/6] Fertig"
echo ""
echo "Starten:           sudo systemctl start ${SERVICE_NAME}"
echo "Status:            sudo systemctl status ${SERVICE_NAME} --no-pager -l"
echo "Logs:              journalctl -u ${SERVICE_NAME} -f"
echo "URL:               http://127.0.0.1:${PORT}"
echo "Login:             Linux-Benutzername und Passwort des Spark"
