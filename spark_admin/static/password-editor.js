const passwordInput = document.getElementById('shared-password');
const passwordFile = document.getElementById('password-file');
const passwordKey = document.getElementById('password-key');
const passwordStatus = document.getElementById('password-status');
const togglePasswordButton = document.getElementById('toggle-password');
const reloadPasswordButton = document.getElementById('reload-password');
const savePasswordButton = document.getElementById('save-password');

function showToast(message, isError = false) {
    const node = document.createElement('div');
    node.className = 'toast';
    node.textContent = message;
    node.style.borderColor = isError ? 'rgba(239,68,68,0.5)' : 'rgba(34,197,94,0.4)';
    document.body.appendChild(node);
    setTimeout(() => node.remove(), 2800);
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options,
    });

    if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
            const payload = await response.json();
            detail = payload.detail || detail;
        } catch {
        }
        throw new Error(detail);
    }

    return response.json();
}

function setStatus(message, isError = false) {
    if (!passwordStatus) return;
    passwordStatus.textContent = message;
    passwordStatus.style.color = isError ? 'var(--bad)' : 'var(--muted)';
}

function applyPasswordConfig(config) {
    if (passwordInput) passwordInput.value = config.value || '';
    if (passwordFile) passwordFile.textContent = config.path || 'n/a';
    if (passwordKey) passwordKey.textContent = config.key || 'n/a';
}

async function loadPassword() {
    setStatus('Lade Passwort...');
    try {
        const config = await fetchJson('/api/shared-password');
        applyPasswordConfig(config);
        setStatus('Passwort geladen.');
    } catch (error) {
        setStatus(`Fehler beim Laden: ${error.message}`, true);
        showToast(error.message, true);
    }
}

async function savePassword() {
    setStatus('Speichere Passwort...');
    try {
        const payload = await fetchJson('/api/shared-password', {
            method: 'PUT',
            body: JSON.stringify({ value: passwordInput?.value ?? '' }),
        });
        applyPasswordConfig(payload.config);
        setStatus(payload.message || 'Gespeichert.');
        showToast(payload.message || 'Passwort gespeichert.');
    } catch (error) {
        setStatus(`Fehler beim Speichern: ${error.message}`, true);
        showToast(error.message, true);
    }
}

if (togglePasswordButton && passwordInput) {
    togglePasswordButton.addEventListener('click', () => {
        const hidden = passwordInput.type === 'password';
        passwordInput.type = hidden ? 'text' : 'password';
        togglePasswordButton.textContent = hidden ? 'Verbergen' : 'Anzeigen';
    });
}

if (reloadPasswordButton) {
    reloadPasswordButton.addEventListener('click', loadPassword);
}

if (savePasswordButton) {
    savePasswordButton.addEventListener('click', savePassword);
}

loadPassword();
