#!/usr/bin/env bash
# Bootstrap script for running Claude Code unattended with the autocuda plugin.
#
# Does three things, idempotently:
#   1. Installs Claude Code (native installer) if not already present.
#   2. Writes ~/.claude/settings.json with the unattended-mode defaults
#      (bypass permissions, sandboxed, max effort, opus-4-7) and appends
#      PATH / alias / env exports to ~/.bashrc.
#   3. Symlinks each skill under plugins/autocuda/skills/ into
#      ~/.claude/skills/ so Claude Code picks them up as user-scope skills.
#      (Equivalent to `/plugin marketplace add brycelelbach/autocuda` +
#      `/plugin install autocuda@brycelelbach-autocuda`, but runs without a
#      Claude Code session.)
#
# Can be run from a local checkout or piped via `curl ... | bash`. In the
# piped case, the repo is cloned into ~/autocuda (override with AUTOCUDA_DIR).
#
# Safe to re-run. Existing ~/.claude/settings.json is backed up before overwrite.

set -euo pipefail

REPO_URL="https://github.com/brycelelbach/autocuda.git"
CLAUDE_DIR="${HOME}/.claude"
SKILLS_DIR="${CLAUDE_DIR}/skills"
SETTINGS_FILE="${CLAUDE_DIR}/settings.json"
BASHRC="${HOME}/.bashrc"
BASHRC_MARKER_BEGIN="# >>> autocuda bootstrap >>>"
BASHRC_MARKER_END="# <<< autocuda bootstrap <<<"

log() { printf '[bootstrap] %s\n' "$*"; }

# Resolve the repo path. When running from a local checkout, BASH_SOURCE[0]
# names this file and we use its directory. When piped via `curl ... | bash`,
# BASH_SOURCE[0] is empty (set -u would trip without the :- default), so we
# clone the repo to a stable location and use that.
_src="${BASH_SOURCE[0]:-}"
if [[ -n "${_src}" && -f "${_src}" ]]; then
    REPO_DIR="$(cd "$(dirname "$(readlink -f "${_src}")")" && pwd)"
else
    REPO_DIR="${AUTOCUDA_DIR:-${HOME}/autocuda}"
    if [[ -d "${REPO_DIR}/.git" ]]; then
        log "using existing autocuda checkout at ${REPO_DIR}"
    else
        command -v git >/dev/null 2>&1 || { log "ERROR: git not installed; cannot clone ${REPO_URL} into ${REPO_DIR}"; exit 1; }
        log "cloning ${REPO_URL} into ${REPO_DIR}..."
        git clone "${REPO_URL}" "${REPO_DIR}"
    fi
fi
unset _src

# ---------------------------------------------------------------------------
# 1. Install Claude Code (native installer) if missing.
# ---------------------------------------------------------------------------
install_claude() {
    if command -v claude >/dev/null 2>&1; then
        log "claude already installed: $(command -v claude) ($(claude --version 2>&1 | head -1))"
        return
    fi
    if [[ -x "${HOME}/.local/bin/claude" ]]; then
        log "claude found at ~/.local/bin/claude but not on PATH yet; will be picked up after PATH update"
        return
    fi
    log "installing Claude Code via native installer..."
    curl -fsSL https://claude.ai/install.sh | bash
}

# ---------------------------------------------------------------------------
# 2a. Write ~/.claude/settings.json.
# ---------------------------------------------------------------------------
write_settings() {
    mkdir -p "${CLAUDE_DIR}"
    if [[ -f "${SETTINGS_FILE}" ]]; then
        local backup="${SETTINGS_FILE}.bak.$(date +%Y%m%d-%H%M%S)"
        cp "${SETTINGS_FILE}" "${backup}"
        log "backed up existing settings.json -> ${backup}"
    fi
    cat > "${SETTINGS_FILE}" <<'JSON'
{
  "model": "claude-opus-4-7",
  "effortLevel": "max",
  "permissions": {
    "defaultMode": "bypassPermissions"
  },
  "skipDangerousModePermissionPrompt": true,
  "env": {
    "CLAUDE_CODE_SANDBOXED": "1",
    "CLAUDE_CODE_EFFORT_LEVEL": "max"
  }
}
JSON
    log "wrote ${SETTINGS_FILE}"
}

# ---------------------------------------------------------------------------
# 2b. Append unattended-mode block to ~/.bashrc (idempotent via markers).
# ---------------------------------------------------------------------------
update_bashrc() {
    touch "${BASHRC}"
    if grep -qF "${BASHRC_MARKER_BEGIN}" "${BASHRC}"; then
        log "bashrc already has autocuda block; leaving it alone"
        return
    fi
    cat >> "${BASHRC}" <<EOF

${BASHRC_MARKER_BEGIN}
# Sources env file created by the Claude Code native installer, ensures
# ~/.local/bin is on PATH, and makes every interactive 'claude' invocation
# skip the permission prompt so the agent can run unattended.
if [ -f "\$HOME/.local/bin/env" ]; then
    . "\$HOME/.local/bin/env"
fi
export PATH="\$HOME/.local/bin:\$PATH"
export CLAUDE_CODE_SANDBOXED=1
alias claude='claude --dangerously-skip-permissions'
${BASHRC_MARKER_END}
EOF
    log "appended unattended-mode block to ${BASHRC}"
}

# ---------------------------------------------------------------------------
# 3. Symlink the autocuda plugin's skills into ~/.claude/skills/.
# ---------------------------------------------------------------------------
link_skills() {
    mkdir -p "${SKILLS_DIR}"
    local plugin_skills_dir="${REPO_DIR}/plugins/autocuda/skills"
    if [[ ! -d "${plugin_skills_dir}" ]]; then
        log "warning: no plugin skills dir at ${plugin_skills_dir}"
        return
    fi
    local src
    local count=0
    shopt -s nullglob
    for src in "${plugin_skills_dir}"/*/; do
        src="${src%/}"
        local name
        name="$(basename "${src}")"
        local dest="${SKILLS_DIR}/${name}"
        if [[ -L "${dest}" || -e "${dest}" ]]; then
            rm -rf "${dest}"
        fi
        ln -s "${src}" "${dest}"
        log "linked skill: ${dest} -> ${src}"
        count=$((count + 1))
    done
    shopt -u nullglob
    if [[ "${count}" -eq 0 ]]; then
        log "warning: no skills found under ${plugin_skills_dir}"
    fi
}

main() {
    log "repo: ${REPO_DIR}"
    install_claude
    write_settings
    update_bashrc
    link_skills
    log "done. Open a new shell (or 'source ~/.bashrc') so the PATH / alias take effect."
}

main "$@"
