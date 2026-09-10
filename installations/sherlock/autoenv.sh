# EnforceFlux — load the Sherlock toolchain on entering the project directory.
#
# Source this from ~/.bashrc; see the snippet at the bottom of this file.
#
# Entering the tree loads installations/sherlock/modules.sh and activates
# .venv; leaving puts your previous module set back.
#
# The project root is derived from THIS FILE's own location, so the checkout
# can live anywhere and be named anything. Export ENFORCEFLUX_ROOT before
# sourcing to override.
#
# Interactive shells only. ~/.bashrc is also read by non-interactive sessions,
# and anything written to stdout there corrupts scp/sftp transfers.
case $- in
    *i*) ;;
    *) return 0 2>/dev/null || exit 0 ;;
esac

# Locate the checkout from this script's own path: <root>/installations/sherlock.
# A relative path would be wrong here — PROMPT_COMMAND runs from whatever
# directory you happen to be in, so there is nothing stable to be relative to.
if [ -z "${ENFORCEFLUX_ROOT:-}" ]; then
    _efx_self="${BASH_SOURCE[0]:-$0}"
    # Follow symlinks when the file is linked in from elsewhere.
    if command -v readlink >/dev/null 2>&1; then
        _efx_resolved="$(readlink -f "$_efx_self" 2>/dev/null)" && [ -n "$_efx_resolved" ] \
            && _efx_self="$_efx_resolved"
    fi
    ENFORCEFLUX_ROOT="$(cd -P "$(dirname "$_efx_self")/../.." >/dev/null 2>&1 && pwd)"
    unset _efx_self _efx_resolved
fi
export ENFORCEFLUX_ROOT

# Refuse to guess: without a real checkout the hook would silently do nothing.
if [ ! -f "$ENFORCEFLUX_ROOT/installations/sherlock/modules.sh" ]; then
    echo "autoenv.sh: cannot locate the EnforceFlux checkout (tried '$ENFORCEFLUX_ROOT');" \
         "export ENFORCEFLUX_ROOT and re-source." >&2
    return 0 2>/dev/null || exit 0
fi

_efx_in_project() {
    # Bail on an empty root: the case pattern would collapse to /* and match
    # every directory on the system.
    [ -n "${ENFORCEFLUX_ROOT:-}" ] || return 1
    case "$PWD/" in
        "$ENFORCEFLUX_ROOT"/*) return 0 ;;
        *) return 1 ;;
    esac
}

_efx_load() {
    [ -n "${_EFX_ACTIVE:-}" ] && return 0
    # Re-check at call time: ENFORCEFLUX_ROOT can be changed after sourcing.
    [ -f "$ENFORCEFLUX_ROOT/installations/sherlock/modules.sh" ] || return 0
    # modules.sh starts with `module reset`, so record what was loaded first —
    # otherwise leaving the tree would strand you without your usual stack.
    _EFX_SAVED_MODULES="$(module -t list 2>&1 | grep -v ':$' | grep -v '^No modules' | tr '\n' ' ')"
    _EFX_SAVED_PS1="$PS1"

    . "$ENFORCEFLUX_ROOT/installations/sherlock/modules.sh" >/dev/null 2>&1

    if [ -f "$ENFORCEFLUX_ROOT/.venv/bin/activate" ]; then
        VIRTUAL_ENV_DISABLE_PROMPT=1 . "$ENFORCEFLUX_ROOT/.venv/bin/activate"
    fi

    # Built binaries, so `FLEXPART` / `microhh` are runnable by name.
    [ -d "$ENFORCEFLUX_ROOT/flexpart/src" ] && PATH="$ENFORCEFLUX_ROOT/flexpart/src:$PATH"
    [ -d "$ENFORCEFLUX_ROOT/microhh/build" ] && PATH="$ENFORCEFLUX_ROOT/microhh/build:$PATH"
    export PATH
    # The CPU/MPI build; the GPU build lives in microhh/build_gpu.
    [ -x "$ENFORCEFLUX_ROOT/microhh/build/microhh" ] \
        && export MICROHH_BIN="$ENFORCEFLUX_ROOT/microhh/build/microhh"

    _EFX_ACTIVE=1
    PS1="(efx) $PS1"
    printf '\033[32m✓ EnforceFlux env\033[0m  python %s  |  eccodes %s  |  %s\n' \
        "$(python3 -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null || echo '?')" \
        "${ECCODES_PREFIX:-unset}" \
        "$([ -n "${ENFORCEFLUX_SHERLOCK:-}" ] && echo 'modules loaded' || echo 'MODULES FAILED')"
}

_efx_unload() {
    [ -z "${_EFX_ACTIVE:-}" ] && return 0
    command -v deactivate >/dev/null 2>&1 && deactivate
    module reset >/dev/null 2>&1 || module purge >/dev/null 2>&1
    if [ -n "${_EFX_SAVED_MODULES:-}" ]; then
        module load $_EFX_SAVED_MODULES >/dev/null 2>&1
    fi
    [ -n "${_EFX_SAVED_PS1:-}" ] && PS1="$_EFX_SAVED_PS1"
    unset _EFX_ACTIVE _EFX_SAVED_MODULES _EFX_SAVED_PS1 ENFORCEFLUX_SHERLOCK MICROHH_BIN
    echo "• EnforceFlux env unloaded"
}

_efx_auto() {
    if _efx_in_project; then _efx_load; else _efx_unload; fi
}

# Manual escape hatches, for batch scripts or when you want it without cd'ing.
efx-env() { _efx_load; }
efx-env-off() { _efx_unload; }

case ";${PROMPT_COMMAND:-};" in
    *";_efx_auto;"*) ;;
    *) PROMPT_COMMAND="_efx_auto;${PROMPT_COMMAND:-}" ;;
esac

# ── ~/.bashrc snippet ────────────────────────────────────────────────────────
# Something has to know where the checkout is; everything past that point is
# derived. Set ENFORCEFLUX_ROOT if yours is not in one of the usual spots:
#
#   for _d in "${ENFORCEFLUX_ROOT:-}" "$HOME/EnforceFlux" \
#             "$HOME/projects/EnforceFlux" "$GROUP_HOME/EnforceFlux"; do
#       [ -n "$_d" ] && [ -f "$_d/installations/sherlock/autoenv.sh" ] && {
#           . "$_d/installations/sherlock/autoenv.sh"; break; }
#   done
#   unset _d
