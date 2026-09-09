# EnforceFlux — load the Sherlock toolchain on entering the project directory.
#
# Source this from ~/.bashrc:
#     [ -f "$HOME/EnforceFlux/installations/sherlock/autoenv.sh" ] \
#         && . "$HOME/EnforceFlux/installations/sherlock/autoenv.sh"
#
# Entering the tree loads installations/sherlock/modules.sh and activates
# .venv; leaving puts your previous module set back. Override the location by
# exporting ENFORCEFLUX_ROOT before sourcing.
#
# Interactive shells only. ~/.bashrc is also read by non-interactive sessions,
# and anything written to stdout there corrupts scp/sftp transfers.
case $- in
    *i*) ;;
    *) return 0 2>/dev/null || exit 0 ;;
esac

: "${ENFORCEFLUX_ROOT:=$HOME/EnforceFlux}"
export ENFORCEFLUX_ROOT

_efx_in_project() {
    case "$PWD/" in
        "$ENFORCEFLUX_ROOT"/*) return 0 ;;
        *) return 1 ;;
    esac
}

_efx_load() {
    [ -n "${_EFX_ACTIVE:-}" ] && return 0
    if [ ! -f "$ENFORCEFLUX_ROOT/installations/sherlock/modules.sh" ]; then
        return 0
    fi
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
