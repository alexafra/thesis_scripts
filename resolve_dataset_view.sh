#!/usr/bin/env bash
# Sourced by the train/evaluation runners after entering Isaac-GR00T.
# No change to the legacy single-root path unless explicitly selected.
resolve_dataset_view() {
    if [[ -z "${DATASET_SOURCES:-}" && -z "${VIRTUAL_DATASET_ROOT:-}" ]]; then
        return 0
    fi
    if [[ -z "${DATASET_SOURCES:-}" || -z "${VIRTUAL_DATASET_ROOT:-}" ]]; then
        echo "Error: set both DATASET_SOURCES (colon-separated split roots) and VIRTUAL_DATASET_ROOT." >&2
        return 1
    fi
    if [[ -n "${DATASET_ROOT:-}" || -n "${TRAIN_DATASET:-}" || -n "${VALIDATION_DATASET:-}" || -n "${TEST_DATASET:-}" ]]; then
        echo "Error: DATASET_SOURCES cannot be combined with DATASET_ROOT or explicit split overrides." >&2
        return 1
    fi
    if [[ "$DATASET_SOURCES" == :* || "$DATASET_SOURCES" == *: || "$DATASET_SOURCES" == *::* ]]; then
        echo "Error: DATASET_SOURCES contains an empty path." >&2
        return 1
    fi
    local -a dataset_view_sources dataset_view_splits
    case "${EVALUATION_SPLIT:-validation}" in
        validation) dataset_view_splits=(train validation) ;;
        none) dataset_view_splits=(train) ;;
        test)
            if [[ "${ALLOW_TEST_EVALUATION:-0}" != "1" ]]; then
                echo "Error: test evaluation is deferred; ALLOW_TEST_EVALUATION=1 is required." >&2
                return 1
            fi
            dataset_view_splits=(train validation test)
            ;;
        *)
            echo "Error: unsupported EVALUATION_SPLIT for dataset-view preparation." >&2
            return 1
            ;;
    esac
    IFS=: read -r -a dataset_view_sources <<< "$DATASET_SOURCES"
    if ! .venv/bin/python -m gr00t.data.virtual_dataset \
        --sources "${dataset_view_sources[@]}" \
        --output "$VIRTUAL_DATASET_ROOT" --reuse --splits "${dataset_view_splits[@]}"; then
        echo "Error: virtual dataset preparation failed; training/evaluation will not start." >&2
        return 1
    fi
    DATASET_ROOT="$VIRTUAL_DATASET_ROOT"
    TRAIN_DATASET="$DATASET_ROOT/train"
    if [[ "${EVALUATION_SPLIT:-validation}" != "none" ]]; then
        VALIDATION_DATASET="$DATASET_ROOT/validation"
    fi
    if [[ "${EVALUATION_SPLIT:-validation}" == "test" ]]; then
        TEST_DATASET="$DATASET_ROOT/test"
    fi
}

resolve_dataset_view
