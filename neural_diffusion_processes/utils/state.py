"""Based on code from Erik Bodin"""

import equinox as eqx
import optax
import os

from jaxtyping import Array, PyTree


_DIRECTORY_PATH_CHECKPOINTS = "{path}/checkpoints"
_STEP_FILENAME_PREFIX = "ckpt-{step}"
_STEP_FILENAME_EXT = ".eqx"


class TrainingState(eqx.Module):
    params: PyTree
    params_ema: PyTree
    opt_state: optax.OptState
    key: Array
    step: Array


def save_checkpoint(training_state, directory_path: str, step_index: int):
    assert isinstance(step_index, int)
    _cond_mkdir(directory_path)
    directory_path = _DIRECTORY_PATH_CHECKPOINTS.format(path=directory_path)
    _save_pytree(
        training_state,
        directory_path=directory_path,
        step_index=step_index,
    )


def load_checkpoint(pytree_like, directory_path: str, step_index: int) -> TrainingState:
    directory_path = _DIRECTORY_PATH_CHECKPOINTS.format(path=directory_path)
    print("Loading checkpoint from", directory_path)
    return _load_pytree(pytree_like, directory_path, step_index=step_index)


def find_latest_checkpoint_step_index(directory_path: str):
    path = _DIRECTORY_PATH_CHECKPOINTS.format(path=directory_path)
    if not os.path.exists(path):
        return None
    indices = [
        _checkpoint_filename_to_index(name)
        for name in os.listdir(path)
        if name.endswith(_STEP_FILENAME_EXT)
    ]
    if len(indices) == 0:
        return None
    return max(indices)


def _save_pytree(pytree: PyTree, directory_path: str, step_index: int):
    _cond_mkdir(directory_path)
    filepath = (
        directory_path + "/" + _index_to_checkpoint_filename(step_index=step_index)
    )
    eqx.tree_serialise_leaves(filepath, pytree=pytree)


def _load_pytree(pytree_like, directory_path: str, step_index: int):
    filepath = (
        directory_path + "/" + _index_to_checkpoint_filename(step_index=step_index)
    )
    return eqx.tree_deserialise_leaves(filepath, like=pytree_like)


def _index_to_checkpoint_filename(step_index: int):
    return _STEP_FILENAME_PREFIX.format(step=step_index) + _STEP_FILENAME_EXT


def _checkpoint_filename_to_index(filename: str):
    start = filename.index("-") + 1
    end = filename.index(_STEP_FILENAME_EXT)
    filename_part = filename[start:end]
    return int(filename_part)


def _cond_mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)
