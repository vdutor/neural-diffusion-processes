# Copyright 2022 The CLU Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PeriodicActions execute small actions periodically in the training loop."""

import abc
import collections
import time
from typing import Callable, Iterable, Optional

from absl import logging

from . import asynclib


class _PeriodicAction(abc.ABC):
    """Abstract base class for perodic actions.
    The idea is that the user creates periodic actions and calls them after
    each training step. The base class will trigger in fixed step/time interval
    but subclasses can overwrite `_should_trigger()` to change this behavior.
    Subclasses must implement `_apply()` to perform the action.
    """

    def __init__(
        self,
        *,
        every_steps: Optional[int] = None,
        every_secs: Optional[float] = None,
        on_steps: Optional[Iterable[int]] = None,
    ):
        """Creates an action that triggers periodically.
        Args:
          every_steps: If the current step is divisible by `every_steps`, then an
            action is triggered.
          every_secs: If no action has triggered for specified `every_secs`, then
            an action is triggered. Note that the previous action might have been
            triggered by `every_steps` or by `every_secs`.
          on_steps: If the current step is included in this set, then an action is
            triggered.
        """
        self._every_steps = every_steps
        self._every_secs = every_secs
        self._on_steps = set(on_steps or [])
        # Step and timestamp for the last time the action triggered.
        self._previous_step = None
        self._previous_time = None
        # Just for checking that __call__() was called every step.
        self._last_step = None

    def _init_and_check(self, step: int, t: float):
        """Initializes and checks it was called at every step."""
        if self._previous_step is None:
            self._previous_step = step
            self._previous_time = t
            self._last_step = step
        elif self._every_steps is not None and step - self._last_step != 1:
            raise ValueError(
                f"PeriodicAction must be called after every step once "
                f"(every_steps={self._every_steps}, "
                f"previous_step={self._previous_step}, step={step})."
            )
        else:
            self._last_step = step

    def _should_trigger(self, step: int, t: float) -> bool:
        """Return whether the action should trigger this step."""
        if self._every_steps is not None and step % self._every_steps == 0:
            return True
        if self._every_secs is not None and t - self._previous_time > self._every_secs:
            return True
        if step in self._on_steps:
            return True
        return False

    def _after_apply(self, step: int, t: float):
        """Called after each time the action triggered."""
        self._previous_step = step
        self._previous_time = t

    def __call__(self, step: int, t: Optional[float] = None) -> bool:
        """Method to call the hook after every training step.
        Args:
          step: Current step.
          t: Optional timestamp. Will use `time.monotonic()` if not specified.
        Returns:
          True if the action triggered, False otherwise. Note that the first
          invocation never triggers.
        """
        if t is None:
            t = time.monotonic()

        self._init_and_check(step, t)
        if self._should_trigger(step, t):
            self._apply(step, t)
            self._after_apply(step, t)
            return True
        return False

    @abc.abstractmethod
    def _apply(self, step: int, t: float):
        pass


class PeriodicCallback(_PeriodicAction):
    """This hook calls a callback function each time it triggers."""

    def __init__(
        self,
        *,
        every_steps: Optional[int] = None,
        every_secs: Optional[float] = None,
        on_steps: Optional[Iterable[int]] = None,
        callback_fn: Callable,
        execute_async: bool = False,
        pass_step_and_time: bool = True,
    ):
        """Initializes a new periodic Callback action.
        Args:
          every_steps: See `PeriodicAction.__init__()`.
          every_secs: See `PeriodicAction.__init__()`.
          on_steps: See `PeriodicAction.__init__()`.
          callback_fn: A callback function. It must accept `step` and `t` as
            arguments; arguments are passed by keyword.
          execute_async: if True wraps the callback into an async call.
          pass_step_and_time: if True the step and t are passed to the callback.
        """
        super().__init__(every_steps=every_steps, every_secs=every_secs, on_steps=on_steps)
        self._cb_results = collections.deque(maxlen=1)
        self.pass_step_and_time = pass_step_and_time
        if execute_async:
            logging.info(
                "Callback will be executed asynchronously. "
                "Errors are raised when they become available."
            )
            self._cb_fn = asynclib.Pool(callback_fn.__name__)(callback_fn)
        else:
            self._cb_fn = callback_fn

    def __call__(self, step: int, t: Optional[float] = None, **kwargs) -> bool:
        if t is None:
            t = time.monotonic()

        self._init_and_check(step, t)
        if self._should_trigger(step, t):
            # Additional arguments to the callback are passed here through **kwargs.
            self._apply(step, t, **kwargs)
            self._after_apply(step, t)
            return True
        return False

    def get_last_callback_result(self):
        """Returns the last cb result."""
        return self._cb_results[0]

    def _apply(self, step, t, **kwargs):
        if self.pass_step_and_time:
            result = self._cb_fn(step=step, t=t, **kwargs)
        else:
            result = self._cb_fn(**kwargs)
        self._cb_results.append(result)
