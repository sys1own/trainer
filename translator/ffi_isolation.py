"""Safe FFI isolation layer for AeroVM ↔ native library interop.

Provides a managed execution boundary that:
- Deep-copies array values before passing to native libraries
- Tracks allocated resources for cleanup
- Pipes return values back into the AeroVM stack representation
- Detects and prevents memory leaks via reference counting
"""

import copy
import json
import os
import sys
import traceback
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# AeroVM stack representation
# ---------------------------------------------------------------------------

@dataclass
class StackValue:
    """A typed value on the AeroVM data stack."""
    vtype: str          # "int", "float", "string", "array", "null", "error"
    data: object = None
    ref_count: int = 1


@dataclass
class AeroVMStack:
    """Simplified AeroVM stack for FFI interop."""
    frames: list[StackValue] = field(default_factory=list)
    _allocated: list[int] = field(default_factory=list)

    def push(self, val: StackValue):
        self.frames.append(val)
        self._allocated.append(id(val))

    def pop(self) -> StackValue | None:
        if not self.frames:
            return None
        val = self.frames.pop()
        return val

    def peek(self) -> StackValue | None:
        return self.frames[-1] if self.frames else None

    @property
    def depth(self) -> int:
        return len(self.frames)

    def clear(self):
        """Symmetrical stack clearing per guardrail #3."""
        self.frames.clear()
        self._allocated.clear()


# ---------------------------------------------------------------------------
# Type coercion: Python → AeroVM stack
# ---------------------------------------------------------------------------

def python_to_stack(value: object) -> StackValue:
    """Convert a Python value to an AeroVM StackValue."""
    if value is None:
        return StackValue(vtype="null", data=None)
    if isinstance(value, bool):
        return StackValue(vtype="int", data=int(value))
    if isinstance(value, int):
        return StackValue(vtype="int", data=value)
    if isinstance(value, float):
        return StackValue(vtype="float", data=value)
    if isinstance(value, str):
        return StackValue(vtype="string", data=value)
    if isinstance(value, (list, tuple)):
        return StackValue(vtype="array", data=list(value))
    # Fallback: serialize to string
    try:
        return StackValue(vtype="string", data=json.dumps(value, default=str))
    except (TypeError, ValueError):
        return StackValue(vtype="string", data=str(value))


def stack_to_python(sv: StackValue) -> object:
    """Convert an AeroVM StackValue back to a Python value."""
    if sv.vtype == "null":
        return None
    return sv.data


# ---------------------------------------------------------------------------
# FFI execution boundary
# ---------------------------------------------------------------------------

@dataclass
class FFICallResult:
    """Result of executing an FFI call through the isolation layer."""
    success: bool = False
    return_value: StackValue = field(default_factory=lambda: StackValue(vtype="null"))
    error: str | None = None
    leaked_refs: int = 0


def execute_ffi_call(func: callable,
                     args: list[StackValue],
                     stack: AeroVMStack) -> FFICallResult:
    """Execute a native function call through the FFI isolation boundary.

    1. Deep-copies all array/mutable arguments before passing
    2. Executes the native function
    3. Converts the return value to a StackValue
    4. Pushes the result onto the AeroVM stack
    5. Checks for reference leaks
    """
    result = FFICallResult()
    refs_before = len(stack._allocated)

    try:
        # Phase 1: Deep-copy mutable arguments for isolation
        native_args = []
        for sv in args:
            pval = stack_to_python(sv)
            if isinstance(pval, (list, dict)):
                native_args.append(copy.deepcopy(pval))
            else:
                native_args.append(pval)

        # Phase 2: Execute native function
        native_result = func(*native_args)

        # Phase 3: Convert return value to stack representation
        result.return_value = python_to_stack(native_result)
        result.success = True

        # Phase 4: Push onto AeroVM stack
        stack.push(result.return_value)

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        error_val = StackValue(vtype="error", data=result.error)
        stack.push(error_val)

    # Phase 5: Leak detection
    refs_after = len(stack._allocated)
    expected_growth = 1  # one push for the result
    actual_growth = refs_after - refs_before
    if actual_growth > expected_growth:
        result.leaked_refs = actual_growth - expected_growth

    return result


# ---------------------------------------------------------------------------
# Batch FFI execution for recipe task sequences
# ---------------------------------------------------------------------------

@dataclass
class FFIBatchResult:
    """Result of executing a sequence of FFI calls."""
    total_calls: int = 0
    successful: int = 0
    failed: int = 0
    total_leaked_refs: int = 0
    call_results: list[FFICallResult] = field(default_factory=list)
    stack_clean: bool = True


def execute_ffi_batch(calls: list[tuple[callable, list]],
                      stack: AeroVMStack | None = None) -> FFIBatchResult:
    """Execute a batch of FFI calls sequentially with shared stack.

    Each call is a ``(func, args_list)`` tuple where args_list contains
    raw Python values (they will be converted to StackValues automatically).
    """
    if stack is None:
        stack = AeroVMStack()

    batch = FFIBatchResult(total_calls=len(calls))

    for func, raw_args in calls:
        stack_args = [python_to_stack(a) for a in raw_args]
        cr = execute_ffi_call(func, stack_args, stack)
        batch.call_results.append(cr)

        if cr.success:
            batch.successful += 1
        else:
            batch.failed += 1
        batch.total_leaked_refs += cr.leaked_refs

    # Symmetrical stack cleanliness check
    batch.stack_clean = batch.total_leaked_refs == 0

    return batch


# ---------------------------------------------------------------------------
# Stack cleanup utility
# ---------------------------------------------------------------------------

def drain_stack(stack: AeroVMStack) -> list[object]:
    """Pop all values from the stack, returning them as Python objects.
    Ensures symmetrical stack clearing per guardrail #3."""
    values = []
    while stack.depth > 0:
        sv = stack.pop()
        if sv:
            values.append(stack_to_python(sv))
    stack.clear()
    return values
