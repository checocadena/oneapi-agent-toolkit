"""Find SYCLomatic helper (``dpct::``) usage that still needs converting to idiomatic SYCL.

Comments and string literals are stripped before scanning (line numbers are
preserved), so the ``DPCTxxxx`` explanation blocks do not count as code.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# Ordered: first matching rule wins. Patterns match the symbol *after* "dpct::".
CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("library_wrappers", re.compile(r"^(blas|lapack|fft|rng|sparse|dnnl|ccl|blas_utils)\b")),
    ("dpl_algorithms", re.compile(r"^(dpl|make_counting_iterator|device_pointer|device_vector|"
                                  r"device_iterator|malloc_device|get_raw_pointer|"
                                  r"iota|sort|transform_if|partition_point|"
                                  r"inclusive_scan_by_segment|exclusive_scan_by_segment|"
                                  r"reduce_by_key|copy_if|remove_if)\b")),
    ("images_textures", re.compile(r"^(image_|sampling_info|image_wrapper|image_matrix|"
                                   r"image_channel|image_data|image_accessor|"
                                   r"experimental::(bindless|image))")),
    ("device_queue_management", re.compile(r"^(get_current_device|get_default_queue|"
                                           r"get_in_order_queue|get_out_of_order_queue|"
                                           r"device_ext|dev_mgr|select_device|get_device|"
                                           r"get_current_device_id|get_device_id|"
                                           r"device_info|get_saved_queue|queue_ptr|"
                                           r"cpu_device|get_major_version|get_minor_version|"
                                           r"has_capability_or_fail|"
                                           r"get_default_context)\b")),
    ("memory_management", re.compile(r"^(dpct_malloc|dpct_free|dpct_memcpy|dpct_memset|"
                                     r"async_dpct_|memcpy_direction|"
                                     r"device_memory|global_memory|constant_memory|"
                                     r"shared_memory|get_buffer|get_access|pointer_attributes|"
                                     r"memory_region|mem_mgr|pitched_data|memcpy_parameter|"
                                     r"detail::(dpct_|mem_mgr)|"
                                     r"host_memory|free_async|"
                                     r"get_host_ptr|is_device_ptr|"
                                     r"deprecated)\b")),
    ("error_handling", re.compile(r"^(err0|err1|get_error_string_dummy|"
                                  r"error_code|exception_handler)\b")),
    ("events_timing", re.compile(r"^(event_ptr|sync_barrier|get_current_time|"
                                 r"destroy_event|timer|"
                                 r"wait_and_throw)\b")),
    ("subgroup_warp", re.compile(r"^(select_from_sub_group|shift_sub_group_left|"
                                 r"shift_sub_group_right|permute_sub_group_by_xor|"
                                 r"match_any_over_sub_group|match_all_over_sub_group|"
                                 r"experimental::nd_range_barrier|"
                                 r"group::|experimental::(logical_group|group_))")),
    ("atomics", re.compile(r"^(atomic_|atomic\b)")),
    ("math_intrinsics", re.compile(r"^(min|max|pow|cast_|fast_|byte_level_permute|"
                                   r"clamp|vectorized_|reverse_bits|ffs|"
                                   r"length|compare|unordered_compare|"
                                   r"relu|cabs|cmul|cdiv|cconj|"
                                   r"bit_cast|funnelshift|"
                                   r"extend_|dp2a|dp4a|"
                                   r"vectorized_binary|vectorized_unary)\b")),
    ("types_launch", re.compile(r"^(dim3|kernel_function|kernel_functor|"
                                r"kernel_library|image_matrix_p|"
                                r"get_kernel_function|load_kernel_library)\b")),
]

SUGGESTIONS: dict[str, str] = {
    "device_queue_management": (
        "Own an explicit sycl::queue (e.g. created once in main and passed down) instead of "
        "dpct::get_current_device()/in_order_queue(); query device info with "
        "q.get_device().get_info<sycl::info::device::...>()."
    ),
    "memory_management": (
        "Use sycl::malloc_device/malloc_shared + q.memcpy/q.memset + sycl::free(ptr, q). Replace "
        "dpct::constant_memory/global_memory with a plain USM allocation or a kernel argument."
    ),
    "error_handling": (
        "SYCL reports errors with exceptions: delete dpct::err0/DPCT_CHECK_ERROR plumbing and wrap "
        "the work in try { ... } catch (const sycl::exception&) plus an async_handler on the queue."
    ),
    "events_timing": (
        "Use sycl::event returned by submit/parallel_for; for timing create the queue with "
        "sycl::property::queue::enable_profiling and read event.get_profiling_info<...>()."
    ),
    "subgroup_warp": (
        "Use SYCL 2020 group algorithms: sycl::shift_group_left/right, sycl::select_from_group, "
        "sycl::permute_group_by_xor, sycl::reduce_over_group on it.get_sub_group()."
    ),
    "atomics": (
        "Use sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, "
        "sycl::access::address_space::global_space>(*ptr).fetch_add(v)."
    ),
    "math_intrinsics": (
        "Use sycl:: math builtins (sycl::min, sycl::max, sycl::pow, sycl::clamp, sycl::bit_cast) "
        "or std:: equivalents in host code."
    ),
    "types_launch": (
        "Use sycl::range<N>/sycl::nd_range<N> directly instead of dpct::dim3 and launch kernels "
        "with q.parallel_for."
    ),
    "images_textures": (
        "Replace dpct image wrappers with plain USM buffers + explicit interpolation, or "
        "sycl::ext::oneapi::experimental bindless images if hardware sampling is required."
    ),
    "library_wrappers": (
        "Call the library directly instead of dpct::blas/fft/rng wrappers: oneMath "
        "(oneapi::math::blas::column_major::gemm(q, ...), vendor-portable) or oneMKL / oneDNN / "
        "oneCCL. Replace a dpct::blas::descriptor handle with the sycl::queue itself."
    ),
    "dpl_algorithms": (
        "Call oneDPL directly (oneapi::dpl::execution::make_device_policy(q) + std-like algorithms)."
    ),
    "include": "Remove #include <dpct/...> once no dpct:: symbols remain.",
    "macro": "Remove DPCT_* macros (e.g. DPCT_CHECK_ERROR, DPCT_PROFILING_ENABLED) and use plain SYCL.",
    "other": "Unclassified dpct:: helper: inspect and replace with the SYCL 2020 equivalent.",
}

_SYMBOL_RE = re.compile(r"\bdpct::((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)")
_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"](dpct/[^>\"]+)[>\"]")
_MACRO_RE = re.compile(r"\b(DPCT_[A-Z0-9_]+)\b")


@dataclass
class HelperUse:
    file: str
    line: int
    symbol: str  # e.g. "dpct::get_current_device", "#include <dpct/dpct.hpp>", "DPCT_CHECK_ERROR"
    category: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def categorize(symbol_after_ns: str) -> str:
    for name, pattern in CATEGORY_RULES:
        if pattern.search(symbol_after_ns):
            return name
    return "other"


def strip_comments_and_strings(src: str) -> str:
    """Blank out comments and string/char literals, preserving newlines (and thus line numbers)."""
    out: list[str] = []
    i, n = 0, len(src)
    state = "code"
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            if c == '"':
                state = "string"
                out.append('"')
                i += 1
                continue
            if c == "'":
                state = "char"
                out.append("'")
                i += 1
                continue
            out.append(c)
            i += 1
        elif state == "line_comment":
            if c == "\n":
                out.append("\n")
                state = "code"
            i += 1
        elif state == "block_comment":
            if c == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
            if c == "\n":
                out.append("\n")
            i += 1
        else:  # string or char literal
            quote = '"' if state == "string" else "'"
            if c == "\\":
                i += 2
                continue
            if c == quote:
                out.append(quote)
                state = "code"
            elif c == "\n":  # unterminated literal; recover
                out.append("\n")
                state = "code"
            i += 1
    return "".join(out)


def scan_source(source: str, filename: str) -> list[HelperUse]:
    """All helper uses in one file. Includes are detected on the raw text (they live in <...>)."""
    uses: list[HelperUse] = []
    raw_lines = source.splitlines()
    for lineno, raw in enumerate(raw_lines, start=1):
        m = _INCLUDE_RE.match(raw)
        if m:
            uses.append(HelperUse(filename, lineno, f"#include <{m.group(1)}>", "include", raw.strip()))
    stripped = strip_comments_and_strings(source)
    for lineno, line in enumerate(stripped.splitlines(), start=1):
        if _INCLUDE_RE.match(line):
            continue
        text = raw_lines[lineno - 1].strip() if lineno - 1 < len(raw_lines) else line.strip()
        for m in _SYMBOL_RE.finditer(line):
            sym = m.group(1)
            uses.append(HelperUse(filename, lineno, f"dpct::{sym}", categorize(sym), text))
        for m in _MACRO_RE.finditer(line):
            uses.append(HelperUse(filename, lineno, m.group(1), "macro", text))
    return uses
