"""Parse ``sycl-ls`` device listings."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# New style: [opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) ... OpenCL 3.0 (Build 0) [2026...]
_NEW_RE = re.compile(r"^\[(?P<backend>\w+):(?P<type>\w+)\]\[(?P=backend):(?P<index>\d+)\]\s*(?P<desc>.*)$")
# Old style: [opencl:cpu:0] Intel(R) OpenCL, ...
_OLD_RE = re.compile(r"^\[(?P<backend>\w+):(?P<type>\w+):(?P<index>\d+)\]\s*(?P<desc>.*)$")
_VERSION_TAIL_RES = (
    re.compile(r"\s+(OpenCL|OpenCL C|CUDA|HIP)\s+\d+\.\d+.*$"),  # "... OpenCL 3.0 (Build 0) [x]"
    re.compile(r"\s+\d+\.\d+(\.\d+)*\s+\[[^\]]*\]$"),  # level_zero: "... 1.3 [1.3.26516]"
)


@dataclass
class SyclDevice:
    backend: str  # opencl | level_zero | cuda | hip | native_cpu
    device_type: str  # cpu | gpu | fpga | acc
    index: int
    platform: str
    name: str
    description: str

    @property
    def selector(self) -> str:
        """Value for ONEAPI_DEVICE_SELECTOR that picks exactly this device."""
        return f"{self.backend}:{self.index}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["selector"] = self.selector
        return d


def parse_sycl_ls(text: str) -> list[SyclDevice]:
    devices: list[SyclDevice] = []
    for raw in text.splitlines():
        line = raw.strip()
        m = _NEW_RE.match(line) or _OLD_RE.match(line)
        if not m:
            continue
        desc = m.group("desc").strip()
        platform, _, rest = desc.partition(", ")
        name = rest or desc
        trimmed = name
        for pat in _VERSION_TAIL_RES:
            trimmed = pat.sub("", trimmed)
        trimmed = trimmed.strip()
        devices.append(
            SyclDevice(
                backend=m.group("backend"),
                device_type=m.group("type"),
                index=int(m.group("index")),
                platform=platform.strip(),
                name=trimmed or name,
                description=desc,
            )
        )
    return devices
