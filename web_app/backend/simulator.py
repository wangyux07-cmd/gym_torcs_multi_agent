from __future__ import annotations

import ctypes
import io
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .settings_store import SettingsStore

try:
    import mss
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    mss = None

try:
    import win32gui
    import win32process
    import win32api
    import win32con
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    win32gui = None
    win32process = None
    win32api = None
    win32con = None


@dataclass
class SimulatorStatus:
    running: bool = False
    window_found: bool = False
    message: str = "Simulator not launched"
    torcs_path: str = ""
    torcs_args: str = ""
    window_title: str = "TORCS"


class SimulatorManager:
    def __init__(self, settings: SettingsStore) -> None:
        self.settings = settings
        self.process: subprocess.Popen | None = None
        self.lock = threading.RLock()

    def launch(self, user_id: str = "default", force_restart: bool = False, race_config: str | None = None) -> dict[str, Any]:
        with self.lock:
            settings = self.settings.load(user_id)
            torcs_path = settings.get("torcs_path", "")
            if not torcs_path:
                raise FileNotFoundError("Set the simulator path in Garage first.")
            exe = Path(torcs_path)
            if not exe.exists():
                raise FileNotFoundError(f"Simulator path does not exist: {exe}")
            if force_restart:
                stop_torcs_processes()
                self.process = None
                time.sleep(0.8)
            if find_torcs_window(torcs_path, settings.get("window_title", "TORCS")) is not None:
                return self.status()
            if self.process is not None and self.process.poll() is None:
                return self.status()
            args_str = str(settings.get("torcs_args", "")).strip()
            args = shlex.split(args_str, posix=False) if args_str else []
            if race_config:
                args = ["-r", race_config] + args
            params = " ".join(
                f'"{a}"' if " " in str(a) else str(a) for a in args
            )
            ret = ctypes.windll.shell32.ShellExecuteW(
                None, "open", str(exe), params or None, str(exe.parent), 1
            )
            if ret <= 32:
                raise FileNotFoundError(
                    f"Could not launch TORCS (ShellExecute code {ret}). "
                    "Check the simulator path is correct."
                )
            self.process = None
            time.sleep(4.5)
            if not has_torcs_process(torcs_path):
                raise FileNotFoundError(
                    "TORCS exited immediately after launch. "
                    "Check the simulator path and race config are correct."
                )
            bring_torcs_to_front(torcs_path, settings.get("window_title", "TORCS"))
            return self.status(user_id)

    def stop(self, user_id: str = "default") -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            self.process = None
            stop_torcs_processes()
            return self.status(user_id)

    def status(self, user_id: str = "default") -> dict[str, Any]:
        settings = self.settings.load(user_id)
        torcs_path = settings.get("torcs_path", "")
        managed_running = self.process is not None and self.process.poll() is None
        external_running = has_torcs_process(torcs_path)
        running = managed_running or external_running
        window = find_torcs_window(torcs_path, settings.get("window_title", "TORCS"))
        if window is not None:
            message = "Simulator ready"
        elif managed_running:
            message = "TORCS is loading, please wait..."
        elif running:
            message = "A TORCS process exists but has no visible window. Try launching again."
        else:
            message = "Launch simulator to open TORCS."
        return SimulatorStatus(
            running=running,
            window_found=window is not None,
            message=message,
            torcs_path=settings.get("torcs_path", ""),
            torcs_args=settings.get("torcs_args", ""),
            window_title=settings.get("window_title", "TORCS"),
        ).__dict__

    def frame_jpeg(self) -> bytes:
        settings = self.settings.load("default")
        window = find_torcs_window(settings.get("torcs_path", ""), settings.get("window_title", "TORCS"))
        if window is None:
            return placeholder_jpeg("Waiting for TORCS window")
        if mss is None or win32gui is None:
            return placeholder_jpeg("Install mss and pywin32 for live view")
        rect = win32gui.GetWindowRect(window)
        left, top, right, bottom = rect
        width = max(1, right - left)
        height = max(1, bottom - top)
        monitor = {"left": left, "top": top, "width": width, "height": height}
        try:
            with mss.mss() as capture:
                shot = capture.grab(monitor)
            image = Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception:
            return placeholder_jpeg("Live view unavailable")
        return image_to_jpeg(image)


def find_torcs_window(torcs_path: str = "", title_hint: str = "TORCS") -> int | None:
    if win32gui is None or win32process is None:
        return None
    hint = title_hint.lower().strip()
    candidates: list[int] = []
    fallback_candidates: list[int] = []

    def callback(hwnd: int, _: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_path = get_process_image_path(pid)
        if is_torcs_process(process_path, torcs_path):
            candidates.append(hwnd)
            return

        title = win32gui.GetWindowText(hwnd)
        lowered = title.lower()
        if title_matches_torcs(lowered, hint):
            fallback_candidates.append(hwnd)

    win32gui.EnumWindows(callback, None)
    if candidates:
        return candidates[0]
    return fallback_candidates[0] if fallback_candidates else None


def bring_torcs_to_front(torcs_path: str = "", title_hint: str = "TORCS") -> None:
    if win32gui is None or win32con is None:
        return
    hwnd = find_torcs_window(torcs_path, title_hint)
    if hwnd is None:
        return
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass


def navigate_torcs_to_race(torcs_path: str = "", title_hint: str = "TORCS") -> None:
    """Navigate TORCS from main menu to Practice → New Race using PowerShell SendKeys."""
    script = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class TorcsNav {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hWnd, uint Msg, int wParam, int lParam);
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool f);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
}
'@

$proc = Get-Process -Name @('wtorcs','torcs') -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $proc) { Write-Host "no torcs process"; exit }

$hwnd = $proc.MainWindowHandle
if ($hwnd -eq [IntPtr]::Zero) { Write-Host "no window handle"; exit }

# AttachThreadInput lets a background process call SetForegroundWindow reliably
[TorcsNav]::ShowWindow($hwnd, 9)
$curTid = [TorcsNav]::GetCurrentThreadId()
$wndPid = [uint32]0
$wndTid = [TorcsNav]::GetWindowThreadProcessId($hwnd, [ref]$wndPid)
[TorcsNav]::AttachThreadInput($curTid, $wndTid, $true)
[TorcsNav]::SetForegroundWindow($hwnd)
Start-Sleep -Milliseconds 600
[TorcsNav]::AttachThreadInput($curTid, $wndTid, $false)
Start-Sleep -Milliseconds 300

$WM_KEYDOWN = [uint32]0x0100
$WM_KEYUP   = [uint32]0x0101
$VK_RETURN  = 0x0D

for ($i = 0; $i -lt 3; $i++) {
    [TorcsNav]::PostMessage($hwnd, $WM_KEYDOWN, $VK_RETURN, 0x001C0001)
    Start-Sleep -Milliseconds 80
    [TorcsNav]::PostMessage($hwnd, $WM_KEYUP,   $VK_RETURN, 0xC01C0001)
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
    Start-Sleep -Milliseconds 1500
}
Write-Host "navigated"
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        pass


def has_torcs_process(torcs_path: str = "") -> bool:
    if current_process_is_running(torcs_path):
        return True
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('wtorcs.exe','torcs.exe') } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return False
    return bool(result.stdout.strip())


def stop_torcs_processes() -> None:
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('wtorcs.exe','torcs.exe') } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        pass


def current_process_is_running(torcs_path: str = "") -> bool:
    return False


def get_process_image_path(pid: int) -> str:
    if win32api is None or win32process is None or win32con is None:
        return ""
    access = win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ
    handle = None
    try:
        handle = win32api.OpenProcess(access, False, pid)
        return str(win32process.GetModuleFileNameEx(handle, 0))
    except Exception:
        return ""
    finally:
        if handle is not None:
            try:
                win32api.CloseHandle(handle)
            except Exception:
                pass


def is_torcs_process(process_path: str, configured_path: str = "") -> bool:
    if not process_path:
        return False
    path = Path(process_path)
    configured = Path(configured_path) if configured_path else None
    try:
        if configured is not None and path.resolve() == configured.resolve():
            return True
    except OSError:
        pass
    return path.name.lower() in {"wtorcs.exe", "torcs.exe"}


def title_matches_torcs(lowered_title: str, hint: str) -> bool:
    title = lowered_title.strip()
    if not title:
        return False
    if hint and title == hint:
        return True
    if title in {"torcs", "the open racing car simulator"}:
        return True
    if title.startswith("torcs 1.") or title.startswith("torcs:"):
        return True
    # Initial window title is the full exe path before TORCS finishes loading
    return title.endswith("wtorcs.exe") or title.endswith("torcs.exe")


def image_to_jpeg(image: Image.Image) -> bytes:
    image = fit_to_canvas(image, (1280, 720))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=78, optimize=True)
    return output.getvalue()


def fit_to_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas_width, canvas_height = size
    image = image.convert("RGB")
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        return Image.new("RGB", size, (7, 17, 31))
    scale = min(canvas_width / source_width, canvas_height / source_height)
    target_width = max(1, int(source_width * scale))
    target_height = max(1, int(source_height * scale))
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (2, 10, 20))
    left = (canvas_width - target_width) // 2
    top = (canvas_height - target_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def placeholder_jpeg(message: str) -> bytes:
    image = Image.new("RGB", (1280, 720), (7, 17, 31))
    draw = ImageDraw.Draw(image)
    draw.rectangle((34, 34, 1246, 686), outline=(30, 167, 255), width=3)
    draw.line((120, 520, 1160, 520), fill=(72, 216, 255), width=6)
    draw.text((76, 74), "Race View", fill=(237, 247, 255))
    draw.text((76, 118), message, fill=(143, 178, 213))
    return image_to_jpeg(image)


def mjpeg_stream(manager: SimulatorManager):
    while True:
        frame = manager.frame_jpeg()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.25)
