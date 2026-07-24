"""Delivery channels for the daily signal: Outlook e-mail and a native popup.

Both are optional and self-contained:

* ``send_outlook`` drives a locally-installed Outlook via COM (``win32com`` from
  the ``pywin32`` package). It sends the mail from — and, by default, to — the
  logged-in Outlook account, so no SMTP server, password, or app-password is
  needed. This is the win32/Outlook path the task asked for.
* ``show_popup`` uses ``ctypes`` + the Win32 ``MessageBoxW`` API — pure standard
  library, no dependencies at all. Good when Outlook is closed or you just want a
  glance-and-dismiss reminder.

Neither raises on a non-Windows box or a missing dependency; they return a
``(ok, message)`` tuple so the orchestrator can log the outcome and fall back.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def send_outlook(subject: str, body: str, to: str = "", attachment: Path | None = None):
    """Send `body` as a plain-text Outlook e-mail. Returns (ok, message)."""
    if not _is_windows():
        return False, "Outlook unavailable: not running on Windows"
    try:
        import win32com.client  # type: ignore
    except Exception:
        return False, "Outlook unavailable: pywin32 not installed (pip install pywin32)"

    try:
        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)  # 0 = olMailItem
        # Blank recipient -> send to self (the account's own SMTP address).
        if to:
            mail.To = to
        else:
            ns = outlook.GetNamespace("MAPI")
            mail.To = ns.CurrentUser.Address
        mail.Subject = subject
        # Monospace body so the ASCII sheet lines up; keep a plain-text copy too.
        mail.BodyFormat = 2  # olFormatHTML
        mail.HTMLBody = (
            "<pre style='font-family:Consolas,Menlo,monospace;font-size:13px'>"
            + _html_escape(body)
            + "</pre>"
        )
        if attachment is not None and Path(attachment).exists():
            mail.Attachments.Add(str(Path(attachment).resolve()))
        mail.Send()
        return True, f"Outlook mail sent ({mail.To})"
    except Exception as exc:
        return False, f"Outlook send failed: {exc!r}"


def show_popup(title: str, body: str):
    """Show a native Windows message box. Returns (ok, message)."""
    if not _is_windows():
        return False, "popup unavailable: not running on Windows"
    try:
        import ctypes

        # MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST
        flags = 0x0 | 0x40 | 0x10000 | 0x40000
        ctypes.windll.user32.MessageBoxW(0, body, title, flags)
        return True, "popup shown"
    except Exception as exc:
        return False, f"popup failed: {exc!r}"


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
