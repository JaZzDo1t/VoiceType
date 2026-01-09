"""
Runtime hook for pynput to ensure proper initialization in frozen builds.

This hook runs BEFORE main.py and ensures pynput's Win32 backend is properly
loaded with all required ctypes dependencies.

The critical fix here is pre-importing ctypes and loading user32.dll before
pynput tries to set up its keyboard listener. Without this, pynput may fail
silently in PyInstaller frozen builds.
"""
import sys
import os

# Only run in frozen builds
if getattr(sys, 'frozen', False):
    _success = False
    _error = None

    try:
        print("[rthook_pynput] Initializing pynput for frozen build...", file=sys.stderr)

        # Step 1: Pre-import ctypes (required for pynput Win32 backend)
        import ctypes
        import ctypes.wintypes

        # Step 2: Pre-load user32.dll (used by pynput for keyboard hooks)
        try:
            user32 = ctypes.windll.user32
            # Verify the DLL is actually loaded by calling a benign function
            _ = user32.GetKeyboardLayout(0)
            print("[rthook_pynput] user32.dll loaded successfully", file=sys.stderr)
        except Exception as dll_err:
            print(f"[rthook_pynput] Warning: user32.dll issue: {dll_err}", file=sys.stderr)

        # Step 3: Pre-load kernel32.dll (also used by pynput)
        try:
            kernel32 = ctypes.windll.kernel32
            _ = kernel32.GetCurrentThreadId()
            print("[rthook_pynput] kernel32.dll loaded successfully", file=sys.stderr)
        except Exception as dll_err:
            print(f"[rthook_pynput] Warning: kernel32.dll issue: {dll_err}", file=sys.stderr)

        # Step 4: Import pynput base module
        import pynput
        print(f"[rthook_pynput] pynput base imported: {pynput}", file=sys.stderr)

        # Step 5: Import pynput.keyboard and its Win32 backend
        import pynput.keyboard
        print(f"[rthook_pynput] pynput.keyboard imported: {pynput.keyboard}", file=sys.stderr)

        # Step 6: Import Win32-specific backend module
        try:
            import pynput.keyboard._win32
            print(f"[rthook_pynput] pynput.keyboard._win32 imported: {pynput.keyboard._win32}", file=sys.stderr)
        except ImportError as win32_err:
            print(f"[rthook_pynput] Warning: _win32 import failed: {win32_err}", file=sys.stderr)
            # Try alternative import path
            try:
                from pynput.keyboard import _win32
                print(f"[rthook_pynput] _win32 imported via alternative path", file=sys.stderr)
            except ImportError:
                print(f"[rthook_pynput] _win32 backend not available", file=sys.stderr)

        # Step 7: Import pynput.mouse (may also be needed)
        try:
            import pynput.mouse
            import pynput.mouse._win32
            print(f"[rthook_pynput] pynput.mouse._win32 imported", file=sys.stderr)
        except ImportError:
            # Mouse is optional, keyboard is what we need
            pass

        # Step 8: Verify Listener class is available
        if hasattr(pynput.keyboard, 'Listener'):
            print(f"[rthook_pynput] Listener class available: {pynput.keyboard.Listener}", file=sys.stderr)
            _success = True
        else:
            print("[rthook_pynput] ERROR: Listener class not found!", file=sys.stderr)
            _success = False

        if _success:
            print("[rthook_pynput] pynput initialized successfully", file=sys.stderr)

    except Exception as e:
        _error = e
        import traceback
        print(f"[rthook_pynput] pynput init failed: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"[rthook_pynput] Traceback: {traceback.format_exc()}", file=sys.stderr)

        # Try minimal fallback
        try:
            if 'pynput' not in sys.modules:
                import pynput
            if 'pynput.keyboard' not in sys.modules:
                import pynput.keyboard
            _success = 'pynput.keyboard' in sys.modules
            if _success:
                print("[rthook_pynput] Fallback: pynput.keyboard loaded", file=sys.stderr)
        except Exception as fallback_err:
            print(f"[rthook_pynput] Fallback failed: {fallback_err}", file=sys.stderr)

    # Set markers for app code
    sys._pynput_rthook_success = _success
    sys._pynput_rthook_error = _error
