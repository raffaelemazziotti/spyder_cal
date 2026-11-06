"""JSON-lines worker for the Electron app. Run with an ordinary Python interpreter."""

import contextlib
import importlib.util
import json
import os
import platform
import queue
import sys
import threading
import time
import traceback


class Cancelled(Exception):
    pass


class Backend:
    def __init__(self, output, input_stream=None):
        self.output = output
        self.input = input_stream if input_stream is not None else sys.stdin
        self.commands = queue.Queue()
        self.request_id = None
        self.presenter = None
        self.quitting = False

    def emit(self, event, **data):
        self.output.write(json.dumps({"event": event, "requestId": self.request_id, **data}, allow_nan=False) + "\n")
        self.output.flush()

    def read_commands(self):
        try:
            for line in self.input:
                try:
                    command = json.loads(line)
                    if not isinstance(command, dict):
                        raise ValueError("Command must be an object")
                    self.commands.put(command)
                except (ValueError, TypeError):
                    self.commands.put({"action": "invalid"})
        finally:
            self.commands.put({"action": "quit"})

    def check_cancel(self):
        if self.presenter is not None and self.presenter.event.getKeys(keyList=["escape"]):
            raise Cancelled()
        try:
            while True:
                command = self.commands.get_nowait()
                if command.get("action") in ("cancel", "quit"):
                    self.quitting = command.get("action") == "quit"
                    raise Cancelled()
                # A prompt confirmation is only accepted while its prompt is active.
        except queue.Empty:
            pass

    def wait_for_prompt(self, name, message, completed=False):
        if self.presenter is not None:
            self.presenter.instructions(message + "\n\nPress SPACE or ENTER to continue.\nPress ESC to " + ("return to results." if completed else "cancel."))
        self.emit("prompt", prompt=name, message=message)
        while True:
            if self.presenter is not None:
                keys = self.presenter.event.getKeys(keyList=["space", "return", "escape"])
                if "escape" in keys:
                    if completed:
                        break
                    raise Cancelled()
                if "space" in keys or "return" in keys:
                    break
            try:
                command = self.commands.get(timeout=0.03)
                if command.get("action") in ("cancel", "quit"):
                    self.quitting = command.get("action") == "quit"
                    if completed:
                        break
                    raise Cancelled()
                if command.get("action") == "continue" and command.get("prompt") == name:
                    break
            except queue.Empty:
                pass
        self.emit("prompt_done", prompt=name)

    def pause(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check_cancel()
            time.sleep(min(0.02, max(0, end - time.monotonic())))

    def run_session(self, command):
        self.emit("phase", phase="loading_measurement_tools", repetition=0)
        from calibration_session import SessionSettings, measure_session

        settings = SessionSettings(**command.get("settings", {}))
        if command.get("simulated"):
            raise ValueError("Only physical SpyderX measurements are supported")
        self.emit("started")
        self.emit("phase", phase="loading_psychopy", repetition=0)
        from psychopy import visual, event
        self.check_cancel()
        self.emit("phase", phase="opening_display", repetition=0)
        try:
            self.presenter = PatchWindow(visual, event, settings.screen)
            self.presenter.instructions("Connecting to the SpyderX...\n\nKeep the sensor ready.\nPress ESC to cancel.")
            from cal_lib import SpyderX
            self.emit("phase", phase="connecting_sensor", repetition=0)
            with SpyderX(libusb_path=settings.libusb_path or None) as spyder:
                self.wait_for_prompt("dark", "Close the sensor's lens cover completely, then continue.")
                self.emit("phase", phase="calibrating_sensor", repetition=0)
                self.presenter.instructions("Calibrating the sensor...\n\nKeep the lens cover closed.")
                spyder.calibrate()
                self.emit("phase", phase="dark_complete", repetition=0)
                self.check_cancel()
                self.wait_for_prompt("position", "Open the cover and place the SpyderX against the CENTER of this display.\n\nKeep brightness and display settings fixed.")

                counts = {}
                def read_patch(drive, phase, repetition):
                    key = (phase, repetition)
                    counts[key] = counts.get(key, 0) + 1
                    label = "Original response" if phase == "uncorrected" else "Corrected response"
                    self.presenter.show(drive, f"{label}  |  Run {repetition}/{settings.repetitions}  |  Level {counts[key]}/{settings.num_levels}\nKeep the sensor at the center. Press ESC to cancel.")
                    self.pause(settings.pause)
                    return spyder.get_luminance()

                result = measure_session(settings, read_patch, self.emit, self.check_cancel)
            result["presentation"] = {"patch": "full screen except fixed bottom instruction band", "instruction_band_fraction": 0.10, "sensor_position": "center", "backend": "psychopy.pyglet"}
            summary = result["uncorrected"]
            message = f"Measurement complete. You can remove the sensor.\n\nGamma for PsychoPy: {summary['gamma']:.6f}\nBlack: {summary['black_luminance']:.3f} cd/m2    White: {summary['white_luminance']:.3f} cd/m2"
            if result["corrected"]:
                message += f"\nCorrected error: {result['corrected']['linearity_rmse_percent']:.2f}% RMSE"
            message += "\n\nReturn to Spyder Cal to view both curves and save your files."
            self.wait_for_prompt("complete", message, completed=True)
        except Cancelled:
            raise
        except Exception as error:
            if self.presenter is not None:
                self.wait_for_prompt("error", "Measurement could not finish.\n\n" + str(error), completed=True)
            raise
        finally:
            if self.presenter is not None:
                self.presenter.close()
                self.presenter = None
        self.emit("result", session=result)

    def test_display(self, selected_screen):
        if type(selected_screen) is not int or not 0 <= selected_screen <= 31:
            raise ValueError("Select a valid PsychoPy screen number")
        self.emit("phase", phase="loading_psychopy", repetition=0)
        from psychopy import visual, event
        self.check_cancel()
        self.emit("phase", phase="opening_display", repetition=0)
        try:
            self.presenter = PatchWindow(visual, event, selected_screen)
            self.presenter.instructions(
                f"PsychoPy screen {selected_screen}\n\nDisplay test successful.\nThis window will close in 5 seconds.\n\nPress ESC to close sooner."
            )
            self.emit("phase", phase="display_test_visible", repetition=0)
            self.pause(5)
        finally:
            if self.presenter is not None:
                self.presenter.close()
                self.presenter = None
        self.emit("display_tested", screen=selected_screen)

    def serve(self):
        threading.Thread(target=self.read_commands, daemon=True).start()
        self.emit(
            "ready", python=sys.executable, version=platform.python_version(),
            system=platform.system(), psychopy=importlib.util.find_spec("psychopy") is not None,
        )
        while True:
            command = self.commands.get()
            self.request_id = command.get("requestId")
            action = command.get("action")
            if action == "quit":
                return
            try:
                if action == "probe":
                    from cal_lib import SpyderX
                    with SpyderX(libusb_path=command.get("libusb_path") or None):
                        self.emit("device", connected=True, name="Datacolor SpyderX", usb="085c:0a00")
                elif action == "start":
                    self.run_session(command)
                    if self.quitting:
                        return
                elif action == "test_display":
                    self.test_display(command.get("screen", 0))
                elif action == "export":
                    from calibration_session import export_session
                    paths = export_session(command["session"], command["path"])
                    self.emit("exported", paths=paths)
                elif action == "validate":
                    from calibration_session import validate_session
                    self.emit("validated", session=validate_session(command["session"]))
                elif action not in ("cancel", "continue"):
                    raise ValueError("Unknown command")
            except Cancelled:
                self.emit("cancelled")
                if self.quitting:
                    return
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                self.emit("error", message=str(error), action=action)


class PatchWindow:
    """PsychoPy presents all measurement patches; Electron presents the controls."""

    def __init__(self, visual, event, screen):
        from pyglet import canvas
        screens = canvas.get_display().get_screens()
        if screen >= len(screens):
            raise ValueError(f"PsychoPy screen {screen} is unavailable. Choose a screen from 0 to {len(screens) - 1}.")
        self.event = event
        self.visual = visual
        self.win = visual.Window(
            fullscr=True, screen=screen, allowGUI=False, waitBlanking=True,
            units="norm", color=[-1, -1, -1], gamma=None, winType="pyglet",
            size=(screens[screen].width, screens[screen].height),
        )
        try:
            self.rect = visual.Rect(self.win, width=2, height=2, lineColor=None)
            self.band = visual.Rect(self.win, width=2, height=0.2, pos=(0, -0.9), fillColor=[-0.85]*3, lineColor=None)
            self.status = visual.TextStim(self.win, color=[0.7]*3, height=0.031, wrapWidth=1.9, pos=(0, -0.9))
            self.win.winHandle.set_visible(True)
            self.win.winHandle.activate()
            event.clearEvents()
        except Exception:
            self.win.close()
            raise

    def instructions(self, message=None):
        self.event.clearEvents()
        self.show(0)
        text = self.visual.TextStim(
            self.win, text=message or "Open the sensor cover and place the SpyderX against this display.\n\nPress SPACE to start.\nPress ESC to cancel at any time.",
            color="white", height=0.055, wrapWidth=1.5,
        )
        text.draw()
        self.win.flip()

    def show(self, drive, status=None):
        value = 2 * drive - 1
        self.rect.fillColor = [value, value, value]
        self.rect.draw()
        if status:
            self.band.draw()
            self.status.text = status
            self.status.draw()
        self.win.flip()

    def close(self):
        self.win.close()


def isolate_protocol_input():
    """Keep the JSON pipe private; imported libraries/subprocesses get EOF.

    On Windows, PsychoPy's import-time Git version probe can hang if Git
    inherits the pipe which our command-reader thread is also reading.
    Updating both the CRT descriptor and Win32 standard handle prevents
    third-party processes from inheriting that live command channel.
    """
    protocol = os.fdopen(os.dup(sys.stdin.fileno()), "r", encoding="utf-8")
    null_input = open(os.devnull, "r", encoding="utf-8")
    os.dup2(null_input.fileno(), 0)
    sys.stdin = null_input
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
        kernel32.SetStdHandle.restype = wintypes.BOOL
        if not kernel32.SetStdHandle(-10, msvcrt.get_osfhandle(null_input.fileno())):
            raise ctypes.WinError(ctypes.get_last_error())
    return protocol


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    backend = Backend(sys.stdout, isolate_protocol_input())
    # Existing library/PsychoPy logging must never corrupt the JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            backend.serve()
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            backend.emit("fatal", message=str(error))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
