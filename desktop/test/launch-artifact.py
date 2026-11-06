import json,os,subprocess,sys,time
from pathlib import Path
import psutil
artifact=Path(sys.argv[1]).resolve()
log=Path(sys.argv[2]).resolve()
env=dict(os.environ);env.pop('ELECTRON_RUN_AS_NODE',None)
with log.open('w',encoding='utf-8') as stream:
    app=subprocess.Popen([str(artifact),'--appimage-extract-and-run'] if sys.platform=='linux' else [str(artifact)],stdout=stream,stderr=subprocess.STDOUT,env=env)
    backend=None
    try:
        end=time.monotonic()+60
        while time.monotonic()<end:
            if app.poll() is not None: raise RuntimeError(f'App exited {app.returncode}: '+log.read_text(errors='replace')[-3000:])
            try:
                for process in psutil.Process(app.pid).children(recursive=True):
                    if process.name().lower().startswith('spyder-cal-backend'):
                        backend=process;break
            except psutil.NoSuchProcess: pass
            if backend:
                time.sleep(8)
                assert backend.is_running() and app.poll() is None
                assert 'FATAL:' not in log.read_text(errors='replace'), 'Fatal startup error'
                print(json.dumps({'artifact':str(artifact),'gui_started':True,'bundled_backend_started':True,'backend':backend.exe()}))
                break
            time.sleep(.5)
        else: raise RuntimeError('No bundled backend process: '+log.read_text(errors='replace')[-3000:])
    finally:
        if sys.platform=='win32': subprocess.run(['taskkill','/PID',str(app.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else:
            # Let Electron exit before the AppImage wrapper removes its files.
            if backend:
                try:
                    gui = backend.parent()
                    if gui and gui.pid != app.pid:
                        gui.terminate()
                except psutil.NoSuchProcess: pass
            else: app.terminate()
            try: app.wait(timeout=10)
            except subprocess.TimeoutExpired: app.kill();app.wait()
