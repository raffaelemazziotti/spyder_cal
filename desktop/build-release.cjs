// Run natively on the target OS: Python/PsychoPy cannot be cross-compiled.
// electron-builder 26.18.0 uses unpackDirName: true to omit UNPACK_DIR_NAME,
// leaving NSIS to use a unique $PLUGINSDIR per launch. Its false option is
// implemented differently from its documentation, so pin the builder version.
const { spawnSync } = require('node:child_process');
const { existsSync, writeFileSync } = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const windows = process.platform === 'win32';
if (!windows && process.platform !== 'linux') throw new Error('Build on Windows or Linux.');
const local = path.join(root, '.venv', windows ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.SPYDER_PYTHON || (existsSync(local) ? local : windows ? 'python' : 'python3');
function run(command, args) {
  const result = spawnSync(command, args, { cwd: __dirname, stdio: 'inherit', env: { ...process.env, MPLBACKEND: 'Agg', PSYCHOPYNOPACKAGES: '1' } });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}
run(python, ['-m', 'PyInstaller', '--noconfirm', '--distpath', 'build', '--workpath', 'build/pyinstaller', 'backend.spec']);
const versions = spawnSync(python, ['-m', 'pip', 'freeze'], { encoding: 'utf8' });
if (versions.status !== 0) throw new Error('Could not record runtime versions.');
writeFileSync(path.join(__dirname, 'build/spyder-cal-backend/runtime-versions.txt'), versions.stdout);
run(process.execPath, [require.resolve('electron-builder/out/cli/cli.js'), windows ? '--win' : '--linux', ...(windows ? ['portable'] : ['AppImage', 'tar.gz']), '--x64']);
