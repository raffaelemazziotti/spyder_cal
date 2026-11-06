const { contextBridge, ipcRenderer } = require('electron');
const invoke = async (channel, ...args) => {
  const result = await ipcRenderer.invoke(channel, ...args);
  if (result?.__spyderError) throw new Error(result.__spyderError);
  return result;
};

contextBridge.exposeInMainWorld('spyder', {
  init: () => invoke('app:init'),
  choosePython: () => invoke('python:choose'),
  probe: libusbPath => invoke('device:probe', libusbPath),
  testDisplay: selectedScreen => invoke('display:test', selectedScreen),
  start: settings => invoke('session:start', settings),
  cancel: () => invoke('session:cancel'),
  export: id => invoke('session:export', id),
  open: () => invoke('session:open'),
  copyValue: (id, metric) => invoke('value:copy', id, metric),
  onEvent: callback => {
    const listener = (_event, message) => callback(message);
    ipcRenderer.on('backend:event', listener);
    return () => ipcRenderer.removeListener('backend:event', listener);
  },
});
