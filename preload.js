const { contextBridge } = require('electron');

// 暴露安全的 API 给渲染进程 (如果需要的话)
contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  isElectron: true,
});
