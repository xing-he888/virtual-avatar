// 渲染层唯一能拿到的东西：一个只读标记。不给 Node 能力（contextIsolation）。
const { contextBridge } = require("electron");
contextBridge.exposeInMainWorld("xiaoxingDesktop", {
  isDesktop: true,
  electron: process.versions.electron,
});
