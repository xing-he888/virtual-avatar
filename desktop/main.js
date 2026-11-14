// 小星桌面壳：Electron 主进程。
// 职责只有三件：spawn Python 后端 → 等它就绪 → 开窗加载 localhost。
// 所有业务仍在 web/server.py + dh/——桌面壳是 GUI，不是第二个后端。
// 架构模式来自 electron-python-example（fyears）的社区标准做法。
const { app, BrowserWindow, nativeTheme } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

// stdio 是管道时（被 head 截断、重定向目标关闭等），往 stdout/stderr 写日志
// 会抛 EPIPE——没有监听器就成了主进程的未捕获异常，弹窗打断用户。
// 日志写不出去不该杀 GUI 壳：挂上空监听器吞掉（实测：electron . | head 触发）。
for (const stream of [process.stdout, process.stderr]) {
  if (stream && typeof stream.on === "function") {
    stream.on("error", () => {});
  }
}

const PORT = 8790;               // 与网页版 8787 并行，互不打架
let py = null;
let win = null;

function startBackend() {
  const root = path.resolve(__dirname, "..");
  // 优先用项目 venv（依赖都在里面），退回系统 python3
  const venvPy = path.join(root, ".venv", "bin", "python");
  const pyBin = require("fs").existsSync(venvPy) ? venvPy : "python3";
  py = spawn(pyBin, ["web/server.py", String(PORT), "120"], {
    cwd: root,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },  // 不缓冲：stdio 日志实时可见
  });
  py.stdout.on("data", (d) => console.log("[py]", d.toString().trim()));
  py.stderr.on("data", (d) => console.error("[py!]", d.toString().trim()));
  py.on("exit", (code) => {
    if (!quitting) console.error("[py] 后端提前退出（code=%s）", code);
  });
}

function waitForServer(tries = 60) {
  return new Promise((resolve, reject) => {
    const probe = () => {
      http.get(`http://127.0.0.1:${PORT}/api/state`, (r) => { r.resume(); resolve(); })
        .on("error", () => {
          if (--tries <= 0) reject(new Error("后端 30 秒未就绪"));
          else setTimeout(probe, 500);
        });
    };
    probe();
  });
}

async function create() {
  nativeTheme.themeSource = "dark";
  startBackend();
  await waitForServer();
  win = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 880,
    minHeight: 620,
    backgroundColor: "#010104",
    title: "小星 · 带记忆的虚拟人",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,          // 渲染层拿不到 Node 能力——安全默认
    },
  });
  win.loadURL(`http://127.0.0.1:${PORT}/`);
  win.on("closed", () => { win = null; });
  if (process.env.XIAOXING_SMOKE) {    // 自动化冒烟：5 秒后自退
    setTimeout(() => app.quit(), 5000);
  }
}

let quitting = false;
app.on("before-quit", () => {
  quitting = true;
  try { http.get(`http://127.0.0.1:${PORT}/api/shutdown`); } catch (e) { /* 后端可能已死 */ }
  if (py) { try { py.kill(); } catch (e) { /* 同上 */ } }
});
app.whenReady().then(create);
app.on("window-all-closed", () => app.quit());
