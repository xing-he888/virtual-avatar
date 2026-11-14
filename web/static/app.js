// 小星聊天页前端。
// 情绪呈现的三条通道（调研结论落地）：
//   1. 色相=效价、辉光=唤醒      Valdez & Mehrabian 1994（亮度/饱和度驱动唤醒感知）
//   2. 状态标签克制、不过度表达   Emerald 2024「golden zone」：过度情绪化适得其反
//   3. 打字节奏=韵律替代通道      Pezenka 2024：情绪表达提升拟人化与温暖
const $ = (id) => document.getElementById(id);
const chatEl = $("chat");
let trail = [];          // PAD 轨迹（dial 用）
let typingTimer = null;

// ---------- 工具 ----------
function moodHue(p) {
  // 效价 -> 色相（2026-11 改版）：平静=琥珀 45°，+1 金橙 30°，-1 冷蓝 215°。
  // hue 主导愉悦响应（Valdez & Mehrabian 1994）；基点从黄绿 130° 改成
  // 琥珀 45°——小星平静时整个界面是暖的，只有真的低落才冷下来。
  return Math.round(p >= 0 ? 45 - 15 * p : 45 - 170 * p);   // p=1→30, p=0→45, p=-1→215
}

// 语音条的播放状态图标（单线/实心 SVG，同页面图标体系）
const ICON_PLAY = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.6v12.8a.8.8 0 0 0 1.22.68l10-6.4a.8.8 0 0 0 0-1.36l-10-6.4A.8.8 0 0 0 8 5.6z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="7" y="5.5" width="3.4" height="13" rx="1.2"/><rect x="13.6" y="5.5" width="3.4" height="13" rx="1.2"/></svg>';
const ICON_MUTE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4.5 9.5v5h3.4L13 19V5L7.9 9.5H4.5z"/><path d="M16.5 9.8l4.2 4.4M20.7 9.8l-4.2 4.4"/></svg>';

function applyMood(m) {
  document.documentElement.style.setProperty("--mood-hue", moodHue(m.p));
  document.documentElement.style.setProperty("--mood-glow", Math.max(0, m.a).toFixed(2));
  // 图片头像模式：情绪只走辉光和「心情」章，不把头像换回 emoji
  if (!$("avatar").querySelector("img")) $("avatar").textContent = m.emoji;
  $("moodLabel").textContent = "心情：" + m.label;
  trail.push({ p: m.p, a: m.a });
  if (trail.length > 90) trail.shift();
  drawDial();
}

function drawDial() {
  const cv = $("dial"), ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#35304a";
  ctx.beginPath(); ctx.moveTo(w / 2, 4); ctx.lineTo(w / 2, h - 12);
  ctx.moveTo(4, (h - 12) / 2); ctx.lineTo(w - 4, (h - 12) / 2); ctx.stroke();
  const hue = moodHue(trail.length ? trail[trail.length - 1].p : 0);
  // 轨迹（越旧越淡）：OU 动力学看得见——被拖低、慢慢回基线
  ctx.strokeStyle = `hsla(${hue},70%,60%,.5)`;
  ctx.beginPath();
  trail.forEach((pt, i) => {
    const x = w / 2 + pt.p * (w / 2 - 6);
    const y = (h - 12) / 2 - pt.a * ((h - 16) / 2);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  if (trail.length) {
    const cur = trail[trail.length - 1];
    const x = w / 2 + cur.p * (w / 2 - 6);
    const y = (h - 12) / 2 - cur.a * ((h - 16) / 2);
    ctx.fillStyle = `hsl(${hue},80%,62%)`;
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill();
  }
}

function chip(cls, text) {
  const s = document.createElement("span");
  s.className = "chip " + cls;
  s.textContent = text;
  return s;
}

// ---------- 头像 ----------
// ui_config.avatar_img 开（用户装了图片头像）时，小星的聊天气泡也用照片；
// 用户自己的 🧑 和游移的 💭 是标记不是身份，保持 emoji。
let avatarImgOn = false;

function paintAvatarEl(av, emoji) {
  av.dataset.emoji = emoji || "🙂";
  if (avatarImgOn) {
    if (!av.querySelector("img")) {
      av.textContent = "";
      const img = document.createElement("img");
      img.src = "/avatar.jpg";
      img.alt = "小星";
      av.appendChild(img);
    }
  } else {
    av.textContent = av.dataset.emoji;
  }
}

function repaintAvatars() {
  document.querySelectorAll(".msg .avatar").forEach((av) => {
    if (av.dataset.user) return;                       // 用户的头像不动
    if (avatarImgOn) {
      if (!av.querySelector("img")) {
        av.textContent = "";
        const img = document.createElement("img");
        img.src = "/avatar.jpg";
        img.alt = "小星";
        av.appendChild(img);
      }
    } else {
      av.textContent = av.dataset.emoji || "🙂";
    }
  });
}

// ---------- 消息 ----------
function addBubble(role, text, extra) {
  const row = document.createElement("div");
  row.className = "msg " + (role === "user" ? "user" : "xiaoxing");
  const av = document.createElement("div");
  av.className = "avatar";
  av.style.width = av.style.height = "34px";
  av.style.fontSize = "18px";
  if (role === "user") {
    av.dataset.user = "1";
    av.textContent = "🧑";
  } else {
    paintAvatarEl(av, extra && extra.mood ? extra.mood.emoji : "🙂");
  }
  const col = document.createElement("div");
  if (extra && extra.image) {                        // 用户发的图片（发送时 dataURI / 历史 /img/）
    const im = document.createElement("img");
    im.className = "msg-img";
    im.src = extra.image;
    im.title = "发过的图片（缩略图留在项目 data/images/）";
    col.appendChild(im);
  }
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  col.appendChild(b);
  if (role === "xiaoxing" && extra) {
    const meta = document.createElement("div");
    meta.className = "meta";
    if (extra.mood) meta.appendChild(chip("", `P ${extra.mood.p > 0 ? "+" : ""}${extra.mood.p} · A ${extra.mood.a > 0 ? "+" : ""}${extra.mood.a}`));
    if (extra.think) meta.appendChild(chip("think", "认真想了想"));
    if (extra.stale) meta.appendChild(chip("stale", "引用的是已失效信息"));
    (extra.sources || []).slice(0, 2).forEach(
      s => meta.appendChild(chip("src", "来源：" + s)));
    if (extra.used_memory) meta.appendChild(chip("src", "用了记忆"));
    if (extra.offline) meta.appendChild(chip("offline", "离线模式"));
    if (extra.corrected) meta.appendChild(chip(extra.corrected.changed ? "ok" : "src",
        extra.corrected.changed ? "已修正 ✓" : "本来就这么记的"));
    if (extra.cited && extra.cited.length === 1) {
      // 引用了恰好一条记忆：给一个当场修正的入口（点后预填修正句）
      const fix = chip("fix", "记错了？");
      fix.title = "点这里，然后补上正确的值发出去——小星会当场改掉这条记忆";
      fix.onclick = () => {
        const i = $("input");
        i.value = "不对，是";
        i.focus();
      };
      meta.appendChild(fix);
    }
    if (extra.late) meta.appendChild(chip("think", "迟到的话——你插话时它还在想"));
    col.appendChild(meta);
  }
  row.append(av, col);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  return b;
}

// 打字节奏：字间隔 = 38ms / rate（唤醒高 -> 快），标点停顿加倍。
// 页面在后台时 setTimeout 会被浏览器节流成分钟级——那种时候直接整句落地，
// 不然发送按钮永远停在 disabled，界面锁死。
function typeOut(bubble, text, rate, done) {
  let i = 0;
  const startedAt = Date.now();
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bubble.after(cursor);
  const finish = () => {
    bubble.textContent = text;
    cursor.remove();
    done && done();
  };
  const step = () => {
    if (document.hidden || Date.now() - startedAt > 8000) return finish();
    if (i >= text.length) { cursor.remove(); done && done(); return; }
    const ch = text[i++];
    bubble.textContent += ch;
    chatEl.scrollTop = chatEl.scrollHeight;
    let delay = 38 / Math.max(rate, 0.6);
    if ("，。！？…、".includes(ch)) delay *= 4;
    typingTimer = setTimeout(step, delay);
  };
  step();
}

function thinkingBubble() {
  const row = document.createElement("div");
  row.className = "msg xiaoxing";
  const av = document.createElement("div");
  av.className = "avatar";
  av.style.width = av.style.height = "34px";
  av.style.fontSize = "18px";
  paintAvatarEl(av, "🙂");
  const col = document.createElement("div");
  const b = document.createElement("div");
  b.className = "bubble";
  b.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  col.appendChild(b);
  row.append(av, col);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { row, bubble: b };
}

// ---------- 图片发送（他会真的「看到」） ----------
let pendingImage = null;

function compressImage(file, done) {
  const rd = new FileReader();
  rd.onload = () => {
    const img = new Image();
    img.onload = () => {
      const W = 768;                                 // 和 vision/camera.py 同一口径
      const scale = Math.min(1, W / img.width);
      const cv = document.createElement("canvas");
      cv.width = Math.round(img.width * scale);
      cv.height = Math.round(img.height * scale);
      cv.getContext("2d").drawImage(img, 0, 0, cv.width, cv.height);
      done(cv.toDataURL("image/jpeg", 0.8));
    };
    img.src = rd.result;
  };
  rd.readAsDataURL(file);
}

function showPreview(uri) {
  const box = $("picPreview");
  box.querySelector("img").src = uri;
  box.hidden = false;
}

function clearPreview() {
  pendingImage = null;
  const box = $("picPreview");
  box.hidden = true;
  // 必须移除属性而不是置空串：空 src 会被 Safari 当相对 URL 解析成本页，
  // 画一个破图图标
  box.querySelector("img").removeAttribute("src");
}

// ---------- 发送 ----------
// 异步纪律：发送按钮从不锁死——小星"在想"的时候你随时插话，
// 插话立即进入下一轮；被插断的那句想完会带「迟到的话」标记回来。
async function send() {
  const input = $("input");
  const text = input.value.trim();
  if (!text && !pendingImage) return;
  const image = pendingImage;                        // 图片随消息一起走
  input.value = "";
  clearPreview();
  const userRow = addBubble("user", text, { image });
  const th = thinkingBubble();
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(image ? { text, image } : { text }),
    });
    const data = await r.json();
    th.row.remove();
    if (data.error === "llm-failed") {
      addBubble("xiaoxing", "（调用模型失败：）" + (data.message || "接口异常"),
                { mood: { emoji: "😔", p: 0, a: 0 } });
    } else if (data.voice) {
      // 微信式语音条：音频不落盘，文字进记忆
      addVoiceBubble(data.voice.seconds, data.voice.data, data.text);
      applyMood(data.mood);
    } else {
      const b = addBubble("xiaoxing", "", data);
      // 回复延迟里带一点"想想"的停顿：唤醒低时更从容（克制的表达）
      const preDelay = 220 + 260 * Math.max(0, 1 - Math.abs(data.mood.a));
      setTimeout(() => typeOut(b, data.text || data.reply, data.mood.rate, () => {
        input.focus();
      }), preDelay);
      applyMood(data.mood);
    }
  } catch (e) {
    th.row.remove();
    // 发送失败：把这条消息从界面上收回去，图片放回预览框——
    // 否则图既不在输入框也不在预览里，用户只能重新挑一次文件
    userRow.remove();
    if (image) { pendingImage = image; showPreview(image); }
    addBubble("xiaoxing", "（网络异常：" + e.message + "）", { mood: { emoji: "😔", p: 0, a: 0 } });
  }
  refreshState();
}

// ---------- 语音条（微信式，CosyVoice）----------
// 发不发语音由后端 agent 自己决定（send_voice 工具）：用户只是正常说话，
// 模型判断此刻发语音更应景时才回一条语音条。前端不提供「把上句话念出来」
// 的入口——那会和 agent 的判断打架。
// 前端只负责播放：音频由服务端合成进内存、base64 直达前端——本地零文件留存，
// 「发过语音」作为经历写进记忆（只有摘要）。
// Safari 只允许「点击手势内」出声——先播 0 字节静音 wav 解锁元素。
let _audioEl = null;
let _activeVoiceBar = null;
const _SILENT_WAV =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";

function unlockAudio() {
  if (!_audioEl) {
    _audioEl = new Audio();
    _audioEl.preload = "auto";
    _audioEl.onended = () => {
      if (_activeVoiceBar) {
        const i = _activeVoiceBar.querySelector(".vicon");
        if (i) i.innerHTML = ICON_PLAY;
        _activeVoiceBar = null;
      }
    };
  }
  _audioEl.src = _SILENT_WAV;
  _audioEl.play().catch(() => {});
}

function addVoiceBubble(seconds, src, tip) {
  const row = document.createElement("div");
  row.className = "msg xiaoxing";
  const av = document.createElement("div");
  av.className = "avatar";
  av.style.width = av.style.height = "34px";
  av.style.fontSize = "18px";
  paintAvatarEl(av, "🙂");
  const col = document.createElement("div");
  const b = document.createElement("div");
  b.className = "bubble voice-bar" + (src ? "" : " dead");
  b.innerHTML = '<span class="vicon">' + (src ? ICON_PLAY : ICON_MUTE) + "</span> 语音条 " +
    '<span class="vlen">' + seconds + '″</span>';
  if (tip) b.title = tip;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.appendChild(chip("", "语音条 · 本地不留存"));
  col.appendChild(b);
  col.appendChild(meta);
  row.append(av, col);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  if (!src) {
    b.title = "语音条不留档——这条只剩记忆里的摘要";
    return;
  }
  b.onclick = () => {
    if (!_audioEl) unlockAudio();
    if (_activeVoiceBar === b) {          // 再点 = 暂停
      _audioEl.pause();
      b.querySelector(".vicon").innerHTML = ICON_PLAY;
      _activeVoiceBar = null;
      return;
    }
    _audioEl.src = src;
    _audioEl.play().then(() => {
      if (_activeVoiceBar) _activeVoiceBar.querySelector(".vicon").innerHTML = ICON_PLAY;
      _activeVoiceBar = b;
      b.querySelector(".vicon").innerHTML = ICON_PAUSE;
    }).catch(() => { b.title = "浏览器拦截了播放——再点一次就出声"; });
  };
}

// ---------- 状态轮询 ----------
let lastWanderT = null;
let stateFirst = true;       // 首次快照只记基线，不弹旧游移

// 小星可自改的前端外观：/api/state 里的 ui 段 → 直接刷 CSS 变量。
// 白名单预设+钳制在服务端 set_ui 里做，这里 dumb-apply 零判断——
// LLM 改完头像/背景，下一拍全端生效（网页/桌面共用）。
// 主题跟随：背景键是 ink（墨黑）时按钮显示太阳（点它回白），否则显示月亮。
const ICON_SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5 5l1.6 1.6M17.4 17.4 19 19M19 5l-1.6 1.6M6.6 17.4 5 19"/></svg>';
const ICON_MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/></svg>';

let currentBg = "deepspace";     // 当前背景预设键（applyUI 维护，切换按钮读它）

function setThemeIcon(background) {
  currentBg = background || "deepspace";
  const b = $("btnTheme");
  if (!b) return;
  b.innerHTML = currentBg === "ink" ? ICON_SUN : ICON_MOON;
}

function applyUI(ui) {
  if (!ui) return;
  const av = $("avatar");
  const imgOn = !!ui.avatar_img;
  const flipped = imgOn !== avatarImgOn;
  avatarImgOn = imgOn;
  if (imgOn) {
    // 图片头像：圆框里塞 <img>（情绪辉光/色相环继续作用在框上）。
    // 一次渲染成 img 后，applyMood 的 emoji 轮询就不该再覆盖它
    if (!av.querySelector("img")) {
      const img = document.createElement("img");
      img.src = "/avatar.jpg";
      img.alt = "小星";
      av.textContent = "";
      av.appendChild(img);
    }
  } else if (ui.avatar_emoji) {
    av.textContent = ui.avatar_emoji;
  }
  if (flipped) repaintAvatars();   // 开关切换时，已有聊天气泡的头像一起刷
  const r = document.documentElement.style;
  const vars = ui.vars || {};
  for (const k in vars) r.setProperty(k, String(vars[k]));
  if (ui.avatar_hue != null) r.setProperty("--avatar-hue", String(ui.avatar_hue));
  if (ui.mood_glow != null) r.setProperty("--mood-glow", String(ui.mood_glow));
  setThemeIcon(ui.background);
}

function addWanderBubble(w) {
  const row = document.createElement("div");
  row.className = "msg xiaoxing wander";
  const av = document.createElement("div");
  av.className = "avatar";
  av.style.width = av.style.height = "34px";
  av.style.fontSize = "18px";
  av.textContent = "💭";
  const col = document.createElement("div");
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = "（刚才没人说话，我自己想起了些事）\n" +
    w.thoughts.map(t => "· " + t).join("\n");
  col.appendChild(b);
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.appendChild(chip("", "心智游移·点亮 " + w.touched + " 条"));
  col.appendChild(meta);
  row.append(av, col);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function refreshState() {
  try {
    const r = await fetch("/api/state");
    const s = await r.json();
    applyMood(s.mood);
    applyUI(s.ui);
    if (s.wander && s.wander.t) {
      if (stateFirst) {
        lastWanderT = s.wander.t;          // 载入前已发生的游移不重播
        stateFirst = false;
      } else if (s.wander.t !== lastWanderT) {
        lastWanderT = s.wander.t;
        addWanderBubble(s.wander);
      }
    } else if (stateFirst) {
      stateFirst = false;
    }
    // 顶栏副标题只报记忆/经历数（用户裁决 2026-11-14：自我叙事不上顶栏，
    // 它在 🌱 成长面板的「自我认知」栏和给小星的提示词里继续存在）
    $("stateSub").textContent = `记忆 ${s.n_facts} 条 / 经历 ${s.n_obs} 条`;
    const llm = $("llmState");
    if (s.llm.has_key) {
      llm.textContent = `模型已配置（…${s.llm.tail} · ${s.llm.model}）`;
    } else {
      llm.textContent = "未配置 API Key —— 离线模式：记忆问答可用";
    }
    const tts = $("ttsState");
    if (s.tts) {
      if (s.tts.state === "ready") {
        tts.textContent = `🔊 语音就绪（${s.tts.device}）`;
      } else if (s.tts.state === "failed") {
        tts.textContent = "🔇 语音不可用";
        tts.title = s.tts.error || "";
      } else if (!s.tts.supported) {
        tts.textContent = "🔇 未配置语音模型（见 voice/）";
      }
      // untried + supported：不写文案——语音由小星自己决定何时发，
      // 页面上没有「点一下念出来」的按钮，提示它反而误导
    }
  } catch (e) { /* 服务没起，忽略 */ }
}

// ---------- 设置弹窗 ----------
async function openSettings() {
  const r = await fetch("/api/settings");
  const s = await r.json();
  $("setKey").value = "";
  $("setKey").placeholder = s.has_key ? `已保存（尾号 ${s.tail}），留空保存=不变` : "粘贴 DashScope / 兼容接口的 Key（留空=清除）";
  $("setUrl").value = s.base_url || "";
  $("setModel").value = s.model || "";
  $("testResult").textContent = "";
  $("mask").hidden = false;
}

async function saveSettings() {
  const payload = {
    base_url: $("setUrl").value.trim(),
    model: $("setModel").value.trim(),
  };
  const key = $("setKey").value.trim();
  if (key) payload.api_key = key;
  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  $("mask").hidden = true;
  refreshState();
}

async function testConn() {
  $("testResult").textContent = "测试中…";
  $("testResult").className = "test-result";
  const r = await fetch("/api/settings/test", { method: "POST", body: "{}" });
  const s = await r.json();
  $("testResult").textContent = s.ok ? "✓ 连通（模型回了「" + (s.sample || "ok") + "」）"
                                      : "✗ " + (s.error || "失败");
  $("testResult").className = "test-result " + (s.ok ? "ok" : "bad");
}

// ---------- 记忆修正面板 ----------
const REL_LABEL = { 0: "亲眼", 1: "亲口", 2: "听说", 3: "推断", 4: "先验" };

async function openMemories() {
  $("memMask").hidden = false;
  const list = $("memList");
  let s;
  try {
    const r = await fetch("/api/memories");
    s = await r.json();
  } catch (e) {
    list.textContent = "（加载失败：" + e.message + "）";
    return;
  }
  list.innerHTML = "";
  if (!s.memories.length) {
    list.innerHTML = '<p class="hint">小星还没记住任何事实。聊几句「老王的电话是…」这样的句子试试。</p>';
    return;
  }
  for (const m of s.memories) {
    list.appendChild(memRow(m));
  }
}

function memRow(m) {
  const row = document.createElement("div");
  row.className = "mem-row" + (m.status !== "active" ? " dimmed" : "");

  const text = document.createElement("div");
  text.className = "mem-text";
  const ent = document.createElement("b");
  ent.textContent = m.entity;
  const val = document.createElement("span");
  val.className = "mem-val";
  val.textContent = m.value;
  text.append(ent, document.createTextNode("." + m.predicate + " = "), val);
  row.appendChild(text);

  const meta = document.createElement("div");
  meta.className = "mem-meta";
  meta.appendChild(chip("src", REL_LABEL[m.rel] || "来源?"));
  if (m.source) meta.appendChild(chip("", m.source));
  if (m.status !== "active") meta.appendChild(chip("stale", m.status === "superseded" ? "已被新值取代" : "已撤回"));
  row.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "mem-actions";
  const btnFix = document.createElement("button");
  btnFix.className = "ghost";
  btnFix.textContent = "改值";
  btnFix.onclick = () => editMemValue(row, val, m);
  const btnRetract = document.createElement("button");
  btnRetract.className = "ghost";
  btnRetract.textContent = "撤回";
  btnRetract.onclick = async () => {
    if (!confirm(`撤回「${m.entity}.${m.predicate} = ${m.value}」？\n从此小星不再提起这条（原话仍在经历层留底）。`)) return;
    await fetch("/api/forget", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fact_id: m.fact_id }),
    });
    openMemories();
    refreshState();
  };
  actions.append(btnFix, btnRetract);
  row.appendChild(actions);
  return row;
}

function editMemValue(row, valSpan, m) {
  const val = row.querySelector(".mem-val");
  const input = document.createElement("input");
  input.className = "mem-input";
  input.value = m.value;
  val.replaceWith(input);
  input.focus(); input.select();
  const done = async (save) => {
    if (save && input.value.trim() && input.value.trim() !== m.value) {
      await fetch("/api/correct", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fact_id: m.fact_id, value: input.value.trim() }),
      });
      openMemories();
      refreshState();
      return;
    }
    input.replaceWith(val);          // 取消：还原显示
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") done(true);
    if (e.key === "Escape") done(false);
  });
  input.addEventListener("blur", () => done(true));
}

// ---------- 绑定与启动 ----------
$("btnSend").addEventListener("click", send);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
// ---------- 成长面板：性格特质 / 信任 / 自我叙事（/api/persona，只读视图） ----------
const GROW_TRAITS = ["O", "C", "E", "A", "N"];
const GROW_TNAME = { O: "开放", C: "尽责", E: "外向", A: "宜人", N: "神经质" };
const PAD_NAME = ["效价", "唤醒", "支配"];

const f2 = (v) => (v >= 0 ? "+" : "") + Number(v).toFixed(2);

function growSec(title) {
  const sec = document.createElement("div");
  sec.className = "grow-sec";
  const h = document.createElement("h3");
  h.textContent = title;
  sec.appendChild(h);
  return sec;
}

function growBar(v) {                      // v ∈ [-1,1] → 10 格方块条
  const filled = Math.round((v + 1) / 2 * 10);
  let bar = "";
  for (let i = 0; i < 10; i++) bar += i < filled ? "▌" : "░";
  return bar;
}

async function openGrowth() {
  $("growMask").hidden = false;
  const body = $("growBody");
  let p;
  try {
    const r = await fetch("/api/persona");
    p = await r.json();
  } catch (e) {
    body.textContent = "（加载失败：" + e.message + "）";
    return;
  }
  body.innerHTML = "";

  // 特质条：出厂种子 → 现在。动过的标出来——「被经历塑造过」要看得见
  const secT = growSec("性格特质（被经历慢慢塑造）");
  const tlist = document.createElement("div");
  tlist.className = "grow-traits";
  for (const k of GROW_TRAITS) {
    const drift = p.traits[k] - p.seed[k];
    const moved = Math.abs(drift) >= 0.05;
    const row = document.createElement("div");
    row.className = "grow-trait" + (moved ? " moved" : "");
    const name = document.createElement("span");
    name.className = "grow-tname";
    name.textContent = GROW_TNAME[k];
    const bar = document.createElement("code");
    bar.textContent = growBar(p.traits[k]);
    const note = document.createElement("span");
    note.className = "grow-tdrift";
    note.textContent = moved
      ? `出厂 ${f2(p.seed[k])} → 现在 ${f2(p.traits[k])} ${drift > 0 ? "↑" : "↓"}`
      : `出厂设定 ${f2(p.seed[k])}`;
    row.append(name, bar, note);
    tlist.appendChild(row);
  }
  secT.appendChild(tlist);
  body.appendChild(secT);

  // 情绪底色：出厂基线 → 此刻基线（性格 + 昼夜节律——凌晨和傍晚不一样）
  if (p.mu_now) {
    const secM = growSec("情绪底色（性格定基调，昼夜节律微调）");
    const mlist = document.createElement("div");
    mlist.className = "grow-traits";
    p.mu_now.forEach((now, i) => {
      const row = document.createElement("div");
      row.className = "grow-trait";
      const name = document.createElement("span");
      name.className = "grow-tname";
      name.textContent = PAD_NAME[i];
      const note = document.createElement("span");
      note.className = "grow-tdrift";
      note.textContent = `出厂 ${f2(p.mu[i])} → 此刻 ${f2(now)}（此刻是说话的底色）`;
      row.append(name, note);
      mlist.appendChild(row);
    });
    secM.appendChild(mlist);
    body.appendChild(secM);
  }

  // 信任：数值条 + 关系 stance（和提示词里同一套阈值）
  const secR = growSec("对我们的关系");
  const trow = document.createElement("div");
  trow.className = "grow-trust";
  const bar = document.createElement("div");
  bar.className = "grow-bar";
  const fill = document.createElement("i");
  fill.style.width = Math.round(p.trust * 100) + "%";
  bar.appendChild(fill);
  const stance = p.trust < 0.4 ? "有点戒备" : p.trust > 0.75 ? "很亲近，可以开玩笑" : "当朋友处";
  const tlabel = document.createElement("span");
  tlabel.className = "grow-tdrift";
  tlabel.textContent = `信任 ${f2(p.trust)} · ${stance}`;
  trow.append(bar, tlabel);
  secR.appendChild(trow);
  body.appendChild(secR);

  // 成长阶段 + 距下一阶段
  const secS = growSec("成长阶段");
  const sline = document.createElement("p");
  sline.className = "grow-line";
  let s = `当前：${p.stage}`;
  if (p.next) {
    const gap = [];
    if (p.next.turns) gap.push(`${p.next.turns} 轮对话`);
    if (p.next.days) gap.push(`${p.next.days} 天`);
    if (p.next.topics) gap.push(`${p.next.topics} 个新话题`);
    if (gap.length) s += ` · 距下一阶段还差 ${gap.join("、")}`;
  }
  sline.textContent = s;
  secS.appendChild(sline);
  body.appendChild(secS);

  // 自我叙事：他自己说出口的「我变了」
  const secN = growSec("自我认知（他自己说出口的）");
  if (p.narrs && p.narrs.length) {
    const ul = document.createElement("ul");
    ul.className = "grow-narrs";
    for (const n of p.narrs) {
      const li = document.createElement("li");
      li.textContent = n;
      ul.appendChild(li);
    }
    secN.appendChild(ul);
  } else {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "还没有——特质相对出厂变化累计到 0.1 时，他会自己说出「我变了」。";
    secN.appendChild(hint);
  }
  body.appendChild(secN);

  if (p.last_shaped) {
    const ts = parseFloat(p.last_shaped);
    if (ts > 0) {
      const foot = document.createElement("p");
      foot.className = "hint";
      foot.textContent = `上次塑形：${new Date(ts * 1000).toLocaleString()}`;
      body.appendChild(foot);
    }
  }
}

$("btnSettings").addEventListener("click", openSettings);
$("btnClose").addEventListener("click", () => { $("mask").hidden = true; });
$("btnSave").addEventListener("click", saveSettings);
$("btnTest").addEventListener("click", testConn);
$("btnMem").addEventListener("click", openMemories);
$("btnMemClose").addEventListener("click", () => { $("memMask").hidden = true; });
$("btnGrowth").addEventListener("click", openGrowth);
$("btnGrowClose").addEventListener("click", () => { $("growMask").hidden = true; });
// 白底/黑底切换：简白(deepspace) ↔ 墨黑(ink)，走小星自己的 /api/ui 通道，
// 偏好落 meta 全端同步（桌面版下一拍跟随）
$("btnTheme").addEventListener("click", async () => {
  await fetch("/api/ui", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ background: currentBg === "ink" ? "deepspace" : "ink" }),
  });
});
// ---------- 入页还原聊天记录 ----------
// 记录以 obs 落盘（用户/小星 told+text），服务器重启也不丢；
// 这里把它读回来补气泡，之后才决定要不要放欢迎语。
const GREETING = "我醒着呢。跟我说的话我都记着——换号、搬家这种事，过几天再问我也答得上来。右上角⚙里配好 API Key，我就能好好说话。";

async function restoreHistory() {
  let restored = false;
  try {
    const r = await fetch("/api/history");
    const s = await r.json();
    for (const rec of s.history || []) {
      if (rec.channel === "image") {
        // 用户发过的图：缩略图存档在项目 data/images/
        addBubble("user", rec.text || "（一张图）", { image: "/img/" + rec.image });
      } else if (rec.role === "user") addBubble("user", rec.text);
      else if (rec.channel === "voice") addVoiceBubble(rec.seconds || 0, null, rec.text);
      else addBubble("xiaoxing", rec.text, rec);
      restored = true;
    }
  } catch (e) { /* 服务没起，忽略 */ }
  if (!restored) {
    addBubble("xiaoxing", GREETING, { mood: { emoji: "🙂", p: 0, a: 0 } });
  }
}

// ---------- 清空聊天窗口 ----------
// 只清视图 + 记分界时间（持久）；记忆/识海/经历一律不动。
async function clearChat() {
  if (!confirm("清空聊天窗口？\n记忆和识海不受影响——小星依然记得聊过的内容。")) return;
  try {
    await fetch("/api/clear_chat", { method: "POST", body: "{}" });
  } catch (e) { /* 服务没起也照常清视图 */ }
  chatEl.innerHTML = "";
  addBubble("xiaoxing", "（窗口已清空。聊过的内容我都记得——直接继续就行。）",
            { mood: { emoji: "🙂", p: 0, a: 0 } });
  refreshState();
}

refreshState();
setInterval(refreshState, 4000);
restoreHistory();
$("btnClear").addEventListener("click", clearChat);
$("btnPic").addEventListener("click", () => $("filePic").click());
$("filePic").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) compressImage(f, (uri) => { pendingImage = uri; showPreview(uri); });
  e.target.value = "";
});
$("picPreview").querySelector(".pic-x").addEventListener("click", clearPreview);
