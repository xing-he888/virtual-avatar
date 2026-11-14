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
  // 效价 -> 色相：+1 暖橙 35°，-1 冷蓝 225°（hue 主导愉悦响应）
  return Math.round(130 - 95 * p);   // p=1 -> 35, p=-1 -> 225, p=0 -> 130(黄绿中性)
}

function applyMood(m) {
  document.documentElement.style.setProperty("--mood-hue", moodHue(m.p));
  document.documentElement.style.setProperty("--mood-glow", Math.max(0, m.a).toFixed(2));
  $("avatar").textContent = m.emoji;
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

// ---------- 消息 ----------
function addBubble(role, text, extra) {
  const row = document.createElement("div");
  row.className = "msg " + (role === "user" ? "user" : "xiaoxing");
  const av = document.createElement("div");
  av.className = "avatar";
  av.style.width = av.style.height = "34px";
  av.style.fontSize = "18px";
  av.textContent = role === "user" ? "🧑" : (extra && extra.mood ? extra.mood.emoji : "🙂");
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
  av.textContent = "🙂";
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
        if (i) i.textContent = "▶";
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
  av.textContent = "🙂";
  const col = document.createElement("div");
  const b = document.createElement("div");
  b.className = "bubble voice-bar" + (src ? "" : " dead");
  b.innerHTML = '<span class="vicon">' + (src ? "▶" : "🔇") + "</span> 语音条 " +
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
      b.querySelector(".vicon").textContent = "▶";
      _activeVoiceBar = null;
      return;
    }
    _audioEl.src = src;
    _audioEl.play().then(() => {
      if (_activeVoiceBar) _activeVoiceBar.querySelector(".vicon").textContent = "▶";
      _activeVoiceBar = b;
      b.querySelector(".vicon").textContent = "⏸";
    }).catch(() => { b.title = "浏览器拦截了播放——再点一次就出声"; });
  };
}

// ---------- 状态轮询 ----------
let lastWanderT = null;
let stateFirst = true;       // 首次快照只记基线，不弹旧游移

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
    $("stateSub").textContent =
      `记忆 ${s.n_facts} 条 / 经历 ${s.n_obs} 条` +
      (s.narrs.length ? ` · 自我叙事「${s.narrs[0].slice(0, 14)}…」` : "");
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

// ---------- 联想记忆 ----------
async function associate() {
  const input = $("input");
  const text = input.value.trim();
  if (!text) {
    // 空输入不再静默——告诉他这个按钮怎么用
    addBubble("xiaoxing", "（🧠 是联想：在输入框里写个话题——比如「老王」——"
              + "我再顺着它把相关的记忆都点亮给你看。）",
              { mood: { emoji: "🙂", p: 0, a: 0 } });
    return;
  }
  input.value = "";
  $("btnAssoc").disabled = true;
  const th = thinkingBubble();
  try {
    const r = await fetch("/api/associate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await r.json();
    th.row.remove();
    const b = addBubble("xiaoxing", "", data);
    b.textContent = data.text || "";
    const meta = b.parentElement.querySelector(".meta");
    if (meta && data.kind === "associate") {
      meta.prepend(chip("think", `联想 ρ=0.55/K=200 · 重复率 ${data.repeat_rate}`));
    }
  } catch (e) {
    th.row.remove();
    addBubble("xiaoxing", "（联想出错：" + e.message + "）", { mood: { emoji: "😔", p: 0, a: 0 } });
  }
  $("btnAssoc").disabled = false;
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
$("btnAssoc").addEventListener("click", associate);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$("btnSettings").addEventListener("click", openSettings);
$("btnClose").addEventListener("click", () => { $("mask").hidden = true; });
$("btnSave").addEventListener("click", saveSettings);
$("btnTest").addEventListener("click", testConn);
$("btnMem").addEventListener("click", openMemories);
$("btnMemClose").addEventListener("click", () => { $("memMask").hidden = true; });
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
