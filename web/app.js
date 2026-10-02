/* 金銀短線訊號 — client */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };

  let S = null;              // last state
  let sym = store.get("sym", "XAUUSD");
  let tf = store.get("tf", "15m");
  let side = null;           // checklist side
  let lastBarTime = null;
  let lastAlertId = store.get("lastAlertId", 0);
  let firstLoad = true;
  let audioCtx = null;
  let lastStampKey = null;
  let settingsBuilt = false;
  let DEC = 2;               // price decimals of the selected instrument

  // ------------------------------------------------------------ format
  const fmt = (x, d = DEC) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const sgn = (x, d = 2) => (x === null || x === undefined) ? "—" : (x > 0 ? "+" : "") + fmt(x, d);
  const tLocal = (iso, opt) => iso ? new Date(iso).toLocaleString("zh-TW", opt || { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }) : "—";
  const tHM = (iso) => iso ? new Date(iso).toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit", hour12: false }) : "—";
  const dirCls = (d) => d > 0 ? "up-c" : (d < 0 ? "down-c" : "");
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const STATE_WORD = { flat: "觀望", long: "持多", short: "持空", signal_long: "做多訊號", signal_short: "做空訊號" };

  // ------------------------------------------------------------ api
  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`${path} ${r.status}`);
    return r.json();
  }
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  // ------------------------------------------------------------ chart
  let chart, sCandle, sE20, sE50, sHi, sLo, priceLines = [];
  const shift = (t) => t - new Date(t * 1000).getTimezoneOffset() * 60; // show local time on the axis

  function chartColors() {
    return { bg: css("--panel"), text: css("--muted"), grid: css("--rule-soft"), up: css("--up"), down: css("--down"),
      gold: css("--gold"), blue: "#5b7bb5", neutral: css("--neutral") };
  }
  function initChart() {
    const c = chartColors();
    chart = LightweightCharts.createChart($("#chart"), {
      autoSize: true,
      layout: { background: { type: "solid", color: c.bg }, textColor: c.text, fontFamily: "IBM Plex Sans, Noto Sans TC, sans-serif" },
      grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
      rightPriceScale: { borderColor: c.grid },
      timeScale: { borderColor: c.grid, timeVisible: true, secondsVisible: false, rightOffset: 6 },
      crosshair: { mode: 0 },
      localization: { locale: "zh-TW", priceFormatter: (p) => fmt(p, DEC) },
    });
    sCandle = chart.addCandlestickSeries({});
    const line = (w, style) => chart.addLineSeries({ lineWidth: w, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    sE20 = line(2, 0); sE50 = line(2, 0); sHi = line(1, 1); sLo = line(1, 1);
    applyChartColors();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyChartColors);
  }
  function applyChartColors() {
    if (!chart) return;
    const c = chartColors();
    chart.applyOptions({ layout: { background: { type: "solid", color: c.bg }, textColor: c.text },
      grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } } });
    sCandle.applyOptions({ upColor: c.up, downColor: c.down, borderUpColor: c.up, borderDownColor: c.down, wickUpColor: c.up, wickDownColor: c.down,
      priceFormat: { type: "price", precision: DEC, minMove: Math.pow(10, -DEC) } });
    sE20.applyOptions({ color: c.gold }); sE50.applyOptions({ color: c.blue });
    sHi.applyOptions({ color: c.neutral }); sLo.applyOptions({ color: c.neutral });
  }
  let chartFitted = {};
  async function loadCandles() {
    try {
      const want = `${sym}|${tf}`;
      const d = await api(`/api/candles?sym=${sym}&tf=${tf}&limit=${tf === "15m" ? 700 : 500}`);
      if (want !== `${sym}|${tf}` || !d.bars || !d.bars.length) return;
      if (d.decimals !== undefined && d.decimals !== DEC) { DEC = d.decimals; applyChartColors(); }
      const m = (arr) => (arr || []).map((p) => ({ time: shift(p.time), value: p.value }));
      sCandle.setData(d.bars.map((b) => ({ ...b, time: shift(b.time) })));
      sE20.setData(m(d.ema20)); sE50.setData(m(d.ema50));
      sHi.setData(tf === "15m" ? m(d.chan_hi) : []); sLo.setData(tf === "15m" ? m(d.chan_lo) : []);
      const c = chartColors();
      sCandle.setMarkers((d.markers || []).map((x) => ({ ...x, time: shift(x.time), color: x.shape === "circle" ? c.neutral : (x.position === "belowBar" ? c.up : c.down) })));
      priceLines.forEach((pl) => sCandle.removePriceLine(pl));
      priceLines = (d.lines || []).map((l) => sCandle.createPriceLine({ price: l.price, lineWidth: 2, lineStyle: 2, axisLabelVisible: true, title: l.title,
        color: l.kind === "stop" ? (l.dir > 0 ? c.down : c.up) : c.gold }));
      const key = `${sym}|${tf}`;
      if (!chartFitted[key]) {
        chart.timeScale().setVisibleLogicalRange({ from: d.bars.length - (tf === "15m" ? 220 : 160), to: d.bars.length + 5 });
        chartFitted[key] = true;
      }
    } catch (e) { /* keep last chart */ }
  }
  function setTf(next) {
    tf = next; store.set("tf", tf);
    $$(".tf-tabs .tab").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tf === tf)));
    $(".legend .chan").hidden = tf !== "15m";
    loadCandles();
  }
  function setSym(next) {
    if (next === sym) return;
    sym = next; store.set("sym", sym);
    lastBarTime = null; side = null; lastStampKey = null;
    $("#mypos-form").reset();
    refresh().then(() => { loadCandles(); renderBacktest(); });
  }

  // ------------------------------------------------------------ header
  function renderTabs(s) {
    const tabs = s.instruments || [];
    $("#inst-tabs").innerHTML = tabs.map((t) => {
      const st = t.error ? "資料錯誤" : (STATE_WORD[t.state] || "載入中");
      const hot = t.state && t.state !== "flat";
      return `<button class="inst-tab" type="button" role="tab" data-key="${t.key}" aria-selected="${t.key === sym}">
        <span class="seal" aria-hidden="true">${esc(t.seal)}</span><span class="nm">${esc(t.name)} ${t.bid != null ? fmt(t.bid, t.decimals) : ""}</span>
        <span class="st">${hot ? `<b class="${t.state.includes("long") ? "up-c" : "down-c"}">${st}</b>` : st}</span></button>`;
    }).join("");
    $$(".inst-tab").forEach((b) => b.addEventListener("click", () => setSym(b.dataset.key)));
  }
  function renderHeader(s) {
    const p = s.price, inst = s.instrument || {};
    DEC = inst.decimals ?? DEC;
    document.documentElement.dataset.conv = (s.settings && s.settings.color_convention) || "tw";
    $("#brand-mark").textContent = inst.seal || "金";
    $("#sym").textContent = inst.symbol || "";
    $("#px").textContent = p ? fmt(p.bid) : "—";
    $("#spread").textContent = p && p.spread != null ? fmt(p.spread, DEC + 1) : "—";
    const dc = s.day_change, chg = $("#chg");
    if (dc) {
      chg.textContent = `${sgn(dc.abs, DEC)}（${sgn(dc.pct)}%）`;
      chg.className = "chg " + (dc.abs > 0 ? "up-c" : dc.abs < 0 ? "down-c" : "");
      $("#px").className = "px " + (dc.abs > 0 ? "up-c" : dc.abs < 0 ? "down-c" : "");
    } else { chg.textContent = ""; $("#px").className = "px"; }
    $("#session").textContent = s.market ? s.market.session : "—";
    let txt = "連線中", cls = "";
    if (s.status && s.status.error) { txt = "資料錯誤"; cls = "bad"; }
    else if (s.mode === "replay") { txt = "回放"; cls = "warn"; }
    else if (p && p.time) {
      const age = (Date.now() - new Date(p.time).getTime()) / 1000;
      if (!s.market.open) { txt = "休市"; cls = "warn"; }
      else if (age < 30) { txt = `${Math.max(0, Math.round(age))} 秒前`; cls = "ok"; }
      else { txt = `${Math.round(age / 60)} 分鐘前`; cls = age < 300 ? "warn" : "bad"; }
    }
    $("#fresh .dot").className = "dot " + cls; $("#fresh-text").textContent = txt;
    document.title = p ? `${inst.name} ${fmt(p.bid)}｜金銀短線訊號` : "金銀短線訊號";
    $("#news-h").textContent = `${inst.name || ""}相關新聞`;

    const banner = $("#banner"), msgs = [];
    if (s.status && s.status.error) msgs.push(`⚠ ${s.status.error}`);
    if (s.progress && s.progress.total > 1 && s.progress.done < s.progress.total && !s.bar_time)
      msgs.push(`首次啟動：正在下載${inst.name || ""}歷史資料 ${s.progress.done}/${s.progress.total}（之後會快取，約需 1 分鐘）`);
    if (s.calendar && s.calendar.blackout_now) msgs.push(`📅 數據公布禁區：${s.calendar.blackout_title}——前後這段時間不會產生新訊號`);
    banner.hidden = !msgs.length;
    banner.className = "banner" + (s.status && s.status.error ? " bad" : "");
    banner.textContent = msgs.join("　");

    $("#replaybar").hidden = s.mode !== "replay";
    if (s.replay) {
      $("#replay-time").textContent = `目前回放到 ${tLocal(s.replay.cursor_time, { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false })}`;
      $("#replay-pause").textContent = s.replay.paused ? "繼續" : "暫停";
    }
  }

  // ------------------------------------------------------------ ticket
  function field(k, v, extra = "", copy = false, cls = "") {
    return `<div class="field${cls ? " " + cls : ""}"><span class="k">${k}</span><span class="v${copy ? " copy" : ""}" ${copy ? `data-copy="${esc(String(v)).replace(/,/g, "")}" title="點一下複製"` : ""}>${v}${extra ? `<small>${extra}</small>` : ""}</span></div>`;
  }
  function setStamp(text, cls, key) {
    const st = $("#stamp");
    $("#stamp-text").textContent = text;
    st.className = "stamp " + cls;
    st.setAttribute("aria-label", text);
    if (key && key !== lastStampKey && lastStampKey !== null && (cls === "up" || cls === "down")) { void st.offsetWidth; st.classList.add("press"); }
    lastStampKey = key;
  }
  const biasWord = (b) => b > 0 ? "偏多" : b < 0 ? "偏空" : "盤整";
  function lotsInfo(sz) {
    if (!sz || sz.lots === undefined) return ["—", ""];
    if (sz.skip) return ["建議跳過", sz.loss_per_unit ? `最小 ${fmt(sz.min_lot, 2)} 手碰停損就虧 $${fmt(sz.loss_per_unit, 0)}（本金 ${fmt(sz.min_lot_risk_pct, 1)}%），超過設定風險 2 倍` : ""];
    let ex = `碰停損約虧 $${fmt(sz.risk_usd, 0)}（本金 ${fmt(sz.risk_pct_actual, 1)}%）・保證金 $${fmt(sz.margin, 0)}`;
    if (sz.over_budget) ex += `；最小手數已略高於你設定的 ${fmt(sz.risk_pct, 1)}%`;
    if (!sz.confirmed) ex += `；以 1 手 = ${fmt(sz.oz_per_lot, 0)} 盎司計算`;
    return [`${fmt(sz.lots, 2)} 手`, ex];
  }
  const notYet = (k) => (k > 0 ? `獲利未達 ${sgn(k, 2)}R` : "還在虧損");
  function exitPlanText(s, sz) {
    const day = s.day || {};
    let t = (sz && sz.scale_out ? "其餘部位一起平" : "全部一起平") + "：碰到停損或移動停損";
    if (day.on) t += `，或美東 ${day.eod_time} 檢查時${notYet(day.keep_r)}`;
    return t;
  }
  function eodField(s, d) {
    const e = s.eod && s.eod.system;
    if (!e) return "";
    const soon = s.eod.minutes <= 600 ? `（還有 ${s.eod.minutes} 分鐘）` : "";
    if (e.keep) return field(`美東 ${s.eod.time_ny} 收盤前檢查${soon}`, "可以留過夜", e.locked ? "停損已過成本價，最壞大約保本" : `目前 ${sgn(e.R_now, 2)}R，已達 ${sgn(e.keep_r, 2)}R`, false, "wide");
    return field(`美東 ${s.eod.time_ny} 收盤前檢查${soon}`, `<span class="warn-c">屆時${notYet(e.keep_r)}就平倉</span>`,
      `目前 ${sgn(e.R_now, 2)}R；${d > 0 ? "漲到" : "跌到"} ${fmt(e.target_price)} 才可以留過夜`, false, "wide");
  }

  function renderTicket(s) {
    const F = $("#ticket-fields"), N = $("#ticket-note"), H = $("#verdict"), Sub = $("#verdict-sub");
    const sz = s.sizing || {}, inst = s.instrument || {};
    if (!s.bar_time) {
      const err = s.status && s.status.error;
      setStamp("…", "neutral", "loading");
      H.textContent = err ? "暫時拿不到行情" : "準備中";
      Sub.textContent = err ? "網路恢復後會自動繼續，不需要重開程式。" : "正在下載並計算資料（第一次約需 1 分鐘）";
      F.innerHTML = ""; N.innerHTML = ""; return;
    }
    const [lotsTxt, lotsExtra] = lotsInfo(sz);
    const notes = [];
    if (s.calendar && s.calendar.next_high && s.calendar.next_high.minutes <= 180)
      notes.push(`<p class="warn">📅 ${s.calendar.next_high.minutes} 分鐘後公布 ${esc(s.calendar.next_high.title)}（高影響）</p>`);
    const closed = s.market && !s.market.open && s.mode !== "replay";
    const st = s.state;
    if (st === "signal_long" || st === "signal_short") {
      const ns = s.new_signal, d = ns.dir;
      setStamp(d > 0 ? "做多" : "做空", d > 0 ? "up" : "down", "sig" + ns.time);
      H.textContent = `${inst.name}${d > 0 ? "做多" : "做空"}訊號：可以進場`;
      const until = new Date(new Date(ns.time).getTime() + ns.valid_minutes * 60000).toISOString();
      Sub.textContent = `訊號於 ${tHM(ns.time)} 收盤成立，請在 ${tHM(until)} 前決定`;
      F.innerHTML = field("方向", `<span class="${dirCls(d)}">${d > 0 ? "買進（做多）" : "賣出（做空）"}</span>`)
        + field("進場參考價", fmt(ns.ref_price), "", true)
        + field("停損價（務必設定）", fmt(ns.stop), "", true)
        + field("建議手數", lotsTxt, lotsExtra)
        + field("最晚進場價", fmt(ns.max_chase), d > 0 ? "高於此價就不要追" : "低於此價就不要追", true)
        + (sz.scale_out ? field("分批停利", `${fmt(sz.scale_out.lots, 2)} 手 @ ${fmt(sz.scale_out.price)}`, "+2R 先平約 1/3，其餘用移動停損", true)
                        : field("停利", "不設固定停利", "用移動停損出場"));
      if (sz.skip) notes.push(`<p class="warn">以你目前的本金，這筆連最小手數的風險都太高，建議跳過。</p>`);
      else if (sz.scale_out) notes.push(`<p>在 Mitrade 分兩張單：${fmt(sz.scale_out.lots, 2)} 手停利設 <b>${fmt(sz.scale_out.price)}</b>、另外 ${fmt(sz.lots - sz.scale_out.lots, 2)} 手不設停利；兩張停損都設 <b>${fmt(ns.stop)}</b>。</p>`);
      else notes.push(`<p>在 Mitrade：${d > 0 ? "買入" : "賣出"} ${lotsTxt}，停損設 <b>${fmt(ns.stop)}</b>，不設停利。之後每 1–2 小時回來看「目前停損」，只往有利方向移動。</p>`);
      notes.push(`<p>出場：${exitPlanText(s, sz)}。</p>`);
    } else if (st === "long" || st === "short") {
      const a = s.active, d = a.dir;
      setStamp(d > 0 ? "持多" : "持空", d > 0 ? "up" : "down", "pos" + a.entry_time);
      H.textContent = `系統持有${inst.name}${d > 0 ? "多單" : "空單"}`;
      Sub.textContent = `${tLocal(a.entry_time)} 進場，已持有 ${a.hours_held} 小時`;
      F.innerHTML = field("目前停損（移動）", fmt(a.stop), "", true)
        + field("進場價", fmt(a.entry))
        + field("浮動盈虧", `<span class="${a.R_now > 0 ? (d > 0 ? "up-c" : "down-c") : ""}">${sgn(a.R_now, 2)} R</span>`, a.mark ? `現價 ${fmt(a.mark)}` : "")
        + field("最佳曾到", `${sgn(a.best_R, 2)} R`)
        + field("初始停損", fmt(a.init_stop))
        + field("1R =", `$${fmt(a.risk)}`, "每盎司價格距離")
        + eodField(s, d);
      notes.push(`<p>把 Mitrade 的停損改成 <b>${fmt(a.stop)}</b>。停損只會往有利方向移動，碰到就出場——不要把停損拉遠。</p>`);
      const e = s.eod && s.eod.system;
      if (e && !e.keep && s.eod.minutes <= 60) notes.push(`<p class="warn">⏰ ${s.eod.minutes} 分鐘後（美東 ${s.eod.time_ny}）收盤前檢查：目前 ${sgn(e.R_now, 2)}R，屆時${notYet(e.keep_r)}就請平倉。</p>`);
    } else if (closed) {
      setStamp("休市", "neutral", "closed");
      H.textContent = "市場休市中";
      Sub.textContent = "週日 18:00（紐約時間）開盤；平日 17:00–18:00 為每日休息。";
      F.innerHTML = "";
    } else {
      const bias = s.bias, sideKey = bias > 0 ? "long" : bias < 0 ? "short" : null;
      const setup = sideKey ? s.setup[sideKey] : null;
      if (s.cooldown_until) {
        setStamp("冷卻", "neutral", "cool");
        H.textContent = "剛出場，冷卻中"; Sub.textContent = `${tHM(s.cooldown_until)} 之前不出新訊號`;
      } else if (setup && setup.ready) {
        setStamp(sideKey === "long" ? "等多" : "等空", sideKey === "long" ? "up-outline" : "down-outline", "ready" + sideKey);
        H.textContent = sideKey === "long" ? "趨勢一致向上，等待突破" : "趨勢一致向下，等待跌破";
        Sub.textContent = `15 分 K 收盤${sideKey === "long" ? "站上" : "跌破"} ${fmt(setup.trigger)} 才進場（還差 ${fmt(Math.abs(setup.distance))}）`;
      } else {
        setStamp("觀望", "neutral", "wait");
        H.textContent = "現在不要進場";
        const failing = setup ? setup.checks.filter((c) => !c.ok && !["break", "close"].includes(c.key)) : [];
        const day = s.day || {};
        if (bias !== 0 && failing.length && failing[0].key === "timing")
          Sub.textContent = day.on ? `美東 ${day.cutoff} 之後到收盤不開新倉（當日平倉模式；週五午後也不開）。` : "週五午後不開新倉。";
        else Sub.textContent = bias === 0 ? "日線沒有明確方向，系統不交易盤整。" : `日線${biasWord(bias)}，但「${failing[0] ? failing[0].label : "條件"}」還沒成立。`;
      }
      F.innerHTML = field("日線方向", `<span class="${dirCls(bias)}">${biasWord(bias)}</span>`)
        + field(sideKey === "short" ? "跌破價（24小時低點）" : "突破價（24小時高點）", setup ? fmt(setup.trigger) : "—", setup ? `距最新收盤 ${fmt(Math.abs(setup.distance))}` : "")
        + field("若觸發：停損距離", sz.stop_dist ? `$${fmt(sz.stop_dist)}` : "—", "2 × 1小時ATR")
        + field("若觸發：建議手數", lotsTxt, lotsExtra);
      notes.push(`<p>系統只順著日線方向交易；條件到齊後，等 15 分 K 收盤突破才出訊號，並會跳出通知。</p>`);
    }
    if (inst.note) notes.push(`<p class="warn">${esc(inst.note)}</p>`);
    const nh = s.news && s.news.aggregate;
    if (nh && s.active && ((s.active.dir > 0 && nh.score < -0.35) || (s.active.dir < 0 && nh.score > 0.35)))
      notes.push(`<p class="warn">⚠ 近 12 小時新聞情緒${nh.label}，與持倉方向相反，留意波動。</p>`);
    if (s.my_position) {
      const m = s.my_position, e = s.eod && s.eod.mine, bal = (s.sizing && s.sizing.balance) || 0;
      let h = `<p>你的${m.dir > 0 ? "多" : "空"}單${m.lots ? ` ${fmt(m.lots, 2)} 手` : ""} @ ${fmt(m.entry)}：建議停損 <b>${fmt(m.stop)}</b>，目前 ${sgn(m.R_now, 2)} R${m.pnl_usd != null && m.lots ? `（約 ${sgn(m.pnl_usd, 0)} 美元）` : ""}</p>`;
      if (m.lots) h += m.risk_now_usd > 0 ? `<p>打到停損約虧 $${fmt(m.risk_now_usd, 0)}${bal ? `（本金 ${fmt(100 * m.risk_now_usd / bal, 1)}%）` : ""}</p>` : `<p>停損已鎖住約 $${fmt(m.locked_usd, 0)} 獲利</p>`;
      if (e) h += `<p class="${e.keep ? "" : "warn"}">美東 ${s.eod.time_ny} 檢查：${e.keep ? "可以留過夜" : `屆時${notYet(e.keep_r)}就平倉（要${m.dir > 0 ? "漲" : "跌"}到 ${fmt(e.target_price)} 以上才留）`}</p>`;
      $("#mypos-view").innerHTML = h;
    } else { $("#mypos-view").innerHTML = ""; }
    N.innerHTML = notes.join("");
  }

  // ------------------------------------------------------------ checks & context
  function renderChecks(s) {
    if (!s.setup) { $("#checks").innerHTML = ""; return; }
    if (!side) side = s.bias < 0 ? "short" : "long";
    if (s.active) side = s.active.dir > 0 ? "long" : "short";
    $$(".side-tabs .tab").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.side === side)));
    $("#checks").innerHTML = s.setup[side].checks.map((c) => `<li class="${c.ok ? "ok" : "no"}"><span class="ic">${c.ok ? "✓" : "·"}</span><span class="lbl">${esc(c.label)}</span>${c.detail ? `<span class="det">${esc(c.detail)}</span>` : ""}</li>`).join("");
  }
  function renderLots(s) {
    const rows = s.lots_table || [], sz = s.sizing || {};
    if (!rows.length) { $("#lots").innerHTML = ""; $("#lots-note").textContent = ""; $("#lots-bal").textContent = ""; return; }
    $("#lots-bal").textContent = `本金 $${fmt(sz.balance, 0)}・${fmt(sz.leverage, 0)} 倍`;
    const rec = sz.skip ? null : sz.lots, pd = DEC > 2 ? 3 : 1;
    $("#lots").innerHTML = rows.filter((r) => r.lots <= Math.max(sz.max_lots || 0.3, 0.01) + 1e-9).map((r) => {
      const isRec = rec !== null && Math.abs(r.lots - rec) < 1e-9;
      const cls = [isRec ? "rec" : "", r.loss_pct > 10 ? "danger" : r.loss_pct > 3 ? "risky" : ""].join(" ").trim();
      const liq = !r.can_open ? "保證金不夠" : (r.liq_move <= sz.stop_dist ? `<b>$${fmt(r.liq_move, pd)}</b> ⚠` : `$${fmt(r.liq_move, pd)}`);
      return `<tr class="${cls}"><td>${fmt(r.lots, 2)}${isRec ? ' <span class="tagrec">建議</span>' : ""}</td><td>$${fmt(r.per_dollar, r.per_dollar < 10 ? 2 : 0)}</td><td>$${fmt(r.loss_usd, 0)}<small>${fmt(r.loss_pct, 1)}%</small></td><td>$${fmt(r.margin, 0)}</td><td>${liq}</td></tr>`;
    }).join("");
    $("#lots-note").innerHTML = `以現在的停損距離 $${fmt(sz.stop_dist)}（2 × 1小時ATR）計算。建議手數 = 本金 × ${fmt(sz.risk_pct, 1)}% ÷ 每 0.01 手碰停損的虧損，無條件捨去。黃字＝一次停損超過本金 3%，紅字＝超過 10%。⚠ 代表還沒碰到停損就會先被強平（保證金水平 50%）。`;
  }
  function renderMTF(s) {
    if (!s.mtf) { $("#mtf").innerHTML = ""; return; }
    $("#mtf").innerHTML = s.mtf.map((r) => {
      const t = r.trend > 0 ? ["u", "▲ 多"] : r.trend < 0 ? ["d", "▼ 空"] : ["n", "— 盤"];
      return `<tr><td>${r.tf}</td><td class="trend ${t[0]}">${t[1]}</td><td>${fmt(r.close)}</td><td>${r.rsi != null ? fmt(r.rsi, 0) : "—"}</td><td>${r.adx != null ? fmt(r.adx, 0) : "—"}</td><td>${r.atr != null ? fmt(r.atr) : "—"}</td></tr>`;
    }).join("");
  }
  function spark(arr) {
    if (!arr || arr.length < 2) return "";
    const mn = Math.min(...arr), mx = Math.max(...arr), w = 70, h = 22, up = arr[arr.length - 1] >= arr[0];
    const pts = arr.map((v, i) => `${(i / (arr.length - 1)) * w},${h - 2 - ((v - mn) / ((mx - mn) || 1)) * (h - 4)}`).join(" ");
    return `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline fill="none" stroke="${up ? css("--up") : css("--down")}" stroke-width="1.5" points="${pts}"/></svg>`;
  }
  function renderInter(s) {
    const el = $("#inter");
    if (!s.intermarket || !s.intermarket.length) { el.innerHTML = `<li><span class="nm">${s.mode === "replay" ? "回放模式不顯示" : "載入中…"}</span></li>`; return; }
    el.innerHTML = s.intermarket.map((x) => `<li><span><span class="nm">${esc(x.name)}</span><br>${fmt(x.last, x.last > 1000 ? 1 : 3)}</span><span class="${x.chg_pct > 0 ? "up-c" : x.chg_pct < 0 ? "down-c" : ""}">${sgn(x.chg_pct, 2)}%</span>${spark(x.spark)}</li>`).join("");
  }
  function renderCalendar(s) {
    const evs = (s.calendar && s.calendar.events) || [];
    if (!evs.length) { $("#calendar").innerHTML = `<li><span></span><span class="fc">本週沒有高影響數據，或日曆尚未載入。</span></li>`; return; }
    $("#calendar").innerHTML = evs.slice(0, 12).map((e) => {
      const soon = e.minutes > 0 && e.minutes <= 120, past = e.minutes < 0;
      return `<li class="imp-${e.impact}"><span class="when${soon ? " soon" : ""}">${tLocal(e.time, { month: "2-digit", day: "2-digit" })}<br>${tHM(e.time)}</span><span>${esc(e.country)} ${esc(e.title)}<br><span class="fc">${past ? "已公布" : soon ? `${e.minutes} 分鐘後` : ""}${e.forecast ? ` 預期 ${esc(e.forecast)}` : ""}${e.previous ? ` 前值 ${esc(e.previous)}` : ""}</span></span></li>`;
    }).join("");
  }
  function renderNews(s) {
    const n = s.news || {}, items = n.items || [], agg = n.aggregate;
    $("#news-agg").textContent = agg ? `近12小時 ${agg.label}（${sgn(agg.score, 2)}）` : "";
    if (!items.length) { $("#news").innerHTML = `<li class="nmeta">新聞載入中，或新聞來源暫時連不上。</li>`; return; }
    $("#news").innerHTML = items.slice(0, 30).map((it) => {
      const tag = it.sentiment > 0.2 ? ["bull", "利多"] : it.sentiment < -0.2 ? ["bear", "利空"] : ["neu", "中性"];
      const tr = `https://translate.google.com/?sl=en&tl=zh-TW&text=${encodeURIComponent(it.title)}&op=translate`;
      return `<li><a href="${esc(it.link)}" target="_blank" rel="noopener">${esc(it.title)}</a><div class="nmeta"><span class="tag ${tag[0]}">${tag[1]}</span><span>${esc(it.source)}</span><span>${tLocal(it.time)}</span><a href="${tr}" target="_blank" rel="noopener">翻譯</a></div></li>`;
    }).join("");
  }
  function renderHistory(s) {
    const h = s.history || [];
    if (!h.length) { $("#history").innerHTML = `<li><span></span><span class="t">${s.bar_time ? "最近 45 天沒有訊號" : "等待資料…"}</span><span></span></li>`; return; }
    $("#history").innerHTML = h.map((x) => `<li><span class="d ${dirCls(x.dir)}">${x.dir > 0 ? "多" : "空"}</span><span>${fmt(x.entry)} → ${x.exit != null ? fmt(x.exit) : "持倉中"}<br><span class="t">${tLocal(x.entry_time)}${x.exit_time ? " – " + tLocal(x.exit_time) : ""}　${esc(x.reason)}</span></span><span class="r ${x.R > 0 ? (x.dir > 0 ? "up-c" : "down-c") : ""}">${x.R != null ? sgn(x.R, 2) + "R" : ""}</span></li>`).join("");
  }

  // ------------------------------------------------------------ backtest
  async function renderBacktest() {
    const el = $("#backtest");
    try {
      const d = await api(`/api/backtest?sym=${sym}`);
      const b = d.backtest_summary;
      if (!b) { el.innerHTML = `<p class="cap">尚無回測結果。</p>`; return; }
      const eq = b.equity || [];
      let svg = "";
      if (eq.length > 1) {
        const ys = eq.map((p) => p[1]), mn = Math.min(0, ...ys), mx = Math.max(...ys), w = 300, h = 110;
        const y = (v) => h - 4 - ((v - mn) / ((mx - mn) || 1)) * (h - 8);
        const pts = eq.map((p, i) => `${(i / (eq.length - 1)) * w},${y(p[1])}`).join(" ");
        svg = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="累積報酬曲線"><line x1="0" x2="${w}" y1="${y(0)}" y2="${y(0)}" stroke="${css("--rule")}" stroke-width="1"/><polyline fill="none" stroke="${css("--gold")}" stroke-width="1.6" points="${pts}"/></svg>`;
      }
      const row = (label, sub, p) => `<tr><td>${esc(label)}<span class="sub2">${esc(sub)}</span></td><td>${p.n}</td><td>${fmt(p["win%"], 0)}%</td><td>${sgn(p.avgR, 2)}R</td><td>${fmt(p.maxDD_R, 1)}R</td></tr>`;
      const subs = b.period_subs || ["用來設計策略", "調參後首次檢驗", "最後保留的測試"];
      const rows = (b.periods || []).map((p, i) => row(p.label.replace(/^[^\d]*\s+(?=\d)/, ""), subs[i] || "", p)).join("");
      const rv = d.recent_validation;
      const recent = rv && rv.summary && rv.summary.n ? row(`${rv.window[0].slice(0, 7).replace("-", "/")}–${rv.window[1].slice(0, 7).replace("-", "/")}`, sym === "XAUUSD" ? "全新資料，從未用於設計" : "最近 18 個月", rv.summary) : "";
      let cmp = "";
      const md = b.modes;
      if (md && md.day && md.hold) {
        const a = md.day.all, h = md.hold.all;
        cmp = `<p class="cap">同期比較——當日平倉模式：平均 ${sgn(a.avgR, 2)}R、最大回撤 ${fmt(a.maxDD_R, 1)}R、${fmt(md.day.overnight_pct, 0)}% 的單留過夜；原策略（可一直抱著）：${sgn(h.avgR, 2)}R、${fmt(h.maxDD_R, 1)}R、${fmt(md.hold.overnight_pct, 0)}%。</p>`;
      }
      el.innerHTML = `${svg}<p class="cap">${esc(b.caption || "")}</p><table><thead><tr><th>期間</th><th>筆數</th><th>勝率</th><th>平均</th><th>最大回撤</th></tr></thead><tbody>${rows}${recent}</tbody></table>${cmp}<p class="cap">R = 每筆的初始風險。平均 +0.3R 代表每冒 100 美元風險，長期平均每筆約賺 30 美元（已扣點差、滑價與隔夜費）。勝率只有三到四成：多數單小賠出場，靠少數大波段賺回來，連輸 5–10 筆是正常的。</p>${moneyTable(d.money)}`;
    } catch { el.innerHTML = `<p class="cap">回測結果載入失敗。</p>`; }
  }

  function moneyTable(mo) {
    if (!mo || !mo.rows || !mo.rows.length) return "";
    const bal = (S && S.settings && Number(S.settings.account_balance)) || 2000;
    const bals = [...new Set(mo.rows.map((r) => r.balance))];
    const pick = bals.reduce((a, x) => (Math.abs(x - bal) < Math.abs(a - bal) ? x : a), bals[0]);
    const pct = (x) => `${x > 0 ? "+" : ""}${fmt(x, 0)}%`;
    const recRule = (mo.recommended || {})[String(pick)];
    const rows = mo.rows.filter((r) => r.balance === pick).map((r) => `<tr class="${r.rule === recRule ? "rec" : ""}${r.p_dd50 >= 10 ? " danger" : r.p_dd30 >= 10 ? " risky" : ""}"><td>${esc(r.label)}</td><td>${pct(r.med_ret)}</td><td>${pct(r.p5_ret)}</td><td>${fmt(r.p_dd30, 0)}%</td><td>${fmt(r.p_dd50, 0)}%</td></tr>`).join("");
    return `<h3 class="bt-h3">本金 $${fmt(pick, 0)}：每筆下幾手，一年後會怎樣？</h3><table class="money"><thead><tr><th>每筆手數</th><th>一般</th><th>運氣差</th><th>跌三成</th><th>腰斬</th></tr></thead><tbody>${rows}</tbody></table><p class="cap">${esc(mo.caption || "")}</p>`;
  }

  // ------------------------------------------------------------ alerts
  function chime(kind) {
    if (!audioCtx || !(S && S.settings && S.settings.sound)) return;
    const notes = kind === "signal" || kind === "test" ? [880, 1175, 1568] : kind === "exit" || kind === "stop_hit" ? [660, 440] : [740];
    const t0 = audioCtx.currentTime;
    notes.forEach((f, i) => {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = "sine"; o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, t0 + i * 0.16); g.gain.exponentialRampToValueAtTime(0.25, t0 + i * 0.16 + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t0 + i * 0.16 + 0.32);
      o.connect(g).connect(audioCtx.destination); o.start(t0 + i * 0.16); o.stop(t0 + i * 0.16 + 0.35);
    });
  }
  function toast(a) {
    const el = document.createElement("div");
    el.className = "toast " + (a.level || "");
    el.innerHTML = `<strong>${esc(a.title)}</strong><p>${esc(a.body)}</p>`;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), a.level === "signal" ? 60000 : 15000);
    el.addEventListener("click", () => el.remove());
  }
  function handleAlerts(s) {
    const al = s.alerts || [];
    const maxId = al.reduce((m, a) => Math.max(m, a.id), 0);
    if (maxId < lastAlertId) lastAlertId = 0; // server restarted
    al.filter((a) => a.id > lastAlertId && (!firstLoad || (Date.now() - new Date(a.time).getTime()) < 120000)).forEach((a) => {
      toast(a); chime(a.kind);
      if (S && S.settings && S.settings.desktop_notify && "Notification" in window && Notification.permission === "granted") {
        try { new Notification(a.title, { body: a.body, tag: "metal-" + a.id }); } catch { /* ignore */ }
      }
    });
    lastAlertId = Math.max(lastAlertId, maxId); store.set("lastAlertId", lastAlertId);
  }

  // ------------------------------------------------------------ settings
  function buildInstSettings(st) {
    const names = { XAUUSD: ["黃金", 100], XAGUSD: ["白銀", 5000] };
    $("#inst-settings").innerHTML = Object.keys(st.instruments || {}).map((k) => {
      const [nm, oz] = names[k] || [k, 100];
      return `<fieldset class="inst-fs" data-key="${k}"><legend>${nm}</legend>
        <label>每筆風險（% 資金）<input data-k="risk_pct" type="number" step="0.1" min="0.1" max="5"></label>
        <label>每 1 手 = 幾盎司 <input data-k="oz_per_lot" type="number" step="1" min="1"><small>請在 Mitrade ${nm}商品資訊確認（常見為 ${oz}）。</small></label>
        <label class="check"><input data-k="confirmed" type="checkbox"> 我已在 Mitrade 確認過合約大小</label>
        <label>最小手數 <input data-k="min_lot" type="number" step="0.01" min="0.01"></label>
        <label>價格校正（加到本程式報價上）<input data-k="price_offset" type="number" step="0.001"><small>若 Mitrade 報價比這裡高 0.30，就填 0.30。</small></label>
      </fieldset>`;
    }).join("");
  }
  function fillSettings(st) {
    if (!settingsBuilt) { buildInstSettings(st); settingsBuilt = true; }
    const f = $("#settings-form");
    Object.entries(st).forEach(([k, v]) => {
      const el = f.elements[k];
      if (!el || el.type === undefined) return;
      if (el.type === "checkbox") el.checked = !!v; else el.value = v;
    });
    $$(".inst-fs").forEach((fs) => {
      const vals = (st.instruments || {})[fs.dataset.key] || {};
      $$("input", fs).forEach((inp) => { const v = vals[inp.dataset.k]; if (inp.type === "checkbox") inp.checked = !!v; else if (v !== undefined) inp.value = v; });
    });
  }
  async function saveSettings(e) {
    e.preventDefault();
    const f = e.target, data = { instruments: {} };
    [...f.elements].forEach((el) => { if (el.name) data[el.name] = el.type === "checkbox" ? el.checked : el.value; });
    $$(".inst-fs").forEach((fs) => {
      const o = {};
      $$("input", fs).forEach((inp) => { o[inp.dataset.k] = inp.type === "checkbox" ? inp.checked : inp.value; });
      data.instruments[fs.dataset.key] = o;
    });
    fillSettings(await post("/api/settings", data));
    $("#settings-saved").hidden = false; setTimeout(() => ($("#settings-saved").hidden = true), 1800);
    refresh();
  }

  // ------------------------------------------------------------ loop
  async function refresh() {
    try {
      const s = await api(`/api/state?sym=${sym}`);
      S = s;
      renderTabs(s); renderHeader(s); renderTicket(s); renderLots(s); renderChecks(s); renderMTF(s); renderInter(s);
      renderCalendar(s); renderNews(s); renderHistory(s); handleAlerts(s);
      if (!settingsBuilt && s.settings) fillSettings(s.settings);
      if (s.bar_time && s.bar_time !== lastBarTime) { lastBarTime = s.bar_time; loadCandles(); }
      firstLoad = false;
    } catch (e) {
      $("#fresh .dot").className = "dot bad"; $("#fresh-text").textContent = "程式未連線";
      const b = $("#banner"); b.hidden = false; b.className = "banner bad";
      b.textContent = "⚠ 連不到本機程式。請確認啟動視窗還開著（run.py），然後重新整理頁面。";
    }
  }

  function bind() {
    $$(".tf-tabs .tab").forEach((b) => b.addEventListener("click", () => setTf(b.dataset.tf)));
    $$(".side-tabs .tab").forEach((b) => b.addEventListener("click", () => { side = b.dataset.side; if (S) renderChecks(S); }));
    $("#btn-settings").addEventListener("click", () => { const d = $("#settings"); d.hidden = !d.hidden; $("#btn-settings").setAttribute("aria-expanded", String(!d.hidden)); if (!d.hidden && S) fillSettings(S.settings); });
    $("#settings-close").addEventListener("click", () => { $("#settings").hidden = true; $("#btn-settings").setAttribute("aria-expanded", "false"); });
    $("#settings-form").addEventListener("submit", saveSettings);
    $("#btn-test").addEventListener("click", () => post("/api/test_alert", {}).then(refresh));
    $("#btn-enable").addEventListener("click", async () => {
      try { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); await audioCtx.resume(); } catch { /* no audio */ }
      if ("Notification" in window && Notification.permission === "default") { try { await Notification.requestPermission(); } catch { /* ignore */ } }
      chime("signal");
      $("#btn-enable").textContent = "提醒已開啟"; $("#btn-enable").disabled = true;
    });
    document.addEventListener("click", (e) => {
      const v = e.target.closest("[data-copy]");
      if (v && navigator.clipboard) navigator.clipboard.writeText(v.dataset.copy).then(() => toast({ title: "已複製", body: v.dataset.copy, level: "" })).catch(() => {});
    });
    $("#mypos-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const f = e.target;
      const t = f.elements.time.value ? new Date(f.elements.time.value).toISOString() : new Date().toISOString();
      await post(`/api/my_position?sym=${sym}`, { dir: Number(f.elements.dir.value), entry: Number(f.elements.entry.value), lots: Number(f.elements.lots.value || 0), time: t });
      refresh();
    });
    $("#mypos-clear").addEventListener("click", async () => { await post(`/api/my_position?sym=${sym}`, {}); refresh(); });
    $("#replay-speed").addEventListener("change", (e) => post("/api/replay", { speed: Number(e.target.value) }));
    $("#replay-pause").addEventListener("click", () => post("/api/replay", { paused: !(S && S.replay && S.replay.paused) }).then(refresh));
    $("#replay-jump").addEventListener("change", (e) => { if (e.target.value) post("/api/replay", { jump: e.target.value }).then(() => { chartFitted = {}; refresh(); }); });
  }

  initChart(); bind(); setTf(tf); refresh().then(renderBacktest);
  setInterval(refresh, 3000);
  setInterval(() => { if (tf === "15m" || S?.mode === "replay") loadCandles(); }, 20000);
})();
