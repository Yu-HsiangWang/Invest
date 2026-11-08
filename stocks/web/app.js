/* 科技股買賣區間 — front end */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : (x > 0 ? "+" : "") + (x * 100).toFixed(d) + "%";
  const cls = (x) => x > 0 ? "up-c" : (x < 0 ? "down-c" : "");
  const tLocal = (iso) => iso ? new Date(iso).toLocaleString("zh-TW", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }) : "—";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const ZW = { "-2": "很便宜", "-1": "便宜", "0": "合理", "1": "偏貴", "2": "很貴" };
  const MN = { channel: "五線譜", deviation: "年線乖離", pe: "本益比河流圖" };

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
  }
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  const state = { market: localStorage.getItem("mkt") || "TW", sym: null, tf: "1D", cards: [], detail: null, feed: "news", mkTab: null, settings: {}, bands: { ch: true, dv: false, pe: false, ma: true }, seenAlerts: new Set() };
  try { const b = JSON.parse(localStorage.getItem("bands") || "null"); if (b) state.bands = b; } catch (e) { /* ignore */ }

  // ------------------------------------------------------------------ overview
  async function loadOverview() {
    const ov = await api(`/api/overview?market=${state.market}`);
    state.cards = ov.cards;
    $("#st-prices").textContent = ov.status.prices || "—";
    $("#st-feeds").textContent = ov.status.feeds || "—";
    $("#st-time").textContent = tLocal(ov.status.last_price_refresh);
    renderRegime(ov.regime);
    renderWatch();
    if (!state.sym || !state.cards.find((c) => c.symbol === state.sym)) {
      state.sym = localStorage.getItem(`sym-${state.market}`) || (state.cards[0] && state.cards[0].symbol);
      if (state.sym && !state.cards.find((c) => c.symbol === state.sym)) state.sym = state.cards[0] && state.cards[0].symbol;
    }
    if (state.sym) await loadDetail();
  }

  function renderRegime(r) {
    const el = $("#regime");
    el.innerHTML = Object.entries(r || {}).map(([k, v]) => {
      const st = v.faber_up === null ? "" : `<span class="state ${v.faber_up ? "on" : "off"}">10 月均線 ${v.faber_up ? "上" : "下"}</span>`;
      return `<span><b>${esc(v.name)}</b> ${fmt(v.last, 0)} ${st} <span>離高點 ${pct(v.dd_52w, 0)}</span> <span>一年 ${pct(v.ret_1y, 0)}</span></span>`;
    }).join("");
  }

  function zoneChips(zs) {
    const by = Object.fromEntries((zs || []).map((z) => [z.method, z]));
    return `<span class="zones" title="${esc((zs || []).map((z) => `${MN[z.method]}：${z.label}`).join("　"))}">` +
      ["channel", "deviation", "pe"].map((m) => { const z = by[m]; return `<i class="zc ${z && z.zone !== null ? `z${z.zone}` : "zn"}"></i>`; }).join("") + "</span>";
  }

  function renderWatch() {
    const q = ($("#filter").value || "").trim().toLowerCase();
    const sort = $("#sort").value;
    const order = { buy: 0, "buy-soft": 1, hold: 2, "hold-warn": 3, neutral: 4 };
    let rows = state.cards.filter((c) => !q || c.name.toLowerCase().includes(q) || c.symbol.toLowerCase().includes(q) || (c.sector || "").includes(q));
    const key = { verdict: (c) => [order[c.verdict_cls] ?? 9, c.zone_min ?? 9], zone: (c) => [c.zone_min ?? 9, c.zone_max ?? 9], chg: (c) => [-(c.chg_1d ?? -9)], ret1y: (c) => [-((c.ret || {})["1y"] ?? -9)], name: (c) => [c.name] }[sort];
    rows.sort((a, b) => { const ka = key(a), kb = key(b); for (let i = 0; i < ka.length; i++) { if (ka[i] < kb[i]) return -1; if (ka[i] > kb[i]) return 1; } return 0; });
    $("#watch").innerHTML = rows.map((c) => `<tr data-sym="${esc(c.symbol)}" aria-selected="${c.symbol === state.sym}">
      <td><span class="nm">${esc(c.name)}</span><span class="cd">${esc(c.code || c.symbol)}${c.kind === "etf" ? " · ETF" : ""}${c.badges.map((b) => `<span class="badge">${esc(b)}</span>`).join("")}</span></td>
      <td class="num">${fmt(c.price, c.price > 500 ? 0 : 2)}</td>
      <td class="num ${cls(c.chg_1d)}">${pct(c.chg_1d)}</td>
      <td>${zoneChips(c.zones)}</td>
      <td><span class="vtag ${esc(c.verdict_cls)}">${esc(c.verdict)}</span></td></tr>`).join("");
    $$("#watch tr").forEach((tr) => tr.addEventListener("click", () => { state.sym = tr.dataset.sym; localStorage.setItem(`sym-${state.market}`, state.sym); renderWatch(); loadDetail(); }));
  }

  // -------------------------------------------------------------------- detail
  async function loadDetail() {
    const d = await api(`/api/instrument?sym=${encodeURIComponent(state.sym)}`);
    if (d.error) return;
    state.detail = d;
    renderTicket(d);
    renderCloud(d);
    renderDefense(d);
    renderFeeds(d);
    await loadChart();
  }

  function renderTicket(d) {
    const v = d.verdict || {};
    const stamp = $("#stamp");
    stamp.className = `stamp ${v.cls || "neutral"}`;
    $("#stamp-text").textContent = v.stamp || "—";
    $("#d-name").textContent = d.name;
    $("#d-sym").textContent = d.symbol + (d.kind === "etf" ? " · ETF" : "");
    $("#d-sector").textContent = d.sector || "";
    $("#d-px").textContent = fmt(d.price, d.price > 500 ? 0 : 2);
    $("#d-chg").textContent = pct(d.chg_1d); $("#d-chg").className = `chg ${cls(d.chg_1d)}`;
    const r = d.ret || {};
    $("#d-ret").innerHTML = ["1m", "3m", "6m", "1y"].map((k) => `${k.replace("m", " 個月").replace("1y", "一年")} <b class="${cls(r[k])}">${pct(r[k], 0)}</b>`).join("　") + `　離 52 週高點 <b>${pct(d.dd_from_high_52w, 0)}</b>`;
    $("#d-title").textContent = v.title || "";
    $("#d-body").textContent = v.body || "";
    $("#d-hist").textContent = v.hist || "";
    // levels
    $("#levels").innerHTML = (d.zones || []).map((z) => {
      const L = z.levels || {};
      const rowsDef = z.method === "pe"
        ? [["很貴（本益比 90%）", "up2", z.pe_bands && z.pe_bands.up2], ["偏貴（75%）", "up1", z.pe_bands && z.pe_bands.up1], ["中位數", "center", z.pe_bands && z.pe_bands.mid], ["便宜（25%）", "dn1", z.pe_bands && z.pe_bands.dn1], ["很便宜（10%）", "dn2", z.pe_bands && z.pe_bands.dn2]]
        : z.method === "channel"
          ? [["+2σ 很貴", "up2"], ["+1σ 偏貴", "up1"], ["趨勢中線", "center"], ["−1σ 便宜", "dn1"], ["−2σ 很便宜", "dn2"]]
          : [["乖離 95%", "up2"], ["乖離 85%", "up1"], ["200 日均線", "center"], ["乖離 15%", "dn1"], ["乖離 5%", "dn2"]];
      const cur = z.zone;
      const curKey = { 2: "up2", 1: "up1", 0: "center", "-1": "dn1", "-2": "dn2" }[String(cur)];
      const extra = z.method === "channel" ? `z = ${fmt(z.zscore, 2)}，趨勢年化 ${pct(z.slope_annual, 0)}`
        : z.method === "deviation" ? `乖離 ${pct(z.dev, 1)}，在自己 5 年的第 ${fmt((z.dev_rank || 0) * 100, 0)} 百分位`
          : `本益比 ${fmt(z.pe, 1)}（第 ${fmt((z.pe_rank || 0) * 100, 0)} 百分位），近 4 季 EPS ${fmt(z.eps_ttm, 2)}`;
      return `<div class="lv"><h3>${esc(MN[z.method])} <span class="vtag ${cur <= -1 ? (cur <= -2 ? "buy" : "buy-soft") : cur >= 1 ? "hold-warn" : "hold"}">${esc(z.label)}</span></h3>
        <table>${rowsDef.map(([lab, k, pe]) => `<tr class="${k === curKey ? "cur-row" : ""}"><td>${esc(lab)}${pe !== undefined && pe !== null ? ` <span class="zl">P/E ${fmt(pe, 1)}</span>` : ""}</td><td>${fmt(L[k], L[k] > 500 ? 0 : 2)}</td></tr>`).join("")}</table>
        <div class="zl">${extra}</div></div>`;
    }).join("");
    // trend
    const t = d.trend || {}; const votes = t.votes || {};
    const names = { above_100: "100 日", above_150: "150 日", above_200: "200 日", above_250: "250 日", slope_200: "年線上揚", mom_12_1: "12 個月動能", ret_6m: "6 個月" };
    const mt = d.market_trend || {};
    $("#d-trend").innerHTML = `<span><b>趨勢：${esc(t.label || "—")}</b></span>` +
      `<span>${Object.entries(votes).map(([k, v]) => `<i class="vote ${v > 0 ? "up" : v < 0 ? "down" : ""}" title="${esc(names[k] || k)}：${v > 0 ? "多" : v < 0 ? "空" : "—"}"></i>`).join("")} ${Object.values(votes).filter((v) => v > 0).length}/${Object.keys(votes).length} 票看多</span>` +
      (mt.faber_up !== undefined && mt.faber_up !== null ? `<span>大盤 10 月均線：<span class="state ${mt.faber_up ? "on" : "off"}">${mt.faber_up ? "多" : "空"}</span>（現價離均線 ${pct(mt.dist_pct, 1)}${mt.live_above_10m === false && mt.faber_up ? "，盤中已跌破，月底確認" : ""}）</span>` : "") +
      `<span class="muted">${esc(t.reason || "")}</span>`;
  }

  function cloudRow(label, c) {
    if (!c || c.hit === undefined) return `<tr class="sub"><td>${esc(label)}</td><td colspan="5" class="muted">樣本不足</td></tr>`;
    return `<tr><td>${esc(label)}</td><td class="num"><b>${fmt(c.hit * 100, 0)}%</b> <span class="muted">(${fmt(c.hit_all * 100, 0)}%)</span></td><td class="num">${pct(c.median, 0)}</td><td class="num">${pct(c.p10, 0)} ～ ${pct(c.p90, 0)}</td><td class="num muted">${fmt(c.n_eff, 0)}</td></tr>`;
  }
  function renderCloud(d) {
    const cloud = d.cloud || {}; const pooled = d.pooled || {};
    const zoneOf = Object.fromEntries((d.zones || []).map((z) => [z.method, z.zone]));
    let html = "";
    for (const m of ["channel", "deviation", "pe"]) {
      if (!cloud[m] && !pooled[m]) continue;
      const z = zoneOf[m]; if (z === undefined || z === null) continue;
      html += `<h3 class="muted" style="font-size:13px;margin:10px 0 4px">${esc(MN[m])}：現在在「${esc(ZW[String(z)])}」</h3>
      <table class="grid"><thead><tr><th></th><th class="num">上漲機率 (平時)</th><th class="num">中位數</th><th class="num">10–90% 區間</th><th class="num">樣本</th></tr></thead><tbody>`;
      for (const h of ["63", "126", "252"]) {
        const lab = { 63: "3 個月", 126: "6 個月", 252: "12 個月" }[h];
        html += cloudRow(`自己 · ${lab}`, (cloud[m] || {})[h]);
      }
      const p = (pooled[m] || {})[String(z)];
      if (p) html += `<tr class="sub"><td>同市場合併 · 12 個月</td><td class="num"><b>${fmt(p.hit * 100, 0)}%</b> (${fmt(p.base_hit * 100, 0)}%)</td><td class="num">${pct(p.median, 0)}</td><td class="num">平均比平時 ${pct(p.excess_mean, 0)}</td><td class="num muted">${fmt(p.n_eff, 0)}</td></tr>`;
      html += "</tbody></table>";
    }
    if (cloud.trend) {
      html += `<h3 class="muted" style="font-size:13px;margin:10px 0 4px">趨勢狀態：${esc(d.trend.label)}</h3><table class="grid"><tbody>` +
        ["63", "126", "252"].map((h) => cloudRow(`自己 · ${{ 63: "3 個月", 126: "6 個月", 252: "12 個月" }[h]}`, cloud.trend[h])).join("") + "</tbody></table>";
    }
    $("#cloud").innerHTML = html || `<p class="muted">歷史不足。</p>`;
  }

  function renderDefense(d) {
    const b = d.defense_backtest || {};
    const keys = Object.keys(b);
    if (!keys.length) { $("#defense").innerHTML = `<p class="muted">歷史不足 2.5 年。</p>`; return; }
    $("#defense").innerHTML = `<table class="grid"><thead><tr><th>期間</th><th class="num">買進持有 年化／最大回撤</th><th class="num">10 月均線 年化／最大回撤</th><th class="num">在場時間</th></tr></thead><tbody>` +
      keys.map((k) => { const r = b[k]; return `<tr><td>${esc(k)}（${r.years} 年）</td><td class="num">${pct(r.bh_cagr, 1)} ／ ${pct(r.bh_dd, 0)}</td><td class="num">${pct(r.faber_cagr, 1)} ／ ${pct(r.faber_dd, 0)}</td><td class="num">${fmt(r.faber_time_in * 100, 0)}%</td></tr>`; }).join("") +
      `</tbody></table>` + (state.settings.defense_mode && d.kind === "etf" && d.market_trend && d.market_trend.faber_up === false ? `<p class="v-title" style="color:var(--z2)">防守模式：大盤 10 月均線向下，建議這檔 ETF 減碼至半倉或空手，等月底站回均線再買回。</p>` : "");
  }

  // --------------------------------------------------------------------- feeds
  function lab(s) { return s > 0.2 ? `<span class="lab bull">利多</span>` : s < -0.2 ? `<span class="lab bear">利空</span>` : `<span class="lab neu">中性</span>`; }
  function renderFeeds(d) {
    const f = d.feeds || {};
    const agg = f.news_agg || {};
    $("#feeds-agg").textContent = agg.n ? `新聞 ${agg.label}（利多 ${agg.bullish}／利空 ${agg.bearish}，近 3 天 ${agg.n} 則）` : "";
    const isUS = d.market === "US";
    $$("#feeds-h ~ .feed-tabs .tab").forEach((b) => { b.hidden = (b.dataset.feed === "congress" && !isUS) || (b.dataset.feed === "flow" && isUS); });
    if ((state.feed === "congress" && !isUS) || (state.feed === "flow" && isUS)) state.feed = "news";
    $$(".feed-tabs .tab[data-feed]").forEach((b) => b.setAttribute("aria-selected", b.dataset.feed === state.feed));
    let html = "";
    if (state.feed === "news") {
      html = `<ul>${(f.news || []).slice(0, 30).map((n) => `<li><span class="t">${tLocal(n.time)} · ${esc(n.source)}</span><a href="${esc(n.link)}" target="_blank" rel="noopener">${esc(n.title)}</a>${lab(n.sentiment)}</li>`).join("") || "<li class='muted'>沒有新聞（離線或尚未抓取）</li>"}</ul>`;
    } else if (state.feed === "insiders") {
      if (isUS) {
        const s = f.insider_summary || {};
        html = `<dl class="kv"><dt>近 180 天 Form 4</dt><dd>公開市場買進 <b>${s.buys || 0}</b> 筆（${s.buyers || 0} 人，$${fmt((s.buy_value || 0) / 1e6, 1)}M）／賣出 ${s.sells || 0} 筆（$${fmt((s.sell_value || 0) / 1e6, 1)}M）</dd></dl>` +
          `<ul>${(f.insiders || []).map((r) => `<li><span class="t">${esc(r.date || r.filed)}</span>${esc(r.owner)} <span class="muted">${esc(r.role)}</span> ${r.code === "P" ? '<span class="lab buy">買進</span>' : '<span class="lab sell">賣出</span>'} ${fmt(r.shares, 0)} 股 @ ${fmt(r.price, 2)}（$${fmt((r.value || 0) / 1e3, 0)}K）</li>`).join("") || "<li class='muted'>沒有資料（SEC 需在本機連線）</li>"}</ul>`;
      } else {
        html = `<p class="muted" style="font-size:13px">證交所「內部人轉讓事前申報」（董監、經理人、大股東預定賣出／轉讓；只有賣方需事前申報，所以這裡幾乎都是賣出，研究顯示內部人賣出資訊很少）。</p>` +
          `<ul>${(f.insiders || []).map((r) => `<li><span class="t">${esc(r.published)}</span>${esc(r.who)} <span class="muted">${esc(r.role)}</span> ${esc(r.method)} ${fmt(r.shares, 0)} 股（持有 ${fmt(r.holding, 0)}）</li>`).join("") || "<li class='muted'>最近沒有申報，或尚未抓取</li>"}</ul>`;
      }
    } else if (state.feed === "congress") {
      const c = f.congress || {};
      html = `<dl class="kv"><dt>近 90 天</dt><dd>買進 <b>${c.buys || 0}</b> 筆（${c.buyers || 0} 位）／賣出 ${c.sells || 0} 筆（${c.sellers || 0} 位）${c.cluster ? ' <span class="lab buy">集群買進</span>' : ""}</dd></dl>` +
        `<ul>${(f.congress_trades || []).map((t) => `<li><span class="t">申報 ${esc(t.filed)} · 交易 ${esc(t.traded)}</span>${esc(t.who)} <span class="muted">${esc(t.party)}·${esc(t.chamber)}</span> ${/^Purchase/.test(t.type) ? '<span class="lab buy">買</span>' : '<span class="lab sell">賣</span>'} ${esc(t.amount)}${t.late ? ' <span class="lab neu">逾期申報</span>' : ""} <a href="${esc(t.doc)}" target="_blank" rel="noopener">原始文件</a></li>`).join("") || "<li class='muted'>近一年沒有申報</li>"}</ul>`;
    } else {
      const s = f.institutional_summary;
      const rev = f.revenue_history || (f.revenue || []).map((r) => ({ period: r.period, revenue: r.revenue, yoy: r.yoy }));
      html = (s ? `<dl class="kv"><dt>外資淨買賣超</dt><dd>5 日 <b class="${cls(s.foreign_5d)}">${fmt(s.foreign_5d / 1000, 0)}</b> 張／20 日 <b class="${cls(s.foreign_20d)}">${fmt(s.foreign_20d / 1000, 0)}</b> 張</dd><dt>投信 20 日</dt><dd class="${cls(s.trust_20d)}">${fmt(s.trust_20d / 1000, 0)} 張</dd></dl>` : `<p class="muted">法人資料需在本機連線（FinMind）。</p>`) +
        (f.valuation_today ? `<dl class="kv"><dt>證交所 ${esc(f.valuation_today.date)}</dt><dd>本益比 ${fmt(f.valuation_today.pe, 1)}、殖利率 ${fmt(f.valuation_today.yield, 2)}%、股價淨值比 ${fmt(f.valuation_today.pb, 2)}</dd></dl>` : "") +
        (rev.length ? `<table class="grid"><thead><tr><th>月營收</th><th class="num">金額（千元）</th><th class="num">年增率</th></tr></thead><tbody>${rev.slice(-12).reverse().map((r) => `<tr><td>${esc(r.period)}</td><td class="num">${fmt(r.revenue / 1000, 0)}</td><td class="num ${cls(r.yoy)}">${r.yoy === null || r.yoy === undefined ? "—" : (r.yoy > 0 ? "+" : "") + fmt(r.yoy, 1) + "%"}</td></tr>`).join("")}</tbody></table>` : `<p class="muted">月營收資料需在本機連線。</p>`);
    }
    $("#feed-body").innerHTML = html;
  }
  $$(".feed-tabs .tab[data-feed]").forEach((b) => b.addEventListener("click", () => { state.feed = b.dataset.feed; if (state.detail) renderFeeds(state.detail); }));

  // --------------------------------------------------------------------- chart
  let chart, candles, lines = {};
  const shift = (t) => t - new Date(t * 1000).getTimezoneOffset() * 60;
  function chartColors() {
    return { background: { type: "solid", color: css("--panel") }, textColor: css("--muted") };
  }
  function initChart() {
    chart = LightweightCharts.createChart($("#chart"), { layout: chartColors(), grid: { vertLines: { color: css("--rule-soft") }, horzLines: { color: css("--rule-soft") } }, rightPriceScale: { borderColor: css("--rule") }, timeScale: { borderColor: css("--rule"), timeVisible: false }, crosshair: { mode: 1 }, autoSize: true });
    const up = css("--up"), down = css("--down");
    candles = chart.addCandlestickSeries({ upColor: up, downColor: down, borderUpColor: up, borderDownColor: down, wickUpColor: up, wickDownColor: down, priceLineVisible: true });
    const line = (color, w = 1, style = 0) => chart.addLineSeries({ color, lineWidth: w, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    lines = {
      ch_up2: line(css("--z2"), 1, 2), ch_up1: line(css("--z1"), 1, 2), ch_center: line(css("--z0"), 1, 0), ch_dn1: line(css("--z-1"), 1, 2), ch_dn2: line(css("--z-2"), 1, 2),
      dv_up2: line(css("--z2"), 1, 1), dv_up1: line(css("--z1"), 1, 1), dv_dn1: line(css("--z-1"), 1, 1), dv_dn2: line(css("--z-2"), 1, 1),
      pe_up2: line(css("--z2"), 1, 3), pe_up1: line(css("--z1"), 1, 3), pe_mid: line(css("--z0"), 1, 3), pe_dn1: line(css("--z-1"), 1, 3), pe_dn2: line(css("--z-2"), 1, 3),
      sma200: line(css("--gold"), 2, 0),
    };
  }
  function bandOf(k) { return k.startsWith("ch_") ? "ch" : k.startsWith("dv_") ? "dv" : k.startsWith("pe_") ? "pe" : "ma"; }
  async function loadChart() {
    if (!chart) initChart();
    const c = await api(`/api/chart?sym=${encodeURIComponent(state.sym)}&tf=${state.tf}&limit=${state.tf === "1D" ? 1500 : 3000}`);
    candles.setData(c.bars.map((b) => ({ time: shift(b.time), open: b.open, high: b.high, low: b.low, close: b.close })));
    for (const [k, s] of Object.entries(lines)) {
      const on = state.bands[bandOf(k)];
      const pts = c.lines[k];
      if (!on || !pts) { s.setData([]); continue; }
      s.setData(pts.filter((p) => p[1] !== null).map(([t, v]) => ({ time: shift(t), value: v })));
    }
    chart.timeScale().fitContent();
  }
  $$(".tf-tabs .tab").forEach((b) => b.addEventListener("click", () => { state.tf = b.dataset.tf; $$(".tf-tabs .tab").forEach((x) => x.setAttribute("aria-selected", x === b)); loadChart(); }));
  $$(".band-toggles input").forEach((i) => { i.checked = !!state.bands[i.dataset.band]; i.addEventListener("change", () => { state.bands[i.dataset.band] = i.checked; localStorage.setItem("bands", JSON.stringify(state.bands)); loadChart(); }); });

  // -------------------------------------------------------------------- market
  async function loadMarket() {
    const m = await api(`/api/market?market=${state.market}`);
    const reg = m.regime || {};
    $("#market").innerHTML = `<div class="mk">${Object.values(reg).map((v) => `<div class="row"><span><b>${esc(v.name)}</b> <span class="muted">${esc(v.date)}</span></span><span>${fmt(v.last, 0)} · <span class="state ${v.faber_up ? "on" : "off"}">10 月均線${v.faber_up ? "上" : "下"}</span> <span class="state ${v.above_200 ? "on" : "off"}">200 日${v.above_200 ? "上" : "下"}</span></span></div>`).join("")}</div>` +
      (state.settings.defense_mode ? `<p class="hint">防守模式開啟：以「上個月底」的 10 月均線狀態為準，盤中跌破不算。</p>` : `<p class="hint">趨勢只用來控風險（防守模式在設定裡），不代表看多看空。</p>`);
    const cal = m.calendar || {};
    $("#calendar").innerHTML = `<div class="cal">` +
      (cal.fomc && cal.fomc.length ? `<div class="row"><span>下次 FOMC</span><b>${cal.fomc.map(esc).join("、")}</b></div>` : "") +
      (state.market === "TW" ? `<div class="row"><span>月營收公告截止</span><b>${esc(cal.tw_revenue_deadline)}</b></div>` : "") +
      (m.macro || []).map((x) => `<div class="row"><span>${esc(x.name)} <span class="muted">${esc(x.date)}</span></span><b>${fmt(x.value, 2)} <span class="muted ${cls(x.chg_1m)}">${x.chg_1m === null ? "" : (x.chg_1m > 0 ? "+" : "") + fmt(x.chg_1m, 2) + " /月"}</span></b></div>`).join("") + `</div>`;
    const agg = m.news_agg || {};
    $("#mk-agg").textContent = agg.n ? `新聞 ${agg.label}（利多 ${agg.bullish}／利空 ${agg.bearish}）` : "";
    const tabs = state.market === "US" ? [["news", "新聞"], ["congress", "國會交易（名單）"], ["congress_all", "國會交易（全部）"], ["flow", "議員買賣週統計"]] : [["news", "新聞"], ["material", "重大訊息"], ["insider", "內部人申報"]];
    if (!tabs.find((t) => t[0] === state.mkTab)) state.mkTab = "news";
    $("#mk-tabs").innerHTML = tabs.map(([k, l]) => `<button class="tab" data-mk="${k}" aria-selected="${k === state.mkTab}">${esc(l)}</button>`).join("");
    $$("#mk-tabs .tab").forEach((b) => b.addEventListener("click", () => { state.mkTab = b.dataset.mk; renderMarketFeed(m); }));
    renderMarketFeed(m);
    $("#alerts").innerHTML = (m.alerts || []).slice(0, 25).map((a) => `<li><span class="t">${tLocal(a.time)} · ${esc(a.symbol)}</span><br><b>${esc(a.title)}</b> <span class="muted">${esc(a.body)}</span></li>`).join("") || "<li class='muted'>還沒有提醒</li>";
    notifyNew(m.alerts || []);
    const bad = Object.entries(m.feed_status || {}).filter(([, v]) => /error/.test(v));
    const banner = $("#banner");
    if (bad.length > 3) { banner.hidden = false; banner.textContent = `有 ${bad.length} 個資料來源抓不到（${bad.slice(0, 3).map(([k]) => k.replace(/^https?:\/\//, "").slice(0, 40)).join("、")}…）。可能是離線或對方網站暫時擋住；其餘功能不受影響。`; } else banner.hidden = true;
  }
  function renderMarketFeed(m) {
    $$("#mk-tabs .tab").forEach((b) => b.setAttribute("aria-selected", b.dataset.mk === state.mkTab));
    let html = "";
    const congressLi = (t) => `<li><span class="t">${esc(t.filed)}</span>${esc(t.who)} <span class="muted">${esc(t.party)}·${esc(t.chamber)}</span> ${/^Purchase/.test(t.type) ? '<span class="lab buy">買</span>' : '<span class="lab sell">賣</span>'} <b>${esc(t.ticker || "")}</b> ${esc(t.asset || "").slice(0, 40)} ${esc(t.amount)} <a href="${esc(t.doc)}" target="_blank" rel="noopener">文件</a></li>`;
    if (state.mkTab === "news") html = `<ul>${(m.news || []).slice(0, 40).map((n) => `<li><span class="t">${tLocal(n.time)} · ${esc(n.source)}</span><a href="${esc(n.link)}" target="_blank" rel="noopener">${esc(n.title)}</a>${lab(n.sentiment)}</li>`).join("") || "<li class='muted'>沒有新聞（離線或尚未抓取）</li>"}</ul>`;
    else if (state.mkTab === "congress") html = `<p class="muted" style="font-size:12.5px">名單內標的，近 45 天申報。資料：kadoa congress-trading-monitor 每日解析官方申報；${esc(m.congress_status || "")}</p><ul>${(m.congress_recent || []).map(congressLi).join("") || "<li class='muted'>沒有</li>"}</ul>`;
    else if (state.mkTab === "congress_all") html = `<ul>${(m.congress_all_recent || []).map(congressLi).join("") || "<li class='muted'>沒有</li>"}</ul>`;
    else if (state.mkTab === "flow") html = `<table class="grid"><thead><tr><th>週</th><th class="num">買進筆數</th><th class="num">賣出筆數</th><th class="num">申報人數</th></tr></thead><tbody>${(m.congress_flow || []).slice().reverse().map((w) => `<tr><td>${esc(w.week)}</td><td class="num">${w.buys}</td><td class="num">${w.sells}</td><td class="num">${w.members}</td></tr>`).join("")}</tbody></table><p class="hint">依申報日統計、所有股票。只是「議員整體在買還是賣」的粗略溫度計。</p>`;
    else if (state.mkTab === "material") html = `<ul>${(m.material_info || []).map((x) => `<li><span class="t">${esc(x.date)} · ${esc(x.name)}</span>${esc(x.subject)}</li>`).join("") || "<li class='muted'>需在本機連線到證交所</li>"}</ul>`;
    else if (state.mkTab === "insider") html = `<ul>${(m.insider_transfers || []).map((x) => `<li><span class="t">${esc(x.published)} · ${esc(x.name)}</span>${esc(x.who)} <span class="muted">${esc(x.role)}</span> 申報轉讓 ${fmt(x.shares, 0)} 股</li>`).join("") || "<li class='muted'>需在本機連線到證交所</li>"}</ul>`;
    $("#mk-body").innerHTML = html;
  }

  // -------------------------------------------------------------------- alerts
  let audioCtx;
  function beep() {
    try { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); const o = audioCtx.createOscillator(), g = audioCtx.createGain(); o.connect(g); g.connect(audioCtx.destination); o.frequency.value = 880; g.gain.value = 0.08; o.start(); o.stop(audioCtx.currentTime + 0.18); } catch (e) { /* ignore */ }
  }
  function toast(title, body) {
    const el = document.createElement("div"); el.className = "toast"; el.innerHTML = `<b>${esc(title)}</b>${esc(body)}`; $("#toasts").appendChild(el); setTimeout(() => el.remove(), 9000);
  }
  function notifyNew(alerts) {
    const first = state.seenAlerts.size === 0;
    for (const a of alerts) {
      const key = a.time + a.title;
      if (state.seenAlerts.has(key)) continue;
      state.seenAlerts.add(key);
      if (first) continue;
      toast(a.title, a.body);
      if (state.settings.sound) beep();
      if (state.settings.desktop_notify && "Notification" in window && Notification.permission === "granted") new Notification(a.title, { body: a.body });
    }
  }
  $("#btn-enable").addEventListener("click", async () => { if ("Notification" in window) await Notification.requestPermission(); beep(); toast("提醒已開啟", "進入很便宜／很貴區、內部人買進、國會集群買進時會通知。"); });

  // ------------------------------------------------------------------ settings
  async function loadSettings() {
    const s = await api("/api/settings");
    state.settings = s;
    document.documentElement.dataset.conv = s.color_convention || "tw";
    const f = $("#settings-form");
    for (const el of f.elements) {
      if (!el.name) continue;
      const v = s[el.name];
      if (el.type === "checkbox") el.checked = !!v;
      else if (Array.isArray(v)) el.value = v.join(", ");
      else if (v !== undefined && v !== null) el.value = v;
    }
  }
  $("#settings-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = {};
    for (const el of e.target.elements) { if (!el.name) continue; data[el.name] = el.type === "checkbox" ? el.checked : el.value; }
    state.settings = await post("/api/settings", data);
    document.documentElement.dataset.conv = state.settings.color_convention || "tw";
    $("#settings-saved").hidden = false; setTimeout(() => { $("#settings-saved").hidden = true; }, 1500);
    setTimeout(refreshAll, 1500);
  });
  $("#btn-settings").addEventListener("click", () => { $("#settings").hidden = false; $("#research").hidden = true; });
  $("#settings-close").addEventListener("click", () => { $("#settings").hidden = true; });
  $("#btn-test").addEventListener("click", () => post("/api/test_alert", {}));
  $("#btn-refresh").addEventListener("click", async () => { await post("/api/refresh", {}); toast("重新抓資料", "價格與資訊流會在背景更新，約 1–3 分鐘。"); setTimeout(refreshAll, 20000); });
  $("#add-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const r = await post("/api/watchlist", { market: state.market, symbol: fd.get("symbol"), name: fd.get("name") });
    toast("已加入名單", `${r.symbol}，資料下載中…`); e.target.reset(); setTimeout(refreshAll, 15000);
  });
  $$(".mkt-tabs .tab").forEach((b) => b.addEventListener("click", () => { state.market = b.dataset.mkt; localStorage.setItem("mkt", state.market); $$(".mkt-tabs .tab").forEach((x) => x.setAttribute("aria-selected", x === b)); state.sym = null; refreshAll(); }));
  $$(".mkt-tabs .tab").forEach((x) => x.setAttribute("aria-selected", x.dataset.mkt === state.market));
  $("#sort").addEventListener("change", renderWatch); $("#filter").addEventListener("input", renderWatch);

  // ------------------------------------------------------------------ research
  async function openResearch() {
    $("#research").hidden = false; $("#settings").hidden = true;
    const r = await api("/api/research");
    const m = (r.markets || {})[state.market] || {};
    const pooled = m.pooled || {};
    const zrow = (tbl, z) => { const x = (tbl || {})[String(z)]; return x ? `<tr><td>${esc(ZW[String(z)])}</td><td class="num">${fmt(x.hit * 100, 0)}% (${fmt(x.base_hit * 100, 0)}%)</td><td class="num">${pct(x.excess_mean, 0)}</td>${["IS", "VAL", "TEST"].map((p) => `<td class="num">${pct((x.by_period || {})[p], 0)}</td>`).join("")}<td class="num muted">${x.n_sym}</td></tr>` : ""; };
    const tbl = (title, t) => `<h3>${esc(title)}</h3><table class="grid"><thead><tr><th>區間</th><th class="num">12 個月上漲機率 (基準)</th><th class="num">超額（全期）</th><th class="num">≤2012</th><th class="num">2013–18</th><th class="num">2019–</th><th class="num">檔數</th></tr></thead><tbody>${[-2, -1, 0, 1, 2].map((z) => zrow(t, z)).join("")}</tbody></table>`;
    let html = `<p>市場：<b>${state.market === "TW" ? "台股" : "美股"}</b>名單。每一天依位置分組，看之後 12 個月的報酬。「超額」＝該組平均報酬減全部日子的平均（百分點）。完整紀錄在 <code>docs/RESEARCH.md</code>。</p>`;
    html += tbl("五線譜（3.5 年回歸 ±σ）", (pooled.channel || {})["252"]);
    html += tbl("年線乖離（自己 5 年百分位）", (pooled.deviation || {})["252"]);
    if (pooled.pe) html += tbl("本益比河流圖（台股 2015–2025，證交所本益比）", pooled.pe["252"]);
    const tr = (pooled.trend || {})["252"] || {};
    html += `<h3>趨勢狀態</h3><table class="grid"><thead><tr><th>狀態</th><th class="num">12 個月上漲機率 (基準)</th><th class="num">超額</th><th class="num">≤2012</th><th class="num">2013–18</th><th class="num">2019–</th></tr></thead><tbody>${[["1", "趨勢向上"], ["0", "盤整"], ["-1", "趨勢向下"]].map(([k, l]) => { const x = tr[k]; return x ? `<tr><td>${l}</td><td class="num">${fmt(x.hit * 100, 0)}% (${fmt(x.base_hit * 100, 0)}%)</td><td class="num">${pct(x.excess_mean, 0)}</td>${["IS", "VAL", "TEST"].map((p) => `<td class="num">${pct((x.by_period || {})[p], 0)}</td>`).join("")}</tr>` : ""; }).join("")}</tbody></table>
      <p class="muted">趨勢向上「之後」的報酬反而較低——這批強勢股的下跌都是回頭看的買點。所以趨勢只拿來控風險，不當看多看空的理由。</p>`;
    const st = m.strategies || {};
    html += `<h3>當成進出場規則（每檔分別回測，中位數；年化差＝比買進持有多或少幾個百分點）</h3><table class="grid"><thead><tr><th>規則</th><th class="num">年化差 全期</th><th class="num">≤2012</th><th class="num">2013–18</th><th class="num">2019–</th><th class="num">回撤減少</th><th class="num">在場時間</th></tr></thead><tbody>` +
      Object.entries(st).filter(([k]) => k !== "買進持有").map(([k, v]) => `<tr><td>${esc(k)}</td>${["ALL", "IS", "VAL", "TEST"].map((p) => `<td class="num ${cls((v[p] || {}).median_excess_cagr)}">${pct((v[p] || {}).median_excess_cagr, 0)}</td>`).join("")}<td class="num">${fmt(((v.ALL || {}).median_dd_reduction || 0) * 100, 0)}%</td><td class="num">${fmt(((v.ALL || {}).median_time_in_market || 0) * 100, 0)}%</td></tr>`).join("") + `</tbody></table>
      <p class="muted">所有「出區間就空手」的規則都輸給買進持有。10 月均線是唯一值得保留的風險控管工具（回撤少 25–50%，代價每年 2–6 個百分點）。</p>`;
    const etf = m.etf || [];
    if (etf.length) html += `<h3>ETF：買進持有 vs 10 月均線（全期）</h3><table class="grid"><thead><tr><th>ETF</th><th class="num">年數</th><th class="num">買進持有 年化／回撤</th><th class="num">10 月均線 年化／回撤</th></tr></thead><tbody>${etf.map((e) => `<tr><td>${esc(e.sym)}</td><td class="num">${e.years}</td><td class="num">${pct(e.bh_cagr, 1)} ／ ${pct(e.bh_dd, 0)}</td><td class="num">${pct(e.faber_cagr, 1)} ／ ${pct(e.faber_dd, 0)}</td></tr>`).join("")}</tbody></table>`;
    const br = m.broad || {};
    if (br.channel) html += `<h3>全市場對照（${state.market === "TW" ? "1,548 檔台股上市普通股 2000–2019" : "1,247 檔 S&P 500 歷史成分股，含已下市"}）</h3><table class="grid"><thead><tr><th>五線譜區間</th><th class="num">12 個月上漲機率 (基準)</th><th class="num">≤2012</th><th class="num">2013–18</th><th class="num">2019–</th></tr></thead><tbody>${[-2, -1, 0, 1, 2].map((z) => { const x = br.channel[String(z)]; return x ? `<tr><td>${esc(ZW[String(z)])}</td><td class="num">${x.hit === null ? "—" : fmt(x.hit * 100, 0) + "% (" + fmt(x.base_hit * 100, 0) + "%)"}</td>${["IS", "VAL", "TEST"].map((p) => `<td class="num">${pct((x.excess_by_period || {})[p], 0)}</td>`).join("")}</tr>` : ""; }).join("")}</tbody></table>
      <p class="muted">名單裡都是今天的龍頭（活下來的贏家），所以上面的數字偏樂觀；換成所有曾經存在的股票，方向一樣、幅度小很多。</p>`;
    $("#research-body").innerHTML = html;
  }
  $("#btn-research").addEventListener("click", openResearch);
  $("#research-close").addEventListener("click", () => { $("#research").hidden = true; });

  // ---------------------------------------------------------------------- loop
  async function refreshAll() {
    try { await loadSettings(); await loadOverview(); await loadMarket(); } catch (e) { console.error(e); }
  }
  refreshAll();
  setInterval(async () => { try { await loadOverview(); await loadMarket(); } catch (e) { console.error(e); } }, 120000);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (chart) { chart.remove(); chart = null; loadChart(); } });
})();
