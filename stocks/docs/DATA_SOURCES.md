# 資料來源調查（2026-11-08）

圖例：✅ 已實際取回資料並確認欄位／最新日期；⚠️ 開發環境被 robots.txt／WAF 擋住（端點本身可能正常），請在本機第一天確認；❌ 已確認失效。

> 注意：`www.twse.com.tw`（legacy JSON）與 `openapi.twse.com.tw/v1/fund/T86` 對雲端 IP／非瀏覽器 UA 回「FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED」。在家用電腦上請帶 Chrome UA 與 `Referer: https://www.twse.com.tw/`。

## 1. 美股價格與基本面

| 用途 | URL | 認證 | 限制／備註 | 驗證 |
|---|---|---|---|---|
| 日線 OHLCV（主力） | `https://query2.finance.yahoo.com/v8/finance/chart/AAPL?range=1y&interval=1d` | 無；需瀏覽器 UA | chart v8 不需 crumb；高頻會 429；`range=max` 可拿 1980 起 | ⚠️（yfinance 1.8.0 仍以此端點為主） |
| quoteSummary（估值／統計） | `https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL?modules=price,summaryDetail,defaultKeyStatistics,financialData` | 需 cookie＋crumb | yfinance 已封裝 | ⚠️ |
| yfinance 套件 | PyPI `yfinance` 1.8.0（2026-11-03），Apache-2.0 | — | 預設用 curl_cffi 偽裝 Chrome 以避 429；維護活躍 | ✅ |
| 備援日線 CSV | `https://stooq.com/q/d/l/?s=aapl.us&i=d` | 無 | 單 IP 每日下載上限；只做 fallback | ⚠️ |
| XBRL 財報（全部歷史） | `https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json` | 必須帶 `User-Agent: name email` | 10 req/s；AAPL 檔 3.5MB；最新 fact 2026-07 Q3 | ✅ |
| 公司最近申報 | `https://data.sec.gov/submissions/CIK0000320193.json` | 同上 | `filings.recent.form[]` | ✅ |
| FRED 無 key CSV | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10`（多序列 `id=DGS10,DGS2,CPIAUCSL,UNRATE,FEDFUNDS,T10Y2Y`） | 無 | `DATE,DGS10` CSV | ⚠️ |
| ETF 持股（美） | Invesco QQQ：`https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ`；iShares／SSGA 各有 CSV／xlsx | 無 | 路徑偶爾改 | 未驗 |

## 2. 台股

### 2a. TWSE OpenAPI（`https://openapi.twse.com.tw/v1/...`，JSON array，無認證；皆為最近一個交易日快照，無歷史參數 → 必須每日落地存檔）

| 用途 | 路徑 | 欄位／備註 | 驗證 |
|---|---|---|---|
| 全市場日收盤 | `/exchangeReport/STOCK_DAY_ALL` | `Date,Code,Name,TradeVolume,TradeValue,OpeningPrice,HighestPrice,LowestPrice,ClosingPrice,Change,Transaction`；Date 為民國年 `1151106` | ✅ 2026-11-06 |
| 本益比、殖利率、股價淨值比 | `/exchangeReport/BWIBBU_ALL` | `Date,Code,Name,PEratio,DividendYield,PBratio` | ✅ |
| 融資融券餘額 | `/exchangeReport/MI_MARGN` | 無 Date 欄，需自己蓋抓取日 | ✅ |
| 月營收 | `/opendata/t187ap05_L`（上市） | 出表日期,資料年月,公司代號,公司名稱,產業別,當月營收,上月營收,去年當月營收,MoM%,YoY%,累計… | ✅ |
| 季財報 | `/opendata/t187ap06_L_ci`（綜合損益）、`/opendata/t187ap07_L_ci`（資產負債） | 當季全市場 | ✅ swagger |
| 股利 | `/opendata/t187ap45_L`；除權息預告 `/exchangeReport/TWT48U_ALL` | | ✅ swagger |
| 外資持股比例 | `/fund/MI_QFIIS_cat`、`/fund/MI_QFIIS_sort_20` | OpenAPI 沒有 T86 | ✅ swagger |
| 董監持股／內部人 | `/opendata/t187ap11_L` 董監持股餘額；`/opendata/t187ap12_L` 內部人轉讓事前申報；`/opendata/t187ap13_L`（非轉讓）；`/opendata/t187ap02_L` 持股 >10% 大股東；`/opendata/t187ap09_L` 董監質押 | t187ap12_L：申報人身分,姓名,預定轉讓方式及股數,目前持有股數,預定轉讓後持股,有效轉讓期間 | ✅ |
| 重大訊息／公告 | `/opendata/t187ap04_L`；`/news/newsList`；`/announcement/notice`（注意股）、`/announcement/punish`（處置股） | | ✅ swagger |
| 指數 | `/exchangeReport/MI_INDEX`、`/indicesReport/MI_5MINS_HIST`（加權指數歷史） | | ✅ swagger |
| 休市日 | `/holidaySchedule/holidaySchedule` | | ✅ swagger |
| ETF 基本資料 | `/opendata/t187ap47_L` | OpenAPI 無 ETF 淨值、無持股 | ✅ swagger |

### 2b. TWSE legacy JSON（歷史回補；`www.twse.com.tw`，WAF 擋雲端）

| 用途 | URL | 備註 |
|---|---|---|
| 個股日 K（一個月／次） | `https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20261101&stockNo=2330&response=json` | 一次只給該月；同 IP 連續請求會被封幾分鐘，慣例 sleep 3–5 秒 |
| 三大法人（個股日） | `https://www.twse.com.tw/rwd/zh/fund/T86?date=20261106&selectType=ALLBUT0999&response=json` | 一天一次全市場 |
| 三大法人（市場合計） | `.../rwd/zh/fund/BFI82U?dayDate=20261106&type=day&response=json` | |
| 本益比日（歷史） | `.../rwd/zh/afterTrading/BWIBBU_d?date=20261106&selectType=ALL&response=json` | |

### 2c. TPEx（上櫃）：`https://www.tpex.org.tw/openapi/`（`tpex_mainboard_daily_close_quotes`、`tpex_mainboard_peratio_analysis`、`tpex_3insti_daily_trading`、`mopsfin_t187ap05_O`）— 整個 host 擋雲端 IP，請本機驗證。

### 2d. FinMind（`https://api.finmindtrade.com/api/v4/data`）：匿名 300 req/hr、免費註冊 600 req/hr；`TaiwanStockInstitutionalInvestorsBuySell`（✅ 回 `date,stock_id,buy,name,sell`，匿名約延遲一週）、`TaiwanStockPrice`、`TaiwanStockPER`、`TaiwanStockMonthRevenue`、`TaiwanStockMarginPurchaseShortSale`、`TaiwanStockShareholding`、`TaiwanStockHoldingSharesPer`。4xx 連發會被 ban 30 分鐘。

### 2e. Yahoo `.TW`／`.TWO`：2330.TW 約自 2000、0050.TW 自 2003/06、006208 自 2012、00878 自 2020/07。成交量單位為股；調整價偶有缺漏，以 TWSE 為真、Yahoo 為補。

### 2f. 台灣 ETF 淨值／持股：OpenAPI 無。元大 `https://www.yuantaetfs.com/product/detail/0050/ratio`；國泰 `cathaysite.com.tw`；MoneyDJ `https://www.moneydj.com/ETF/X/Basic/Basic0007a.xdjhtm?etfid=0050.TW`。皆為 HTML。

## 3. 美國國會交易揭露

| 來源 | URL | 狀態／作法 | 驗證 |
|---|---|---|---|
| House Clerk（官方，首選） | 年度索引 `https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2026FD.zip`（含 `2026FD.txt` tab 分隔＋XML；欄位 `Prefix,Last,First,Suffix,FilingType,StateDst,Year,FilingDate,DocID`，PTR = `FilingType=P`）→ PDF `https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/{DocID}.pdf` | 無 key；8 位數 DocID 為電子件有文字層（pdfplumber 可解）；7 位數為掃描件；交易日期只在 PDF 內 | ✅ 入口頁 |
| Senate eFD（官方） | `https://efdsearch.senate.gov/search/` | ① GET `/search/home/` 取 csrftoken；② POST `/search/home/` 帶 `prohibition_agreement=1`；③ POST `/search/report/data/`（DataTables：`report_types=[11]` PTR、`filer_types=[1]`、`submitted_start_date`）→ rows 含 `/search/view/ptr/<uuid>/`；④ 解析 HTML 表格 | 公開已知流程 |
| House／Senate Stock Watcher S3 | `house-stock-watcher-data.s3-...`、`senate-stock-watcher-data.s3-...` | ❌ 皆回 403；GitHub `timothycarambat/senate-stock-watcher-data` 最後更新 2021-03（可當 2012–2021 歷史） | ❌ |
| Capitol Trades | `https://www.capitoltrades.com/trades`；舊 BFF `https://bff.capitoltrades.com/trades?page=1&pageSize=50` | HTML 可爬（PyPI `capitolscraper` 0.1.1，2026-10） | ⚠️ |
| Quiver Quant／Unusual Whales | API 需付費 | 不採用 | — |
| Finnhub | `https://finnhub.io/api/v1/stock/congressional-trading?symbol=AAPL&token=KEY`、`/stock/insider-transactions`、`/calendar/earnings` | 免費 key，60 req/min | 未驗 |

開源參考：`kadoa-org/congress-trading-monitor`（220★, MIT, 2026-11）、`timothycarambat/senate-stock-watcher-data`（101★, 2021）、`johnisanerd/Apify-Congressional-Trading-Data-Scraper`、`seralifatih/congress-trading-pipeline`、`TommasoAmici/capitoltrades`（BFF client）。

## 4. 內部人與基金

| 用途 | URL | 備註 | 驗證 |
|---|---|---|---|
| SEC EFTS 全文檢索（Form 4，首選） | `https://efts.sec.gov/LATEST/search-index?q=%22Apple%20Inc%22&forms=4&dateRange=custom&startdt=2026-10-01&enddt=2026-11-08` | JSON `hits.hits[]._source{ciks,display_names,file_date,form,period_ending,adsh}`；再用 adsh 組 `https://www.sec.gov/Archives/edgar/data/320193/{adsh去-}/` 取 XML | ✅ |
| SEC 最新 Form 4 Atom | `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&count=100&output=atom` | | ⚠️ |
| OpenInsider | `http://openinsider.com/screener?s=AAPL&fd=365&td=0&xp=1&xs=1&cnt=1000&page=1` | HTML `table.tinytable`，`pandas.read_html`；約 1 req/2s | ⚠️ |
| 13F | SEC EFTS `forms=13F-HR`；季度資料集 `https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets`；Dataroma `https://www.dataroma.com/m/holdings.php?m=BRK` | | ✅ host |
| Python 套件 | `edgartools`（2.8k★, MIT, 2026-11）：Form 4／13F／companyfacts 解析 | | ✅ |
| 台灣對應 | 董監／內部人：OpenAPI t187ap11/12/13/02；法人：T86（legacy）或 FinMind；外資持股 `/fund/MI_QFIIS_*`；股權分散：TDCC `https://opendata.tdcc.com.tw/getOD.ashx?id=1-5`（每週 CSV） | | ✅／⚠️ |

## 5. 新聞與行事曆

| 用途 | URL | 備註 | 驗證 |
|---|---|---|---|
| 鉅亨網 API（台股新聞，首選） | `https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=30&page=1`（其他：`headline`、`us_stock`、`tw_macro`、`wd_stock`） | JSON `items.data[]{newsId,title,summary,publishAt,categoryName,stock[],keyword}`；文章 `https://news.cnyes.com/news/id/{newsId}` | ✅ 2026-11-08 |
| Yahoo Finance RSS | `https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,NVDA&region=US&lang=en-US`（台股 `s=2330.TW`） | 條數少 | ⚠️ |
| Google News RSS | `https://news.google.com/rss/search?q=NVDA+stock+when:7d&hl=en-US&gl=US&ceid=US:en`；`...q=台積電&hl=zh-TW&gl=TW&ceid=TW:zh-Hant` | 最穩的免費彙整 | ⚠️ |
| CNBC RSS | `https://www.cnbc.com/id/19854910/device/rss/rss.html`（Technology） | 可能擋雲端 | ⚠️ |
| 台灣 RSS | 中央社 `https://feeds.feedburner.com/rsscna/finance`；經濟日報 `https://money.udn.com/rssfeed/news/1001/5591?ch=money`；Yahoo 奇摩股市 `https://tw.stock.yahoo.com/rss?category=tw-market` | | ⚠️ |
| 財報日 | yfinance `get_earnings_dates()`；Nasdaq `https://api.nasdaq.com/api/calendar/earnings?date=2026-11-10` | | ⚠️ |
| FOMC 日程 | `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`；聲明 RSS `https://www.federalreserve.gov/feeds/press_monetary.xml` | | ⚠️ |
| 台股月營收期限 | 每月 10 日前公告上月營收；用 t187ap05_L 的「出表日期」偵測更新 | | ✅ |

## 6. 值得借用的開源專案

| 專案 | 借什麼 |
|---|---|
| `ranaroussi/yfinance`（25.9k★） | 美股＋台股價格層 |
| `FinMind/FinMind`（2.9k★） | 台股 dataset client；`crawler/` 裡的 TWSE／TPEx／MOPS 爬蟲（URL、headers、sleep） |
| `mlouielu/twstock`（1.5k★） | TWSE STOCK_DAY 逐月回補＋3 秒節流 |
| `tradingview/lightweight-charts`（17.8k★） | K 線＋疊帶（五線譜／河流圖） |
| `dgunning/edgartools`（2.8k★） | Form 4、13F、companyfacts 解析 |
| `kadoa-org/congress-trading-monitor`（220★） | 國會交易 ETL 與資料模型 |
| `sd3v/openinsiderData`、`btopn/OpenInsider-MCP` | OpenInsider 參數與解析 |
| `wakin61043/tw-trading-dashboard`（49★） | 自架台股儀表板的資料落地方式 |
| `OpenBB-finance/OpenBB`（AGPL） | 只借 provider 抽象的想法，不混入程式碼 |
| 樂活五線譜／本益比河流圖 | GitHub 只有小專案；自己實作（五線譜＝對數線性回歸 ±1σ／2σ，預設 3.5 年；河流圖＝近 4 季 EPS × 本益比分帶） |
