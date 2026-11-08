"""Parsers that must work offline: TWSE ROC dates, Form 4 XML, FOMC calendar HTML, congress summaries."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stocksignal.data.congress import CongressFeed  # noqa: E402
from stocksignal.data.macro import parse_fomc, tw_revenue_deadline  # noqa: E402
from stocksignal.data.news import label, parse_cnyes, parse_rss, score_headline  # noqa: E402
from stocksignal.data.sec import parse_form4_xml  # noqa: E402
from stocksignal.data.twse import roc_to_date  # noqa: E402

FORM4 = """<?xml version="1.0"?>
<ownershipDocument><issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA CORP</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerCik>0001</rptOwnerCik><rptOwnerName>HUANG JEN HSUN</rptOwnerName></reportingOwnerId>
<reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>1</isOfficer><officerTitle>President and CEO</officerTitle></reportingOwnerRelationship></reportingOwner>
<nonDerivativeTable>
<nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle><transactionDate><value>2026-10-20</value></transactionDate>
<transactionCoding><transactionFormType>4</transactionFormType><transactionCode>S</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>120000</value></transactionShares><transactionPricePerShare><value>181.25</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
<postTransactionAmounts><sharesOwnedFollowingTransaction><value>75000000</value></sharesOwnedFollowingTransaction></postTransactionAmounts></nonDerivativeTransaction>
<nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle><transactionDate><value>2026-10-21</value></transactionDate>
<transactionCoding><transactionCode>M</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>1</value></transactionShares><transactionPricePerShare><value>1</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
</nonDerivativeTable></ownershipDocument>"""


def test_roc_dates():
    assert roc_to_date("1151106") == pd.Timestamp("2026-11-06")
    assert roc_to_date("115/11/06") == pd.Timestamp("2026-11-06")
    assert roc_to_date("") is None and roc_to_date("abc") is None


def test_form4_parse_keeps_only_open_market():
    rows = parse_form4_xml(FORM4, filed="2026-10-22")
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "NVDA" and r["code"] == "S" and r["type"] == "賣出"
    assert r["shares"] == 120000 and abs(r["value"] - 120000 * 181.25) < 1e-6
    assert "CEO" in r["role"]


def test_fomc_parse():
    html = """<div class="panel panel-default"><div class="panel-heading"><h4>2026 FOMC Meetings</h4></div>
    <div class="fomc-meeting"><div class="fomc-meeting__month col-xs-5 col-sm-3 col-md-2"><strong>January</strong></div>
    <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-1">27-28</div></div>
    <div class="fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>December</strong></div>
    <div class="fomc-meeting__date col-xs-4">8-9*</div></div></div>"""
    d = parse_fomc(html)
    assert [x.isoformat() for x in d] == ["2026-01-28", "2026-12-09"]


def test_tw_revenue_deadline():
    from datetime import date
    assert tw_revenue_deadline(date(2026, 11, 3)) == date(2026, 11, 10)
    assert tw_revenue_deadline(date(2026, 11, 11)) == date(2026, 12, 10)
    # 2027-01-10 is a Sunday -> Monday 11th
    assert tw_revenue_deadline(date(2026, 12, 20)) == date(2027, 1, 11)


def test_news_parsers_and_sentiment():
    rss = """<rss><channel><item><title>Nvidia surges to record high after beating estimates</title><link>http://x</link>
    <pubDate>Fri, 06 Nov 2026 14:00:00 GMT</pubDate></item></channel></rss>"""
    items = parse_rss(rss, "test")
    assert items[0]["title"].startswith("Nvidia") and items[0]["time"] is not None
    s, cues = score_headline(items[0]["title"])
    assert s > 0.5 and label(s) == "利多"
    s2, _ = score_headline("台積電法說會後重挫 外資大賣超")
    assert s2 < -0.5 and label(s2) == "利空"
    js = {"items": {"data": [{"newsId": 1, "title": "鴻海 10 月營收年增 20% 創同期新高", "publishAt": 1762500000, "market": [{"key": "2317"}]}]}}
    it = parse_cnyes(js, "cnyes")
    assert it[0]["link"].endswith("/1") and it[0]["stocks"] == ["2317"]


def test_congress_summary_cluster(tmp_path):
    feed = CongressFeed(tmp_path)
    rows = []
    for i, who in enumerate(["a", "b", "c", "d"]):
        rows.append({"id": f"t{i}", "filer_id": who, "filer_name": who.upper(), "ticker": "NVDA", "transaction_type": "Purchase",
                     "filing_date": pd.Timestamp.now().normalize() - pd.Timedelta(days=5 + i), "transaction_date": pd.Timestamp.now().normalize() - pd.Timedelta(days=20),
                     "amount_range_low": 1001, "amount_range_high": 15000, "amount_range_label": "$1,001 - $15,000", "chamber": "house", "party": "R",
                     "source_id": "house_clerk", "state": "TX", "branch": "congress", "owner": "SP", "asset_name": "NVIDIA", "asset_type": "ST",
                     "days_to_file": 10, "is_late": 0, "doc_url": "http://doc"})
    feed.df = feed._clean(pd.DataFrame(rows))
    s = feed.summary("NVDA", 90)
    assert s["buys"] == 4 and s["buyers"] == 4 and s["cluster"] is True and s["sells"] == 0
    assert feed.summary("AAPL")["buys"] == 0


def test_t86_parse(tmp_path):
    from stocksignal.data.twse import TWSELegacy
    js = {"stat": "OK", "fields": ["證券代號", "證券名稱", "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
                                   "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數", "投信買進股數", "投信賣出股數", "投信買賣超股數", "自營商買賣超股數",
                                   "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)", "自營商買進股數(避險)", "自營商賣出股數(避險)",
                                   "自營商買賣超股數(避險)", "三大法人買賣超股數"],
          "data": [["2330", "台積電", "1,000", "500", "500", "0", "0", "0", "100", "50", "50", "10", "5", "0", "5", "5", "0", "5", "560"]]}
    t = TWSELegacy(tmp_path)
    t._get = lambda path, params: js
    r = t.t86_day(pd.Timestamp("2026-11-06")).to_dict("records")[0]
    assert (r["foreign"], r["trust"], r["dealer"], r["total"]) == (500, 50, 10, 560)
