# 自動產生：EV 歷史紀錄器（獨立版，只讀 Firebase 的 EV_charger_data，不寫入、不刪除）
import sys, traceback
from datetime import datetime
from pathlib import Path
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

CRED = r"D:\260916數位孿生目前最終版計劃案代碼\essssimulation3-firebase-adminsdk-fbsvc-6b270aedbf.json"
URL = "https://essssimulation3-default-rtdb.asia-southeast1.firebasedatabase.app/"
NODE = "EV_charger_data"
HIST = Path(r"d:\260916數位孿生目前最終版計劃案代碼\outputs_EV電動車\EV_history.csv")
LOG = HIST.parent / "執行紀錄" / "EV_logger_排程.log"


def rows_of(snap):
    rows = []
    t = pd.to_datetime(snap.get("event_time"), errors="coerce")
    if pd.isna(t):
        return rows
    t = t.tz_localize(None) if t.tzinfo else t
    data, sett, dq = snap.get("data", {}) or {}, snap.get("charger_settings", {}) or {}, snap.get("data_quality", {}) or {}
    for ch, d in data.items():
        s, q = sett.get(ch, {}) or {}, dq.get(ch, {}) or {}
        rows.append({"event_time": t, "charger": ch, "name": d.get("name"),
                     "current_a": d.get("current_a"), "voltage_v": d.get("voltage_v"), "soc": d.get("soc"),
                     "soh": d.get("soh"), "max_temp_c": d.get("max_temp_c"),
                     "psu_current_a": s.get("psu_current_a"), "load_current_a": s.get("load_current_a"),
                     "output_on": s.get("output_on"), "battery_state": s.get("battery_state"),
                     "energy_flow": s.get("energy_flow"), "mode": s.get("mode"),
                     "dq_ok": q.get("ok"), "dq_issues": "; ".join(map(str, q.get("issues", []) or [])),
                     "message_id": snap.get("message_id")})
    return rows


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(CRED), {"databaseURL": URL})
    node = db.reference(NODE).get() or {}
    rows = []
    for v in node.values():
        if isinstance(v, dict) and "data" in v:
            rows += rows_of(v)
    new = pd.DataFrame(rows)
    old = pd.read_csv(HIST, parse_dates=["event_time"]) if HIST.exists() else pd.DataFrame()
    allr = pd.concat([old, new], ignore_index=True)
    if allr.empty:
        msg = "沒有資料"
    else:
        allr = allr.drop_duplicates(["event_time", "charger"]).sort_values(["charger", "event_time"])
        tmp = HIST.with_suffix(".tmp")
        allr.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(HIST)
        latest = pd.to_datetime(allr["event_time"]).max()
        age_h = (datetime.now() - latest).total_seconds() / 3600
        msg = f"共 {len(allr)} 列（新增 {len(allr) - len(old)}），最新 {latest:%m-%d %H:%M}"
        if age_h > 3:
            msg += f"｜⚠ 資料源已 {age_h:.1f} 小時沒更新"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 失敗：{traceback.format_exc()}\n")
        sys.exit(1)
