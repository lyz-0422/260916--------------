"""
ems_realtime_lib.py — 即時監測 / 快速恢復用的獨立模組

用途：kernel 重開之後，不用把整份 notebook（含歷史 CSV 載入、模型訓練那些
很花時間的區塊）從頭跑一次，只要：

    from ems_realtime_lib import *
    model_bundle = joblib.load(MODEL_BUNDLE_PATH)
    ref = init_firebase(FIREBASE_CRED_JSON_PATH)
    run_realtime_loop(ref, model_bundle=model_bundle, poll_interval_sec=60,
                       align_to_clock=True, align_second_offset=10,
                       max_iterations=None, upload_to_firebase=True)

★ 這個檔案要跟 notebook 放在同一個資料夾，且 MODEL_BUNDLE_PATH 指到的
  ems_model_bundle.joblib 必須已經用 notebook 的【階段一】訓練並存檔過一次。
  訓練這件事本來就只需要做一次，不需要每次重開 kernel 都重跑。

★ 這個檔案的內容跟 notebook 裡對應的 cell 是同步抽出來的；如果之後在
  notebook 裡改了 Modbus 設定或即時輪詢邏輯，記得同步更新這個檔案
  （或請 Claude 幫忙重新同步）。
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import time
import struct
import math
import json
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Firebase 設定與函式
# ============================================================
FIREBASE_CRED_JSON_PATH = r"C:\path\to\your\firebase-service-account.json"  # ← 改成你自己的金鑰路徑
FIREBASE_DATABASE_URL = 'https://essssimulation3-default-rtdb.asia-southeast1.firebasedatabase.app/'

def init_firebase(cred_json_path=None, database_url=FIREBASE_DATABASE_URL):
    """
    本機版初始化：不再跳出瀏覽器上傳視窗，改成直接讀取本機的金鑰 JSON 檔路徑。
    """
    print("\n=== Firebase 初始化 ===")
    import firebase_admin
    from firebase_admin import credentials, db

    try:
        app = firebase_admin.get_app()
        print("⚠ Firebase 已經初始化，刪除舊連接...")
        firebase_admin.delete_app(app)
    except ValueError:
        pass

    if cred_json_path is None:
        cred_json_path = FIREBASE_CRED_JSON_PATH

    cred_path = Path(cred_json_path)
    if not cred_path.exists():
        raise FileNotFoundError(
            f"找不到 Firebase 金鑰檔案: {cred_path}\n"
            f"請先把 FIREBASE_CRED_JSON_PATH 改成你實際的金鑰 JSON 檔案路徑。"
        )

    with open(cred_path, 'r', encoding='utf-8') as f:
        cred_data = json.load(f)
        project_id = cred_data.get('project_id', 'your-project-id')

    cred = credentials.Certificate(str(cred_path))

    print(f"📊 專案 ID: {project_id}")
    print(f"🔗 資料庫 URL: {database_url}")

    firebase_admin.initialize_app(cred, {'databaseURL': database_url})
    print("✓ Firebase 連接成功!")
    return db.reference()


def clear_firebase_data(ref):
    print("\n=== 清除Firebase舊數據 ===")
    all_data = ref.get()
    if all_data is None:
        print("✓ Firebase 中沒有數據")
        return
    nodes = list(all_data.keys()) if isinstance(all_data, dict) else []
    if nodes:
        for node in nodes:
            ref.child(node).delete()
        print(f"✓ 已刪除 {len(nodes)} 個舊節點")


def upload_to_firebase(ref, df, batch_size=500):
    """
    批次寫入：先在記憶體組好「相對於 ems_data 節點」的巢狀字典
    （key 用 'date/time' 表示路徑），再分批呼叫 ems_data_ref.update(batch)，
    一次送出一批，大幅減少 HTTP 往返次數。
    """
    print("\n=== 上傳數據到 Firebase（批次寫入）===")
    COLS = ['ess_power', 'soc', 'soh', 'cell_v_diff', 'self_discharge_rate',
            'contract_capacity', 'before_load', 'after_load', 'over_contract',
            'is_anomaly', 'anomaly_type']

    ems_data_ref = ref.child('ems_data')
    total = len(df)
    batch = {}
    uploaded = 0

    for _, row in df.iterrows():
        record = {'timestamp': row['timestamp'].isoformat()}
        for col in COLS:
            record[col] = row[col] if col in row.index else 0
        batch[f"{row['date']}/{row['time']}"] = record

        if len(batch) >= batch_size:
            ems_data_ref.update(batch)
            uploaded += len(batch)
            print(f"  ✓ 已上傳 {uploaded}/{total} 筆")
            batch = {}

    if batch:
        ems_data_ref.update(batch)
        uploaded += len(batch)
        print(f"  ✓ 已上傳 {uploaded}/{total} 筆")

    print(f"✓ 上傳完成，共 {uploaded} 筆")

    first_date = df['date'].iloc[0]
    first_time = df['time'].iloc[0]
    ref.update({'viewerDate': first_date, 'viewerTime': first_time})
    print(f"✓ 已初始化 viewerDate={first_date}, viewerTime={first_time}")


def upload_single_record_to_firebase(ref, record: dict):
    """
    階段二即時推論用：只上傳『這一筆』最新資料，並同步更新 viewer 指標。

    ★ 修正：record['timestamp'] 是 datetime 物件，Firebase REST API 底層用
      json.dumps 序列化時不認得 datetime，會丟 TypeError。這裡另外複製一份
      要上傳的字典，把 timestamp 轉成字串，不動原本 record（buffer 裡的
      resample 還是需要真正的 datetime）。
    """
    date_key = record['date']
    time_key = record['time']

    record_to_send = dict(record)
    ts = record_to_send.get('timestamp')
    if hasattr(ts, 'isoformat'):
        record_to_send['timestamp'] = ts.isoformat()

    ref.child('ems_data').child(date_key).child(time_key).set(record_to_send)
    ref.update({'viewerDate': date_key, 'viewerTime': time_key})

# ============================================================
# 特徵工程（infer_anomaly 需要）
# ============================================================

def engineer_features_advanced(df, window_sizes=[4, 8, 12, 24]):
    print(f"\n=== 進階特徵工程 ===")
    print(f"  滑動視窗大小: {window_sizes}")
    df_p = df.copy()

    df_p['hour']         = df_p['timestamp'].dt.hour
    df_p['day_of_week']  = df_p['timestamp'].dt.dayofweek
    df_p['is_weekend']   = (df_p['day_of_week'] >= 5).astype(int)
    df_p['is_peak_hour'] = ((df_p['hour'] >= 9) & (df_p['hour'] <= 17)).astype(int)
    df_p['is_night']     = ((df_p['hour'] >= 22) | (df_p['hour'] <= 6)).astype(int)
    df_p['is_daytime']   = ((df_p['hour'] >= 6)  & (df_p['hour'] <= 18)).astype(int)
    df_p['hour_sin']     = np.sin(2*np.pi*df_p['hour']/24)
    df_p['hour_cos']     = np.cos(2*np.pi*df_p['hour']/24)
    df_p['day_sin']      = np.sin(2*np.pi*df_p['day_of_week']/7)
    df_p['day_cos']      = np.cos(2*np.pi*df_p['day_of_week']/7)

    df_p['cell_v_diff_ma']   = df_p['cell_v_diff'].rolling(4,  min_periods=1).mean()
    df_p['cell_v_diff_std']  = df_p['cell_v_diff'].rolling(8,  min_periods=1).std().fillna(0)
    df_p['cell_v_diff_max']  = df_p['cell_v_diff'].rolling(12, min_periods=1).max()
    df_p['cell_v_high_flag'] = (df_p['cell_v_diff'] > 0.05).astype(int)

    df_p['self_dis_ma']   = df_p['self_discharge_rate'].rolling(4, min_periods=1).mean()
    df_p['self_dis_std']  = df_p['self_discharge_rate'].rolling(8, min_periods=1).std().fillna(0)
    df_p['self_dis_high'] = (df_p['self_discharge_rate'] > 0.05).astype(int)

    df_p['soh_diff']        = df_p['soh'].diff().fillna(0)
    df_p['soh_low_flag']    = (df_p['soh'] < 80).astype(int)
    df_p['soh_soc_interact'] = df_p['soh'] * df_p['soc'] / 10000.0
    df_p['soc_change_rate'] = df_p['soc'].diff().fillna(0) / 0.25
    df_p['dv_dq_proxy']     = df_p['soc_change_rate'].abs() / (df_p['ess_power'].abs() + 1e-6)

    df_p['soc_diff']    = df_p['soc'].diff().fillna(0)
    df_p['soc_diff2']   = df_p['soc_diff'].diff().fillna(0)
    df_p['power_diff']  = df_p['ess_power'].diff().fillna(0)
    df_p['power_diff2'] = df_p['power_diff'].diff().fillna(0)
    df_p['load_diff']   = df_p['before_load'].diff().fillna(0)

    for w in window_sizes:
        df_p[f'soc_mean_{w}']   = df_p['soc'].rolling(w, min_periods=1).mean()
        df_p[f'soc_std_{w}']    = df_p['soc'].rolling(w, min_periods=1).std().fillna(0)
        df_p[f'soc_range_{w}']  = (df_p['soc'].rolling(w, min_periods=1).max() -
                                    df_p['soc'].rolling(w, min_periods=1).min())
        df_p[f'power_mean_{w}'] = df_p['ess_power'].rolling(w, min_periods=1).mean()
        df_p[f'power_std_{w}']  = df_p['ess_power'].rolling(w, min_periods=1).std().fillna(0)
        df_p[f'load_std_{w}']   = df_p['before_load'].rolling(w, min_periods=1).std().fillna(0)

    df_p['power_to_load_ratio']   = df_p['ess_power']    / (df_p['before_load'] + 1e-6)
    df_p['soc_power_interaction'] = df_p['soc']          * df_p['ess_power']
    df_p['load_reduction_rate']   = (df_p['before_load'] - df_p['after_load']) / (df_p['before_load'] + 1e-6)
    df_p['contract_violation']    = (df_p['after_load']  > df_p['contract_capacity']).astype(int)
    df_p['over_contract_ratio']   = df_p['over_contract'] / (df_p['contract_capacity'] + 1e-6)
    df_p['soc_extreme']           = ((df_p['soc'] < 10) | (df_p['soc'] > 95)).astype(int)
    df_p['power_extreme']         = (np.abs(df_p['ess_power']) > 200).astype(int)
    df_p['power_volatility']      = df_p['ess_power'].rolling(8, min_periods=1).std().fillna(0)

    for lag in [1, 2, 4]:
        df_p[f'soc_lag_{lag}']   = df_p['soc'].shift(lag).bfill()
        df_p[f'power_lag_{lag}'] = df_p['ess_power'].shift(lag).bfill()

    print(f"✓ 特徵工程完成，總欄數: {df_p.shape[1]}")
    return df_p


SYNTHETIC_ONLY_RAW_COLS = ['soh', 'cell_v_diff', 'self_discharge_rate']

SYNTHETIC_ONLY_ENGINEERED_COLS = [
    'cell_v_diff_ma', 'cell_v_diff_std', 'cell_v_diff_max', 'cell_v_high_flag',
    'self_dis_ma', 'self_dis_std', 'self_dis_high',
    'soh_diff', 'soh_low_flag', 'soh_soc_interact',
]
# 注意：soc_change_rate / dv_dq_proxy 只用到 soc 與 ess_power（都是真實欄位），
# 不放進排除清單。


def select_features_for_if(df_processed, use_synthetic_diagnostic_features=True):
    """
    use_synthetic_diagnostic_features:
      True（預設）：使用全部特徵，含模擬電池診斷特徵。
      False（對照實驗）：排除模擬電池診斷特徵（soh, cell_v_diff,
      self_discharge_rate 及其衍生欄位），只留下由真實現場欄位
      （before_load/after_load/contract_capacity/ess_power/soc）
      衍生的特徵，得到不依賴模擬感測值的基準指標。
    """
    exclude_cols = ['timestamp', 'date', 'time', 'is_anomaly', 'anomaly_type']
    if not use_synthetic_diagnostic_features:
        exclude_cols = exclude_cols + SYNTHETIC_ONLY_RAW_COLS + SYNTHETIC_ONLY_ENGINEERED_COLS

    feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
    mode_str = "含模擬電池診斷特徵（原始模式）" if use_synthetic_diagnostic_features else "僅真實欄位衍生特徵（對照實驗基準）"
    print(f"  選擇特徵數: {len(feature_cols)} ({mode_str})")
    return feature_cols


def clean_nan_inf(df_processed, feature_cols):
    """
    NaN / Inf 安全網。
    在特徵工程之後、切 X/y 之前呼叫，先用 ffill/bfill 補值，
    仍殘留 NaN 的整列直接捨棄（極少數，通常是資料開頭的邊界情況）。
    """
    n_before = len(df_processed)
    df_clean = df_processed.copy()
    df_clean[feature_cols] = (
        df_clean[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
    )
    df_clean = df_clean.dropna(subset=feature_cols).reset_index(drop=True)
    n_after = len(df_clean)
    if n_after < n_before:
        print(f"  ⚠ NaN/Inf 安全網：捨棄 {n_before - n_after} 筆殘留缺失值的資料")
    else:
        print(f"  ✓ NaN/Inf 安全網：無殘留缺失值")
    return df_clean

# ============================================================
# 已訓練模型檔路徑
# ============================================================

MODEL_BUNDLE_PATH = str(Path.cwd() / "ems_model_bundle.joblib")


# ============================================================
# Modbus 端點設定 / 解碼 / 連線
# ============================================================

from pymodbus.client import ModbusTcpClient

# ============================================================
# 現場 Modbus TCP 端點設定
# ============================================================
MAIN_METER_CFG = {'name': '主表',     'ip': '192.168.1.202', 'port': 502,  'unit_id': 1}
ESS_METER_CFG  = {'name': '儲能電表', 'ip': '192.168.1.203', 'port': 502,  'unit_id': 1}
EMS_CFG        = {
    'name': 'EMS', 'ip': '192.168.1.204', 'port': 2000, 'unit_id': 1,
    # ⚠ 現場實測：TCP 連得上，但標準 Modbus TCP 請求逾時無回應；
    #   RTU framer 測試時設備會在 ~25ms 內主動關閉連線（代表設備有在解析封包，
    #   但拒絕/不認得這組請求）。在排查出正確 unit_id / 位址前先停用。
    #   要重新啟用時，把這裡改成 True 即可，其餘程式碼不用動。
    'enabled': False,  # ← 現場仍讀不到，先停用；EMS 相關程式碼原封不動保留，之後排查好只要改這裡成 True
}

CONTRACT_CAPACITY_KW = 250.0  # 現場固定契約容量

# --- Acuvim II 電表：FC03, 0x4022~0x4023, float32_be，輸出單位是 W ---
# 已用現場診斷驗證：主表 88.129 kW（registers=[18348, 8267]）、
# 儲能電表 -0.637 kW（registers=[50207, 20008]），跟現場儀表板數字吻合。
METER_POWER_SPEC = {
    'function': 3,      # FC03 read_holding_registers
    'address': 0x4022,
    'count': 2,
    'scale': 0.001,     # 暫存器輸出是 W，*0.001 轉成 kW
    'encoding': 'float32_be',
}

# --- EMS（KAC50DP 第一層）：FC03, 0x0147 = 總有效功率 (×0.1kW, signed)，
#     0x301A / 0x301B = SOC / SOH (×0.1%, unsigned) ---
# 尚未在現場驗證成功；EMS_CFG['enabled']=True 後才會實際被呼叫。
EMS_KAC50DP_LEVEL1_PROFILE = {
    'power': {'function': 3, 'address': 0x0147, 'count': 1, 'scale': 0.1, 'signed': True},
    'soc':   {'function': 3, 'address': 0x301A, 'count': 1, 'scale': 0.1, 'signed': False},
    'soh':   {'function': 3, 'address': 0x301B, 'count': 1, 'scale': 0.1, 'signed': False},
}


def _decode_float32_be(registers, scale=1.0):
    hi, lo = registers
    raw = struct.pack('>HH', hi, lo)
    value = struct.unpack('>f', raw)[0]
    if not math.isfinite(value):
        raise ValueError(f'float32 解碼結果非有限值: {value}')
    return value * scale


def _decode_int16(registers, scale=1.0, signed=True):
    val = registers[0]
    if signed and val >= 0x8000:
        val -= 0x10000
    return val * scale


def ensure_connected(client, name=''):
    """Modbus TCP 斷線重連。每次讀值前檢查 socket 狀態，斷線就重連一次。"""
    if client is None:
        return
    try:
        if not client.is_socket_open():
            print(f"  ⚠ {name} 連線已斷開，嘗試重新連線...")
            ok = client.connect()
            print(f"  {'✓ 重連成功' if ok else '✗ 重連失敗'} {name}")
    except Exception as e:
        print(f"  ⚠ {name} 檢查/重建連線時發生例外: {e}")


def read_meter_power_kw(client, cfg):
    """
    讀 Acuvim II 電表 Psum（FC03 0x4022, float32）。
    失敗直接丟例外，呼叫端負責標記狀態，不在這裡補值。
    回傳 (value_kw, registers)。
    """
    spec = METER_POWER_SPEC
    rr = client.read_holding_registers(address=spec['address'], count=spec['count'],
                                        device_id=cfg['unit_id'])
    if rr is None or rr.isError():
        raise RuntimeError(f"{cfg['name']}: {rr}")
    value = _decode_float32_be(rr.registers, spec['scale'])
    return value, list(rr.registers)


def read_ems_status(client, cfg, profile=EMS_KAC50DP_LEVEL1_PROFILE):
    """
    讀 EMS ess_power/soc/soh（FC03）。任一項失敗就整組視為失敗，
    不補值、不沿用上一筆。
    """
    out = {}
    for key in ['power', 'soc', 'soh']:
        spec = profile[key]
        rr = client.read_holding_registers(address=spec['address'], count=spec['count'],
                                            device_id=cfg['unit_id'])
        if rr is None or rr.isError():
            raise RuntimeError(f"{cfg['name']} {key} (0x{spec['address']:04X}): {rr}")
        out[key] = _decode_int16(rr.registers, spec['scale'], spec.get('signed', True))
    return {'ess_power': out['power'], 'soc': out['soc'], 'soh': out['soh']}


def connect_all_modbus_clients():
    """連線主表、儲能電表，以及（若啟用）EMS。回傳 (main_client, ess_client, ems_client)。"""
    main_client = ModbusTcpClient(MAIN_METER_CFG['ip'], port=MAIN_METER_CFG['port'], timeout=3)
    ess_client  = ModbusTcpClient(ESS_METER_CFG['ip'],  port=ESS_METER_CFG['port'], timeout=3)
    ems_client  = None

    for name, c in [('主表', main_client), ('儲能電表', ess_client)]:
        try:
            ok = c.connect()
            print(f"  {'✓' if ok else '✗'} 連線 {name} ({c})")
        except Exception as e:
            print(f"  ✗ 連線 {name} 發生例外: {type(e).__name__}: {e}")

    if EMS_CFG.get('enabled', False):
        ems_client = ModbusTcpClient(EMS_CFG['ip'], port=EMS_CFG['port'], timeout=3)
        try:
            ok = ems_client.connect()
            print(f"  {'✓' if ok else '✗'} 連線 EMS ({ems_client})")
        except Exception as e:
            print(f"  ✗ 連線 EMS 發生例外: {type(e).__name__}: {e}")
    else:
        print("  ⊘ EMS 已停用（EMS_CFG['enabled']=False），略過連線")

    return main_client, ess_client, ems_client


def close_all_modbus_clients(main_client, ess_client, ems_client):
    for c in [main_client, ess_client, ems_client]:
        if c is not None:
            try:
                c.close()
            except Exception:
                pass


# ============================================================
# 即時輪詢核心函式
# ============================================================

def _sleep_until_next_minute_mark(second_offset=0):
    """
    睡到下一個「整分鐘 + second_offset 秒」的時間點
    （例如 second_offset=10 → 對齊到 11:00:10、11:01:10 ...），
    而不是單純睡 N 秒（會累積時間漂移）。
    """
    now = datetime.now()
    candidate = now.replace(second=second_offset, microsecond=0)
    if candidate <= now:
        candidate += timedelta(minutes=1)
    wait = (candidate - now).total_seconds()
    time.sleep(max(wait, 0))


def poll_once(main_client, ess_client, ems_client, buffer: deque):
    """
    讀一輪現場即時值，組成一筆記錄，放進 rolling buffer。
    ★ 讀取失敗時明確標記狀態，不沿用上一筆、不補 0。
    """
    now = datetime.now()

    record = {
        'timestamp': now,
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M'),
        'before_load': None,
        'after_load': None,
        'ess_power': None,
        'soc': None,
        'soh': None,
        'contract_capacity': CONTRACT_CAPACITY_KW,
        'over_contract': None,
        # 現場目前無 cell 電壓差 / 自放電率的即時暫存器可讀，用健康基準值占位
        # （這兩個欄位只有在 use_synthetic_diagnostic_features=True 訓練出來的
        #   模型才會用到；v4 預設 False，正常不會吃到這兩個值）
        'cell_v_diff': 0.01,
        'self_discharge_rate': 0.002,
        'is_anomaly': 0,
        'anomaly_type': 'normal',
        'modbus_status': {},
    }

    ensure_connected(main_client, MAIN_METER_CFG['name'])
    try:
        before_load, _ = read_meter_power_kw(main_client, MAIN_METER_CFG)
        record['before_load'] = before_load
        record['modbus_status']['main_meter'] = 'GOOD'
    except Exception as e:
        record['modbus_status']['main_meter'] = f'BAD: {e}'
        print(f"  ⚠ 讀主表失敗: {e}")

    ensure_connected(ess_client, ESS_METER_CFG['name'])
    try:
        ess_power, _ = read_meter_power_kw(ess_client, ESS_METER_CFG)
        record['ess_power'] = ess_power  # 已驗證：儲能電表量到的就是 ESS 功率本身
        record['modbus_status']['ess_meter'] = 'GOOD'
    except Exception as e:
        record['modbus_status']['ess_meter'] = f'BAD: {e}'
        print(f"  ⚠ 讀儲能電表失敗: {e}")

    if EMS_CFG.get('enabled', False) and ems_client is not None:
        ensure_connected(ems_client, EMS_CFG['name'])
        try:
            ems_status = read_ems_status(ems_client, EMS_CFG)
            record['ess_power'] = ems_status['ess_power']  # EMS 若可用，優先信任 EMS 自己的值
            record['soc'] = ems_status['soc']
            record['soh'] = ems_status['soh']
            record['modbus_status']['ems'] = 'GOOD'
        except Exception as e:
            record['modbus_status']['ems'] = f'BAD: {e}'
            print(f"  ⚠ 讀 EMS 失敗: {e}")
    else:
        record['modbus_status']['ems'] = 'EMS_DISABLED'

    if record['before_load'] is not None and record['ess_power'] is not None:
        record['after_load'] = record['before_load'] - record['ess_power']
        record['over_contract'] = max(0.0, record['after_load'] - CONTRACT_CAPACITY_KW)

    buffer.append(record)
    return record


def resample_buffer_to_15min(buffer: deque, resample_to='15min'):
    """推論前把 raw buffer 依訓練時相同頻率重新聚合。"""
    df_buf = pd.DataFrame(list(buffer)).sort_values('timestamp')

    numeric_cols = ['before_load', 'after_load', 'contract_capacity', 'ess_power', 'soc', 'soh',
                     'cell_v_diff', 'self_discharge_rate', 'over_contract']
    numeric_cols = [c for c in numeric_cols if c in df_buf.columns]

    df_res = (df_buf.set_index('timestamp')[numeric_cols]
              .resample(resample_to)
              .mean()
              .ffill()
              .bfill()
              .dropna()
              .reset_index())

    df_res['date'] = df_res['timestamp'].dt.strftime('%Y-%m-%d')
    df_res['time'] = df_res['timestamp'].dt.strftime('%H:%M')
    return df_res


def infer_anomaly(model_bundle, buffer: deque):
    """把 buffer 重採樣 → 特徵工程 → 用訓練好的模型對最新一筆做推論。"""
    resample_to = model_bundle.get('resample_to', '15min')
    df_res = resample_buffer_to_15min(buffer, resample_to=resample_to)

    if len(df_res) < 2:
        return 0, 'normal', 0.0

    df_feat = engineer_features_advanced(df_res, window_sizes=model_bundle['window_sizes'])

    feature_cols = model_bundle['feature_cols']
    X_last = df_feat[feature_cols].values[-1:].astype(float)
    X_last = np.nan_to_num(X_last, nan=0.0, posinf=0.0, neginf=0.0)

    idx = model_bundle['selected_indices']
    X_sel = X_last[:, idx]

    if_score = model_bundle['if_model'].decision_function(X_sel)[0]
    sup_proba = model_bundle['supervised_model'].predict_proba(X_sel)[:, 1][0]

    if_min = model_bundle.get('if_score_min', -0.5)
    if_max = model_bundle.get('if_score_max', 0.5)
    if_score_norm = 1.0 - min(max((if_score - if_min) / (if_max - if_min + 1e-8), 0.0), 1.0)

    combined = model_bundle['hybrid_weight'] * if_score_norm + (1 - model_bundle['hybrid_weight']) * sup_proba
    is_anomaly = int(combined >= model_bundle['threshold'])

    anomaly_type = 'normal'
    if is_anomaly and model_bundle['multiclass_model'] is not None:
        mc_idx = model_bundle.get('multiclass_selected_indices')
        X_mc = X_last[:, mc_idx] if mc_idx is not None else X_last
        pred_cls = model_bundle['multiclass_model'].predict(X_mc)[0]
        inv_map = {v: k for k, v in model_bundle['anomaly_type_map'].items()}
        anomaly_type = inv_map.get(int(pred_cls), 'normal')

    return is_anomaly, anomaly_type, float(combined)


def run_realtime_loop(ref, model_bundle=None, poll_interval_sec=15, buffer_hours=None, max_iterations=None,
                       upload_to_firebase=True, align_to_clock=False, align_second_offset=0):
    """
    【階段二】主迴圈：每隔 poll_interval_sec 秒 → 讀 Modbus → （若 SOC 可用）跑模型推論
    → 視 upload_to_firebase 決定是否寫回 Firebase。

    ★ EMS 停用或讀取失敗時（record['soc'] is None），不會假造 SOC 去跑模型，
      而是只記錄功率相關資料，anomaly_type 標成 'EMS_DISABLED_NO_MODEL'。

    align_to_clock:
      False（預設）：每次讀完就單純 sleep(poll_interval_sec) 秒，時間點會隨執行
                    時的啟動時刻漂移（例如 10:42:37 起跳，之後就是 ...:43:37,
                    ...:44:37 ...）。
      True        ：對齊到整分鐘（或整分鐘 + align_second_offset 秒）才讀值，
                    這種模式下 poll_interval_sec 建議設 60（或 60 的因數），
                    每次都會等到下一個對齊點再讀，不會累積時間漂移。

    align_second_offset:
      對齊到每分鐘的第幾秒（只有 align_to_clock=True 時有作用）。
      例如 align_second_offset=10 → 對齊到 11:00:10、11:01:10 ...
      預設 0 → 對齊到整分鐘 11:00:00、11:01:00 ...
    """
    if model_bundle is None:
        model_bundle = joblib.load(MODEL_BUNDLE_PATH)
        print(f"✓ 已載入模型: {MODEL_BUNDLE_PATH}")

    max_window_minutes = (max(model_bundle['window_sizes']) + 2) * 15
    if buffer_hours is None:
        buffer_hours = max_window_minutes / 60.0

    required_raw_points = int(np.ceil((buffer_hours * 3600) / poll_interval_sec)) + 10
    buffer_maxlen = max(required_raw_points, 30)

    resample_to = model_bundle.get('resample_to', '15min')
    resample_minutes = pd.Timedelta(resample_to).total_seconds() / 60.0
    min_raw_needed = max(5, int((resample_minutes * 60 * 2) / poll_interval_sec))

    print("\n" + "=" * 70)
    print("   【階段二】即時 Modbus TCP 監測啟動")
    print(f"   Buffer 涵蓋時間跨度: ~{buffer_hours:.1f} 小時（最多保留 {buffer_maxlen} 筆 raw 輪詢）")
    print(f"   推論前至少累積 raw 筆數: {min_raw_needed}")
    print(f"   EMS 狀態: {'啟用' if EMS_CFG.get('enabled', False) else '停用（SOC/SOH 本輪一律略過模型推論）'}")
    print("=" * 70)

    main_client, ess_client, ems_client = connect_all_modbus_clients()
    buffer = deque(maxlen=buffer_maxlen)

    if align_to_clock:
        now = datetime.now()
        candidate = now.replace(second=align_second_offset, microsecond=0)
        if candidate <= now:
            candidate += timedelta(minutes=1)
        print(f"   對齊時鐘模式：等待到 {candidate.strftime('%H:%M:%S')} 才開始第一次讀值...")
        _sleep_until_next_minute_mark(align_second_offset)

    iteration = 0
    try:
        while True:
            record = poll_once(main_client, ess_client, ems_client, buffer)

            has_soc = record['soc'] is not None
            if has_soc and len(buffer) >= min_raw_needed:
                is_anomaly, anomaly_type, score = infer_anomaly(model_bundle, buffer)
                record['is_anomaly'] = is_anomaly
                record['anomaly_type'] = anomaly_type
                bl = f"{record['before_load']:.1f}" if record['before_load'] is not None else 'N/A'
                al = f"{record['after_load']:.1f}" if record['after_load'] is not None else 'N/A'
                ep = f"{record['ess_power']:.2f}" if record['ess_power'] is not None else 'N/A'
                print(f"[{record['time']}] before={bl}kW after={al}kW ESS={ep}kW "
                      f"SOC={record['soc']:.1f}% → "
                      f"{'⚠ 異常:' + anomaly_type if is_anomaly else '正常'} (score={score:.3f})")
            else:
                record['is_anomaly'] = 0
                record['anomaly_type'] = 'EMS_DISABLED_NO_MODEL' if not has_soc else 'ACCUMULATING'
                bl = f"{record['before_load']:.1f}" if record['before_load'] is not None else 'N/A'
                al = f"{record['after_load']:.1f}" if record['after_load'] is not None else 'N/A'
                ep = f"{record['ess_power']:.2f}" if record['ess_power'] is not None else 'N/A'
                reason = '無 SOC（EMS 停用/失敗），僅記錄功率' if not has_soc else f'累積資料中 ({len(buffer)}/{min_raw_needed})'
                print(f"[{record['time']}] before={bl}kW after={al}kW ESS={ep}kW → {reason}")

            if upload_to_firebase and ref is not None:
                upload_single_record_to_firebase(ref, record)

            iteration += 1
            if max_iterations and iteration >= max_iterations:
                break
            if align_to_clock:
                _sleep_until_next_minute_mark(align_second_offset)
            else:
                time.sleep(poll_interval_sec)

    except KeyboardInterrupt:
        print("\n手動中止即時監測")
    finally:
        close_all_modbus_clients(main_client, ess_client, ems_client)
