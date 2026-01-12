"""Frontend-specific CSV ingestion used for runtime detection uploads."""

"""NOTE: Processing the CSV's from the user"""

"""
**frontend_ingest.py** keeps CSV sanitization, canonicalization, and feature derivation 
(Hour, IP octet, etc.) isolated from the UI. That means Streamlit stays thin while the 
ingest logic can be unit-tested, reused by APIs, or swapped out later (e.g., batch scoring).

"""

import io
from pathlib import Path
from typing import IO, Any

import pandas as pd

CORE_RUNTIME_COLS: tuple[str, ...] = (
    'Timestamp',
    'Source_IP',
    'Destination_IP',
    'Payload_Type',
    'Payload_Description',
    'Severity',
)

DEFAULTS: dict[str, object] = {
    'Source_IP': '0.0.0.0',
    'Destination_IP': '0.0.0.0',
    'Payload_Type': 'Unknown',
    'Payload_Description': '',
    'Severity': 'Unknown',
}

SEVERITY_CANONICAL: dict[str, str] = {
    'critical': 'Critical',
    'high': 'High',
    'medium': 'Medium',
    'moderate': 'Medium',
    'low': 'Low',
    'info': 'Informational',
    'informational': 'Informational',
    'unknown': 'Unknown',
    'na': 'Unknown',
}

PAYLOAD_TYPE_CANONICAL: dict[str, str] = {
    'c2': 'Command-And-Control',
    'command-and-control': 'Command-And-Control',
    'cmd': 'Command-And-Control',
    'exploit': 'Exploit',
    'phishing': 'Phishing',
    'malware': 'Malware',
    'dns': 'DNS',
    'download': 'Download',
    'benign': 'Benign',
    'normal': 'Benign',
    'unknown': 'Unknown',
}


def load_frontend_payload(
    input_data: str | Path | IO[Any] | bytes | bytearray | pd.DataFrame,
    *,
    clean: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """Load user-uploaded CSVs or DataFrames for runtime detection."""

    df = _read_input(input_data, copy=copy)
    if not clean:
        return df

    df = _standardize_columns(df)
    df = _ensure_columns(df)
    df = _fill_defaults(df)
    df = _coerce_timestamp(df)
    df = _normalize_categories(df)
    df = _derive_runtime_features(df)
    return df.reset_index(drop=True)


def _read_input(input_data: str | Path | IO[Any] | bytes | bytearray | pd.DataFrame, *, copy: bool) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy(deep=copy)

    if isinstance(input_data, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(input_data), low_memory=False)

    if hasattr(input_data, 'read'):
        # Streamlit uploads expose UploadedFile with read/seek
        if hasattr(input_data, 'seek'):
            input_data.seek(0)
        return pd.read_csv(input_data, low_memory=False)

    path = Path(input_data)
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")

    return pd.read_csv(path, low_memory=False)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CORE_RUNTIME_COLS:
        if col not in out.columns:
            out[col] = pd.NA if col == 'Timestamp' else DEFAULTS.get(col, '')
    return out


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in DEFAULTS.items():
        if col in out.columns:
            out[col] = out[col].fillna(default)
    return out


def _coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if 'Timestamp' not in df.columns:
        return df

    out = df.copy()
    out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce', utc=True)
    out = out[out['Timestamp'].notna()].copy()
    out['Hour'] = out['Timestamp'].dt.hour.astype('Int64')
    return out


def _normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'Severity' in out.columns:
        normalized = (
            out['Severity']
            .astype('string')
            .str.strip()
            .str.lower()
        )
        out['Severity'] = normalized.map(SEVERITY_CANONICAL).fillna(out['Severity'].fillna('Unknown'))

    if 'Payload_Type' in out.columns:
        normalized = (
            out['Payload_Type']
            .astype('string')
            .str.strip()
            .str.lower()
        )
        out['Payload_Type'] = normalized.map(PAYLOAD_TYPE_CANONICAL).fillna(out['Payload_Type'].fillna('Unknown'))

    return out


def _derive_runtime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if 'Payload_Description' in out.columns:
        out['Payload_Description'] = out['Payload_Description'].astype('string').fillna('')

    # TODO: Attach preliminary MITRE ATT&CK tactic/technique tags (Tip: start with a lookup dict that keys off Payload_Type keywords and high-signal tokens in Payload_Description).

    for col in ('Source_IP', 'Destination_IP'):
        if col in out.columns:
            out[col] = out[col].astype('string').fillna(DEFAULTS.get(col, '')).str.strip()

    if 'Source_IP' in out.columns:
        out['Src_IP_LastOctet'] = out['Source_IP'].map(_ip_last_octet).astype('Int64')

    return out


def _ip_last_octet(value: str) -> int:
    try:
        return int(str(value).strip().split('.')[-1])
    except Exception:
        return -1


__all__ = ['load_frontend_payload']
