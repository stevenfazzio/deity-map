"""Shared paths and helpers for the deity map pipeline."""

import os
import random
import stat
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CANDIDATES_PARQUET = DATA / "candidates.parquet"
ARTICLES_PARQUET = DATA / "articles.parquet"

# Wikimedia rate-limits generic User-Agents; it asks for a project name and contact.
USER_AGENT = "deity-map/0.1 (https://github.com/stevenfazzio/deity-map; fazzios@gmail.com)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WDQS = "https://query.wikidata.org/sparql"

RETRY_STATUS = {429, 500, 502, 503, 504}


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def get_with_retry(
    session: requests.Session, url: str, params=None, timeout: int = 60, max_retries: int = 6
) -> requests.Response:
    """GET, retrying throttling and server faults; any other 4xx raises at once."""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code in RETRY_STATUS and attempt < max_retries - 1:
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else min(2**attempt * 5, 120)
                wait += random.uniform(0, wait * 0.1)
                print(f"  {resp.status_code} from {url}; retrying in {wait:.0f}s", flush=True)
            else:
                resp.raise_for_status()
                return resp
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 5, 120)
            print(f"  attempt {attempt + 1}: {e!r}; retrying in {wait:.0f}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"exhausted retries for {url}")


def wiki_query(session: requests.Session, params: dict, max_lag_retries: int = 10) -> dict:
    """One action-API query with maxlag set; waits out replication lag instead of failing."""
    params = {"action": "query", "format": "json", "formatversion": "2", "maxlag": "5", **params}
    for _ in range(max_lag_retries):
        resp = get_with_retry(session, WIKI_API, params=params)
        data = resp.json()
        # maxlag comes back as HTTP 200 with an error body and a Retry-After header.
        if data.get("error", {}).get("code") == "maxlag":
            time.sleep(float(resp.headers.get("Retry-After", "5")))
            continue
        if "error" in data:
            raise RuntimeError(f"API error: {data['error']}")
        return data
    raise RuntimeError("replication lag did not clear")


def _match_or_default_mode(tmp_path: Path, output_path: Path) -> None:
    # mkstemp creates 0600 and os.replace preserves the source mode.
    if output_path.exists():
        os.chmod(tmp_path, stat.S_IMODE(output_path.stat().st_mode))
    else:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)


def write_bytes_atomic(data: bytes, output_path: Path) -> None:
    """Write so output_path only ever holds a complete file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        _match_or_default_mode(Path(tmp), output_path)
        os.replace(tmp, output_path)
    except BaseException:
        os.unlink(tmp)
        raise


def write_npz_atomic(output_path: Path, check=None, **arrays) -> None:
    """np.savez to a temp file, optionally verify it with check(loaded), then swap in."""
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The suffix must already be .npz, or np.savez appends one and writes elsewhere.
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        if check is not None:
            check(np.load(tmp))
        _match_or_default_mode(Path(tmp), output_path)
        os.replace(tmp, output_path)
    except BaseException:
        os.unlink(tmp)
        raise


def write_parquet_safely(df, output_path: Path) -> None:
    """Write df to a temp file, verify rows and schema from the footer, then swap in."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        _match_or_default_mode(Path(tmp), output_path)
        meta = pq.read_metadata(tmp)
        assert meta.num_rows == len(df), f"row count {meta.num_rows} != {len(df)}"
        # to_arrow_schema(): the Parquet schema's .names are leaf names, so list
        # columns would show up as "element" rather than their top-level name.
        missing = set(df.columns) - set(meta.schema.to_arrow_schema().names)
        assert not missing, f"columns missing on disk: {missing}"
        os.replace(tmp, output_path)
    except BaseException:
        os.unlink(tmp)
        raise
