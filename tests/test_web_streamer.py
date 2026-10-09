"""
Unit tests for WebLiveStreamer (camping_guard/tools/web_streamer.py)
"""

import time
import urllib.request
import json
import numpy as np
import pytest
from camping_guard.tools.web_streamer import WebLiveStreamer, get_local_ips


def test_get_local_ips():
    ips = get_local_ips()
    assert isinstance(ips, list)
    assert len(ips) >= 1
    assert "127.0.0.1" in ips


def test_web_streamer_lifecycle_and_api():
    port = 8999
    streamer = WebLiveStreamer(host="127.0.0.1", port=port, auto_start=True)
    try:
        time.sleep(0.1)

        # 1. Test index page
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=2.0) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "Camping Guard" in html
            assert "Jetson Orin" in html

        # 2. Test status API
        streamer.update_meta(threat_level="L2 警示", state="S2 警戒", fps=24.5, e2e_ms=18.2)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["threat_level"] == "L2 警示"
            assert data["state"] == "S2 警戒"
            assert data["fps"] == 24.5
            assert data["e2e_ms"] == 18.2

        # 3. Test push frame
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        streamer.push_frame(dummy_frame)
        assert streamer.current_jpeg_bytes is not None

        # 4. Test action API
        req_url = f"http://127.0.0.1:{port}/api/action?cmd=pause"
        with urllib.request.urlopen(req_url, timeout=2.0) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["status"] == "ok"

        action = streamer.pop_action()
        assert action == "pause"
        assert streamer.pop_action() is None

    finally:
        streamer.stop()
