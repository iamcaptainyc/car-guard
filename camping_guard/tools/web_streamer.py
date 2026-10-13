# -*- coding: utf-8 -*-
"""
=============================================================================
无头环境高性能 Web 实时看板与控制台 (camping_guard/tools/web_streamer.py)
专为无显示器的 Linux / Jetson Orin / 远程车载服务器打造。
无需安装 Flask / FastAPI / Nginx，纯 Python 标准库 (ThreadingHTTPServer) 驱动。

核心特性：
1. 0% CPU 绘图开销：视频推流 100% 直通 1080P 原始画面，所有遥测与指标完全交由网页 DOM 渲染
2. 动态多维看板：实时展示活动目标列表、滞留时间(驻留秒数)、车身 ROI 扰动、端侧 VLM 战报与控制指令
3. 多线程解耦 (ThreadingHTTPServer)：流推送、遥测 API 轮询与控制指令互不阻塞
4. 零积压防卡顿机制 (Latest-Frame Skip)：仅推送最新帧，慢网络自动跳帧，杜绝缓冲延迟
5. 全 UTF-8 严格编码：彻底根治跨平台字符集问题
=============================================================================
"""

import os
import sys
import time
import json
import socket
import threading
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List
import cv2
import numpy as np


if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def get_local_ips() -> List[str]:
    """探测本机所有局域网可访问的 IPv4 地址"""
    ips = ["127.0.0.1"]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        default_ip = s.getsockname()[0]
        s.close()
        if default_ip not in ips:
            ips.insert(0, default_ip)
    except Exception:
        pass
    return ips


WEB_UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camping Guard - Jetson Orin 车载露营守卫实时中控</title>
    <style>
        :root {
            --bg-dark: #0a0e17;
            --panel-bg: #131926;
            --card-bg: #1a2233;
            --card-hover: #222c42;
            --border-color: #26344d;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-orange: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif; }
        body { background-color: var(--bg-dark); color: var(--text-main); display: flex; flex-direction: column; min-height: 100vh; overflow-x: hidden; }
        
        /* 顶部导航与全局遥测条 */
        header { background: var(--panel-bg); border-bottom: 1px solid var(--border-color); padding: 8px 16px; height: 50px; display: flex; justify-content: space-between; align-items: center; flex-wrap: nowrap; overflow-x: auto; gap: 10px; flex-shrink: 0; }
        .logo { font-size: 17px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
        .logo-tag { font-size: 12px; background: #047857; color: #fff; padding: 2px 8px; border-radius: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; }
        .pulse-dot { width: 8px; height: 8px; background-color: #34d399; border-radius: 50%; display: inline-block; animation: pulse-live 1.5s infinite; }
        @keyframes pulse-live { 0% { transform: scale(0.9); opacity: 0.7; } 50% { transform: scale(1.3); opacity: 1; } 100% { transform: scale(0.9); opacity: 0.7; } }
        
        .telemetry-bar { display: flex; gap: 8px; font-size: 12px; flex-wrap: nowrap; align-items: center; flex-shrink: 0; }
        .badge { background: var(--card-bg); border: 1px solid var(--border-color); padding: 5px 10px; border-radius: 6px; display: flex; align-items: center; gap: 6px; white-space: nowrap; }
        .badge b { color: #fff; }
        
        /* 主体布局 */
        main { display: flex; flex: 1; padding: 12px; gap: 12px; width: 100%; height: calc(100vh - 50px); overflow: hidden; }
        .video-box { flex: 7; background: #000; border-radius: 8px; border: 1px solid var(--border-color); overflow: hidden; display: flex; justify-content: center; align-items: center; position: relative; }
        .video-box img { width: 100%; height: 100%; object-fit: contain; display: block; image-rendering: -webkit-optimize-contrast; }
        
        .sidebar { flex: 5; max-width: 520px; min-width: 380px; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; padding-right: 4px; }
        .sidebar::-webkit-scrollbar { width: 6px; }
        .sidebar::-webkit-scrollbar-thumb { background: #26344d; border-radius: 3px; }
        
        .card { background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; }
        .card h3 { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border-color); padding-bottom: 4px; display: flex; justify-content: space-between; align-items: center; }
        .card-count { font-size: 11px; background: #26344d; color: #94a3b8; padding: 1px 6px; border-radius: 10px; }
        
        /* 交互控制卡片：置顶吸顶锁定，高度固定，任何后台数据轮询与增删均零抖动、零跳动 */
        .control-card { position: sticky; top: 0; z-index: 20; background: var(--panel-bg); box-shadow: 0 4px 12px rgba(0,0,0,0.45); flex-shrink: 0; }
        .btn-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
        button { background: var(--card-bg); border: 1px solid var(--border-color); color: var(--text-main); padding: 8px 4px; border-radius: 6px; cursor: pointer; font-size: 11px; font-weight: 500; transition: background 0.15s, border-color 0.15s; text-align: center; user-select: none; -webkit-user-select: none; }
        button:hover { background: #222d42; border-color: #3b82f6; color: #fff; }
        button:active { transform: translateY(1px); }
        button.warn { background: #5a2e04; border-color: #f59e0b; color: #fff; }
        button.warn:hover { background: #783d05; }
        button.danger { background: #5f1515; border-color: #ef4444; color: #fff; }
        button.danger:hover { background: #7f1d1d; }

        /* 数据表格：固定高度容器，目标出入时不会撑大或收缩页面高度 */
        .table-wrap { width: 100%; overflow-x: auto; height: 145px; max-height: 145px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }
        th { background: #1a2233; color: var(--text-muted); padding: 6px 8px; border-bottom: 1px solid var(--border-color); font-weight: 600; white-space: nowrap; position: sticky; top: 0; z-index: 2; }
        td { padding: 6px 8px; border-bottom: 1px solid rgba(38, 52, 77, 0.4); white-space: nowrap; }
        tr:hover td { background-color: var(--card-hover); }
        .empty-row { text-align: center; color: var(--text-muted); padding: 45px 0; }
        
        /* 滞留时间与状态标签 */
        .dwell-tag { font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
        .dwell-normal { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
        .dwell-warn { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
        .dwell-danger { background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.6); animation: pulse-warn 1s infinite alternate; }
        @keyframes pulse-warn { from { opacity: 0.8; } to { opacity: 1; } }
        
        /* ROI 传感器网格 */
        .roi-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(105px, 1fr)); gap: 6px; }
        .roi-pill { background: var(--card-bg); border: 1px solid var(--border-color); padding: 6px 8px; border-radius: 6px; font-size: 11px; display: flex; flex-direction: column; gap: 2px; }
        .roi-pill .name { color: var(--text-muted); font-size: 10px; }
        .roi-pill .status { font-weight: 600; }
        .roi-pill.active { border-color: #ef4444; background: rgba(239, 68, 68, 0.15); }
        .roi-pill.active .status { color: #f87171; }
        .roi-pill.normal .status { color: #34d399; }
        
        /* 最新威胁事件卡片：固定最小高度以防止出现事件时下方元素跳位 */
        #latest-event-container { min-height: 115px; max-height: 165px; overflow-y: auto; }
        .event-card { background: #182236; border-left: 4px solid #ef4444; padding: 10px; border-radius: 4px; font-size: 12px; line-height: 1.5; }
        .event-card .evt-header { display: flex; justify-content: space-between; font-weight: 700; margin-bottom: 4px; }
        .event-card .evt-meta { color: var(--text-muted); font-size: 11px; margin-bottom: 6px; }
        .event-card .evt-vlm { background: #0f172a; padding: 6px 8px; border-radius: 4px; border: 1px solid #1e293b; color: #38bdf8; font-size: 12px; margin-bottom: 6px; }
        .event-card .evt-actions { display: flex; gap: 4px; flex-wrap: wrap; }
        .evt-act-tag { background: #1e3a8a; color: #bfdbfe; font-size: 10px; padding: 2px 6px; border-radius: 4px; }
        
        /* 推送 JSON 抽屉 */
        .json-box { background: #0b0f19; border: 1px solid #1e293b; border-radius: 4px; padding: 8px; font-family: monospace; font-size: 11px; color: #a5f3fc; max-height: 110px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
        
        /* 状态微章 */
        .pill { padding: 2px 8px; border-radius: 12px; font-weight: 600; font-size: 11px; }
        .pill-safe { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.5); }
        .pill-notice { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.5); }
        .pill-warn { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.5); }
        .pill-critical { background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.6); animation: pulse-warn 1s infinite alternate; }
        
        #toast { position: fixed; bottom: 20px; right: 20px; background: #2563eb; color: #fff; padding: 8px 16px; border-radius: 6px; font-size: 12px; opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 999; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
        @media (max-width: 960px) { main { flex-direction: column; height: auto; } .sidebar { max-width: 100%; min-width: 100%; } }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            Camping Guard <span class="logo-tag"><span class="pulse-dot"></span> Jetson Orin 极速实时流</span>
        </div>
        <div class="telemetry-bar">
            <div class="badge">威胁等级: <b id="threat-pill" class="pill pill-safe">L0 正常安全</b></div>
            <div class="badge">守卫状态: <b id="state-text">S0 待机</b></div>
            <div class="badge">运行模式: <b id="mode-text">车内模式</b></div>
            <div class="badge">流速: <b id="fps-text" style="color:#60a5fa;">-- FPS</b></div>
            <div class="badge">AI耗时: <b id="e2e-text" style="color:#34d399;">-- ms</b></div>
            <div class="badge">UWB钥匙: <b id="uwb-text" style="color:#94a3b8;">远离</b></div>
        </div>
    </header>
    <main>
        <div class="video-box">
            <img id="stream-img" src="/stream.mjpg" alt="实时视频流" />
        </div>
        <div class="sidebar">
            <!-- 1. 远程控制与实车信号注入 (置顶吸顶固定，杜绝任何数据刷新抖动跳位) -->
            <div class="card control-card">
                <h3>
                    <span>控制与实车物理信号注入</span>
                    <span class="card-count" style="color:#60a5fa;">即时响应</span>
                </h3>
                <div class="btn-grid">
                    <button onclick="sendAction('pause')">⏯ 暂停/继续</button>
                    <button onclick="sendAction('step')">⏭ 单帧步进</button>
                    <button onclick="sendAction('reset')">🔄 重播起点</button>
                    <button onclick="sendAction('toggle_key')">🔑 切换钥匙</button>
                    <button class="warn" onclick="sendAction('inject_handle')">🚪 CAN拉门</button>
                    <button class="danger" onclick="sendAction('inject_vibe')">💥 IMU震动</button>
                    <button onclick="sendAction('test_relay')">🛡 中继防盗</button>
                    <button class="danger" onclick="if(confirm('确定要停止守卫并保存视频退出吗？')) sendAction('quit')">⏹ 安全退出</button>
                </div>
            </div>

            <!-- 2. 实时目标感知表 (直接展示滞留时间) -->
            <div class="card">
                <h3>
                    <span>动态目标追踪与滞留研判</span>
                    <span id="target-count" class="card-count">0 目标</span>
                </h3>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>类别</th>
                                <th>测距 / 防区</th>
                                <th>驻留时间</th>
                                <th>径向速度</th>
                                <th>身份状态</th>
                            </tr>
                        </thead>
                        <tbody id="targets-tbody">
                            <tr><td colspan="6" class="empty-row">当前防区内无活动目标，车身环境安全</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- 3. 车身 ROI 微动监测 -->
            <div class="card">
                <h3>
                    <span>车身固定部件微动守护 (ROI)</span>
                    <span id="roi-status-summary" class="card-count">全域正常</span>
                </h3>
                <div class="roi-grid" id="rois-container">
                    <div class="roi-pill normal"><span class="name">左前门把手</span><span class="status">正常</span></div>
                    <div class="roi-pill normal"><span class="name">左前车门</span><span class="status">正常</span></div>
                    <div class="roi-pill normal"><span class="name">左侧车窗</span><span class="status">正常</span></div>
                    <div class="roi-pill normal"><span class="name">后备箱尾门</span><span class="status">正常</span></div>
                </div>
            </div>

            <!-- 4. 最新威胁事件与端侧 VLM 战报 -->
            <div class="card">
                <h3>最新威胁事件与 VLM 复核战报</h3>
                <div id="latest-event-container">
                    <div style="color:var(--text-muted); font-size:12px; padding:35px 10px; text-align:center;">暂无防御威胁事件触发</div>
                </div>
            </div>

            <!-- 5. 远端 App 推送 JSON 预览 -->
            <div class="card">
                <h3>📲 移动端/云端推送格式示例 (JSON)</h3>
                <pre id="push-json" class="json-box">等待事件触发生成推送通知载荷...</pre>
            </div>
        </div>
    </main>
    <div id="toast"></div>

    <script>
        function showToast(msg) {
            const t = document.getElementById('toast');
            t.innerText = msg;
            t.style.opacity = '1';
            setTimeout(() => { t.style.opacity = '0'; }, 2000);
        }
        function sendAction(action) {
            fetch('/api/action?cmd=' + action)
                .then(r => r.json())
                .then(data => {
                    if (data.msg) showToast(data.msg);
                })
                .catch(e => showToast('指令发送失败: ' + e));
        }

        function renderTargets(targets) {
            const tbody = document.getElementById('targets-tbody');
            const countEl = document.getElementById('target-count');
            if (!targets || targets.length === 0) {
                countEl.innerText = '0 目标';
                tbody.innerHTML = '<tr><td colspan="6" class="empty-row">当前防区内无活动目标，车身环境安全</td></tr>';
                return;
            }
            countEl.innerText = targets.length + ' 目标';
            let html = '';
            targets.forEach(t => {
                const dwell = Number(t.dwell_time || 0);
                let dwellClass = 'dwell-normal';
                if (dwell >= 15.0) {
                    dwellClass = 'dwell-danger';
                } else if (dwell >= 5.0) {
                    dwellClass = 'dwell-warn';
                }
                const dist = Number(t.distance || 0).toFixed(1);
                const vr = Number(t.radial_speed || 0);
                const vrStr = (vr > 0 ? '+' : '') + vr.toFixed(1) + ' m/s';
                const statusStr = t.is_whitelisted ? '<span style="color:#34d399;">白名单免扰</span>' : (dwell >= 15.0 ? '<span style="color:#f87171;">长时徘徊</span>' : '实时跟踪');
                const catStr = t.category || '人';
                const idStr = t.session_id ? (t.track_id + ' (' + t.session_id + ')') : t.track_id;
                
                html += `<tr>
                    <td><b>${idStr}</b></td>
                    <td>${catStr}</td>
                    <td>${dist}m <span style="color:#94a3b8; font-size:11px;">(${t.zone || '防区'})</span></td>
                    <td><span class="dwell-tag ${dwellClass}">驻留 ${dwell.toFixed(1)}s</span></td>
                    <td>${vrStr}</td>
                    <td>${statusStr}</td>
                </tr>`;
            });
            tbody.innerHTML = html;
        }

        function renderRois(rois) {
            const container = document.getElementById('rois-container');
            const sumEl = document.getElementById('roi-status-summary');
            if (!rois || rois.length === 0) return;
            let hasActive = false;
            let html = '';
            rois.forEach(r => {
                const isAct = r.disturbed;
                if (isAct) hasActive = true;
                const cls = isAct ? 'roi-pill active' : 'roi-pill normal';
                const statText = isAct ? '触碰微动!' : '正常';
                html += `<div class="${cls}"><span class="name">${r.name}</span><span class="status">${statText}</span></div>`;
            });
            container.innerHTML = html;
            sumEl.innerText = hasActive ? '⚠️ 异常扰动' : '全域正常';
            sumEl.style.background = hasActive ? '#7f1d1d' : '#26344d';
            sumEl.style.color = hasActive ? '#fca5a5' : '#94a3b8';
        }

        function renderLatestEvent(evt) {
            const container = document.getElementById('latest-event-container');
            const jsonBox = document.getElementById('push-json');
            if (!evt) return;

            const threat = evt.threat_level || 'L0';
            const borderCol = threat.includes('L3') ? '#ef4444' : (threat.includes('L2') ? '#f59e0b' : '#3b82f6');
            const dwell = Number(evt.target_dwell_time || 0).toFixed(1);
            const dist = Number(evt.target_distance || 0).toFixed(1);
            const vlmText = evt.vlm_summary || '端侧正在执行语义复核研判...';
            const actions = (evt.executed_actions || []).map(a => `<span class="evt-act-tag">${a}</span>`).join(' ');

            container.innerHTML = `
                <div class="event-card" style="border-left-color: ${borderCol};">
                    <div class="evt-header">
                        <span>[${threat}] ${evt.event_id || ''}</span>
                        <span style="color:#94a3b8; font-weight:normal;">${evt.timestamp ? new Date(evt.timestamp * 1000).toLocaleTimeString() : ''}</span>
                    </div>
                    <div class="evt-meta">
                        目标: <b>${evt.target_category || '人'}</b> | 测距: <b>${dist}m</b> | <b>驻留徘徊: ${dwell}s</b>
                        ${evt.disturbed_rois && evt.disturbed_rois.length ? (' | 扰动部位: ' + evt.disturbed_rois.join(',')) : ''}
                    </div>
                    <div class="evt-vlm">🤖 <b>端侧 Qwen-VLM 战报:</b> "${vlmText}"</div>
                    <div class="evt-actions">整车联动: ${actions || '<span style="color:#94a3b8;">监视存证</span>'}</div>
                </div>
            `;

            if (evt.push_payload) {
                jsonBox.innerText = JSON.stringify(evt.push_payload, null, 2);
            }
        }

        function pollStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(d => {
                    if (d.threat_level) {
                        const tp = document.getElementById('threat-pill');
                        tp.innerText = d.threat_level;
                        if (d.threat_level.includes('L3') || d.threat_level.includes('紧急')) {
                            tp.className = 'pill pill-critical';
                        } else if (d.threat_level.includes('L2') || d.threat_level.includes('预警')) {
                            tp.className = 'pill pill-warn';
                        } else if (d.threat_level.includes('L1') || d.threat_level.includes('提示')) {
                            tp.className = 'pill pill-notice';
                        } else {
                            tp.className = 'pill pill-safe';
                        }
                    }
                    if (d.state) document.getElementById('state-text').innerText = d.state;
                    if (d.mode) document.getElementById('mode-text').innerText = d.mode;
                    if (d.fps !== undefined) {
                        const aiTag = d.ai_fps ? ` (AI: ${d.ai_fps} FPS)` : '';
                        document.getElementById('fps-text').innerText = d.fps + ' FPS' + aiTag;
                    }
                    if (d.e2e_ms !== undefined) document.getElementById('e2e-text').innerText = d.e2e_ms + ' ms';
                    
                    const uwbEl = document.getElementById('uwb-text');
                    if (d.uwb_online) {
                        uwbEl.innerText = '车主在场(免扰)';
                        uwbEl.style.color = '#34d399';
                    } else {
                        uwbEl.innerText = '钥匙远离(设防)';
                        uwbEl.style.color = '#94a3b8';
                    }

                    renderTargets(d.targets);
                    if (d.rois) renderRois(d.rois);
                    if (d.latest_event) renderLatestEvent(d.latest_event);
                })
                .catch(e => {});
        }
        setInterval(pollStatus, 400);
        pollStatus();
    </script>
</body>
</html>
"""


class WebLiveStreamer:
    """
    轻量、多线程并发、零积压 MJPEG 网页流推送与指令接收服务端
    完全消灭 OpenCV 画布 CPU 负担，指标全部结构化推流至网页 DOM。
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, auto_start: bool = True):
        self.host = host
        self.port = port
        self.current_jpeg_bytes = None
        self.frame_seq = 0
        self.condition = threading.Condition()
        self.server = None
        self.thread = None
        self.pending_actions: List[str] = []
        self._action_lock = threading.Lock()

        # 结构化遥测指标存储
        self.latest_telemetry: Dict[str, Any] = {
            "threat_level": "L0 正常安全",
            "state": "S0 待机",
            "mode": "车内模式",
            "fps": 0.0,
            "ai_fps": 0.0,
            "e2e_ms": 0.0,
            "frame_idx": 0,
            "total_frames": 0,
            "uwb_online": False,
            "targets": [],
            "rois": [],
            "latest_event": None
        }

        # 兼容旧版字段
        self.latest_meta: Dict[str, Any] = {
            "threat_level": "L0 正常安全",
            "state": "S0 待机",
            "fps": 0.0,
            "e2e_ms": 0.0
        }

        if auto_start:
            self.start()

    def update_meta(self, threat_level: str = "", state: str = "", fps: float = 0.0, e2e_ms: float = 0.0):
        """向后兼容的轻量指标更新接口"""
        with self._action_lock:
            if threat_level:
                self.latest_meta["threat_level"] = str(threat_level)
                self.latest_telemetry["threat_level"] = str(threat_level)
            if state:
                self.latest_meta["state"] = str(state)
                self.latest_telemetry["state"] = str(state)
            if fps > 0:
                self.latest_meta["fps"] = round(float(fps), 1)
                self.latest_telemetry["fps"] = round(float(fps), 1)
            if e2e_ms > 0:
                self.latest_meta["e2e_ms"] = round(float(e2e_ms), 1)
                self.latest_telemetry["e2e_ms"] = round(float(e2e_ms), 1)

    def update_telemetry(
        self,
        active_targets: Optional[List[Any]] = None,
        active_rois: Optional[List[Any]] = None,
        latest_event: Optional[Any] = None,
        threat_level: str = "",
        state: str = "",
        mode: str = "",
        fps: float = 0.0,
        ai_fps: Optional[float] = None,
        e2e_ms: float = 0.0,
        frame_idx: int = 0,
        total_frames: int = 0,
        uwb_online: bool = False
    ):
        """
        更新全维度结构化遥测指标 (交由前端 DOM 渲染，0% OpenCV CPU 负担)：
        涵盖活动目标、距离、防区、滞留时间(驻留秒数)、车身部件微动、VLM战报与整车动作
        """
        targets_data = []
        for t in (active_targets or []):
            z_val = t.zone.value if hasattr(t.zone, 'value') else str(t.zone)
            z_short = "Zone 2" if "2" in z_val else ("Zone 1" if "1" in z_val else "Zone 0")
            cat_str = "人" if ("person" in str(t.category).lower() or "人" in str(t.category)) else ("车" if ("car" in str(t.category).lower() or "veh" in str(t.category).lower() or "车" in str(t.category)) else str(t.category))
            sess_id = getattr(t, "person_session_id", None)
            targets_data.append({
                "track_id": t.track_id,
                "session_id": sess_id or f"T{t.track_id}",
                "category": cat_str,
                "distance": round(float(getattr(t, "ground_dist", 0.0)), 2),
                "zone": z_short,
                "dwell_time": round(float(getattr(t, "cumulative_dwell_time", 0.0)), 1),
                "radial_speed": round(float(getattr(t, "radial_speed", 0.0)), 2),
                "is_whitelisted": bool(getattr(t, "is_whitelisted", False)),
                "has_skeleton": bool(getattr(t, "keypoints", None))
            })

        rois_data = []
        for r in (active_rois or []):
            roi_name = getattr(r, "roi_name", str(r))
            rois_data.append({
                "name": roi_name,
                "motion_ratio": round(float(getattr(r, "pixel_motion_ratio", 0.0)), 3),
                "disturbed": bool(getattr(r, "is_disturbed", True))
            })

        evt_data = None
        if latest_event is not None:
            if hasattr(latest_event, "to_push_payload"):
                push_pl = latest_event.to_push_payload()
            else:
                push_pl = {}

            if hasattr(latest_event, "event_id"):
                evt_data = {
                    "event_id": latest_event.event_id,
                    "timestamp": getattr(latest_event, "timestamp", time.time()),
                    "threat_level": str(getattr(latest_event, "threat_level", "L0")),
                    "action_type": str(getattr(latest_event, "action_type", "")),
                    "target_category": getattr(latest_event, "target_category", "person"),
                    "target_distance": round(float(getattr(latest_event, "target_distance", 0.0)), 2),
                    "target_dwell_time": round(float(getattr(latest_event, "target_dwell_time", 0.0)), 1),
                    "disturbed_rois": list(getattr(latest_event, "disturbed_rois", [])),
                    "executed_actions": [str(a) for a in getattr(latest_event, "executed_actions", [])],
                    "vlm_summary": getattr(latest_event, "vlm_summary", ""),
                    "evidence_video_path": getattr(latest_event, "evidence_video_path", ""),
                    "push_payload": push_pl
                }
            elif isinstance(latest_event, dict):
                evt_data = latest_event

        with self._action_lock:
            if threat_level:
                self.latest_telemetry["threat_level"] = str(threat_level)
                self.latest_meta["threat_level"] = str(threat_level)
            if state:
                self.latest_telemetry["state"] = str(state)
                self.latest_meta["state"] = str(state)
            if mode:
                self.latest_telemetry["mode"] = str(mode)
            if fps > 0:
                self.latest_telemetry["fps"] = round(float(fps), 1)
                self.latest_meta["fps"] = round(float(fps), 1)
            if ai_fps is not None and ai_fps > 0:
                self.latest_telemetry["ai_fps"] = round(float(ai_fps), 1)
            if e2e_ms > 0:
                self.latest_telemetry["e2e_ms"] = round(float(e2e_ms), 1)
                self.latest_meta["e2e_ms"] = round(float(e2e_ms), 1)

            self.latest_telemetry["frame_idx"] = int(frame_idx)
            self.latest_telemetry["total_frames"] = int(total_frames)
            self.latest_telemetry["uwb_online"] = bool(uwb_online)
            self.latest_telemetry["targets"] = targets_data
            if rois_data:
                self.latest_telemetry["rois"] = rois_data
            if evt_data is not None:
                self.latest_telemetry["latest_event"] = evt_data

    def start(self):
        """启动后台多线程并发 HTTP 服务"""
        if self.server is not None:
            return
        streamer_ref = self

        class CustomHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # 静默请求日志，避免控制台刷屏

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path in ["/", "/index.html"]:
                    body = WEB_UI_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(body)

                elif parsed.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()
                    last_sent_seq = -1
                    with streamer_ref._action_lock:
                        streamer_ref.active_stream_clients = getattr(streamer_ref, "active_stream_clients", 0) + 1
                    try:
                        while True:
                            with streamer_ref.condition:
                                while streamer_ref.frame_seq == last_sent_seq:
                                    streamer_ref.condition.wait(timeout=0.2)
                                frame = streamer_ref.current_jpeg_bytes
                                last_sent_seq = streamer_ref.frame_seq

                            if frame is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(frame)))
                                self.end_headers()
                                self.wfile.write(frame)
                                self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, socket.error):
                        pass
                    finally:
                        with streamer_ref._action_lock:
                            streamer_ref.active_stream_clients = max(0, getattr(streamer_ref, "active_stream_clients", 1) - 1)

                elif parsed.path == "/api/status":
                    with streamer_ref._action_lock:
                        payload = json.dumps(streamer_ref.latest_telemetry, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(payload)

                elif parsed.path == "/api/action":
                    qs = parse_qs(parsed.query)
                    cmd = qs.get("cmd", [""])[0]
                    resp_msg = "系统正在安全封口视频并退出..." if cmd == "quit" else f"指令已执行: {cmd}"
                    resp = {"status": "ok", "msg": resp_msg}
                    if cmd:
                        with streamer_ref._action_lock:
                            streamer_ref.pending_actions.append(cmd)
                    body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(body)

                else:
                    self.send_response(404)
                    self.end_headers()

        try:
            self.server = ThreadingHTTPServer((self.host, self.port), CustomHandler)
            self.server.daemon_threads = True
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

            ips = get_local_ips()
            print("\n" + "=" * 76)
            print(" [*] 【极速 Web 实时中控看板已就绪】(指标原生 DOM 渲染 / 零 CPU 绘图开销):")
            for ip in ips:
                print(f"    --> 请在本地浏览器访问: http://{ip}:{self.port}")
            print("=" * 76 + "\n")
        except Exception as e:
            print(f"[!] 启动 Web 实时推流服务器失败 (端口 {self.port}): {e}")

    def push_frame(self, frame: np.ndarray, quality: int = 85, max_width: int = 1280):
        """
        推送一帧 BGR 画面至网页客户端：
        - 惰性按需编码：首帧初始化后，若当前无任何浏览器客户端连接 /stream.mjpg，直接跳过 cv2.imencode，实现 0ms CPU 零开销
        - 有浏览器观看时自动开启 720p/1080p 高清流畅推流
        """
        if frame is None or frame.size == 0:
            return

        # 若已完成首帧初始化且当前无活跃浏览器视频流连接，直接跳过昂贵的 JPEG 编码
        if self.current_jpeg_bytes is not None and getattr(self, "active_stream_clients", 0) <= 0:
            return

        h, w = frame.shape[:2]
        if w > max_width:
            new_w = max_width
            new_h = int(h * (max_width / float(w)))
            frame_to_encode = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_to_encode = frame

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buffer = cv2.imencode(".jpg", frame_to_encode, encode_param)
        if ret:
            with self.condition:
                self.current_jpeg_bytes = buffer.tobytes()
                self.frame_seq += 1
                self.condition.notify_all()

    def pop_action(self) -> Optional[str]:
        """获取网页端提交的动作指令"""
        with self._action_lock:
            if self.pending_actions:
                return self.pending_actions.pop(0)
        return None

    def stop(self):
        """关闭服务"""
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
