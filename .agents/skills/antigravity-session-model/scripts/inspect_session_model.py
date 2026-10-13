#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Antigravity Session Model Inspector
===================================
Directly queries Antigravity conversation databases to extract:
1. UI Selected Model (Model Enum & Display Label)
2. Language Server Internal Model Definition
3. Routing Profile (model_config.json profile mapping)
4. Backend Returned Model (Field 19 in gen_metadata protobuf)
5. Token Usage Statistics (input, output, thought, cached)
"""

import os
import sys
import glob
import sqlite3
import argparse
import json
from datetime import datetime

# Ensure proper utf-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Model mapping table decoded from language_server.exe ModelInfo and model_config.json
KNOWN_MODELS = {
    1318: {
        "ui_name": "Gemini 3.8 Flash (High)",
        "proto_enum": "MODEL_PLACEHOLDER_M318",
        "internal_name": "gemini-3.8-flash-medium",
        "routed_profile": "MODEL_PLACEHOLDER_M124"
    },
    1319: {
        "ui_name": "Gemini 3.8 Flash (Low)",
        "proto_enum": "MODEL_PLACEHOLDER_M319",
        "internal_name": "gemini-3.8-flash-low",
        "routed_profile": "MODEL_PLACEHOLDER_M124"
    },
    1320: {
        "ui_name": "Gemini 3.7 Flash (High)",
        "proto_enum": "MODEL_PLACEHOLDER_M320",
        "internal_name": "gemini-3.7-flash-high",
        "routed_profile": "MODEL_PLACEHOLDER_M124"
    },
    1298: {
        "ui_name": "Gemini 3.7 Flash (Medium)",
        "proto_enum": "MODEL_PLACEHOLDER_M298",
        "internal_name": "gemini-3.7-flash-medium",
        "routed_profile": "MODEL_PLACEHOLDER_M124"
    },
    1299: {
        "ui_name": "Gemini 3.7 Flash (Low)",
        "proto_enum": "MODEL_PLACEHOLDER_M299",
        "internal_name": "gemini-3.7-flash-low",
        "routed_profile": "MODEL_PLACEHOLDER_M124"
    },
    1059: {
        "ui_name": "Gemini 3.8 Flash (Native High)",
        "proto_enum": "MODEL_PLACEHOLDER_M59",
        "internal_name": "gemini-3.8-flash-high",
        "routed_profile": "MODEL_PLACEHOLDER_M124"
    },
    1018: {
        "ui_name": "Gemini 2.5 Flash",
        "proto_enum": "MODEL_PLACEHOLDER_M18",
        "internal_name": "gemini-2.5-flash",
        "routed_profile": "MODEL_GOOGLE_GEMINI_2_5_FLASH"
    },
    1037: {
        "ui_name": "Gemini 3.1 Pro",
        "proto_enum": "MODEL_PLACEHOLDER_M37",
        "internal_name": "gemini-3.1-pro-low-thinking",
        "routed_profile": "MODEL_GOOGLE_GEMINI_2_5_PRO"
    }
}

def decode_proto(data):
    """Minimal protobuf decoder returning (field_num, wire_type, value) tuples."""
    i = 0
    fields = []
    n = len(data)
    while i < n:
        key = 0
        shift = 0
        while i < n:
            b = data[i]
            i += 1
            key |= (b & 0x7f) << shift
            shift += 7
            if (b & 0x80) == 0:
                break
        field_num = key >> 3
        wire_type = key & 0x7
        if wire_type == 0:  # varint
            val = 0
            shift = 0
            while i < n:
                b = data[i]
                i += 1
                val |= (b & 0x7f) << shift
                shift += 7
                if (b & 0x80) == 0:
                    break
            fields.append((field_num, "varint", val))
        elif wire_type == 1:  # 64-bit
            val = data[i:i+8]
            i += 8
            fields.append((field_num, "64bit", val))
        elif wire_type == 2:  # length-delimited
            length = 0
            shift = 0
            while i < n:
                b = data[i]
                i += 1
                length |= (b & 0x7f) << shift
                shift += 7
                if (b & 0x80) == 0:
                    break
            val = data[i:i+length]
            i += length
            fields.append((field_num, "bytes", val))
        elif wire_type == 5:  # 32-bit
            val = data[i:i+4]
            i += 4
            fields.append((field_num, "32bit", val))
        else:
            break
    return fields

def parse_metadata_blob(blob):
    """Extract model routing info from gen_metadata blob."""
    top_fields = decode_proto(blob)
    res = {
        "requested_enum": None,
        "model_enum": None,
        "returned_model": None,
        "request_id": None,
        "last_step_index": None,
        "tokens": {}
    }
    
    for f_num, w_type, f_val in top_fields:
        if f_num == 1 and w_type == "bytes":
            subfields = decode_proto(f_val)
            for sf_num, sf_type, sf_val in subfields:
                if sf_num == 3 and sf_type == "varint":
                    res["requested_enum"] = sf_val
                elif sf_num == 19 and sf_type == "bytes":
                    try:
                        res["returned_model"] = sf_val.decode("utf-8")
                    except Exception:
                        pass
                elif sf_num == 20 and sf_type == "bytes":
                    kv_fields = decode_proto(sf_val)
                    k_str, v_str = None, None
                    for kf_num, _, kf_val in kv_fields:
                        if kf_num == 1:
                            try: k_str = kf_val.decode("utf-8")
                            except Exception: pass
                        elif kf_num == 2:
                            try: v_str = kf_val.decode("utf-8")
                            except Exception: pass
                    if k_str == "model_enum":
                        res["model_enum"] = v_str
                    elif k_str == "request_id":
                        res["request_id"] = v_str
                    elif k_str == "last_step_index":
                        res["last_step_index"] = v_str
                elif sf_num == 4 and sf_type == "bytes":
                    token_fields = decode_proto(sf_val)
                    for tf_num, tf_type, tf_val in token_fields:
                        if tf_type == "varint":
                            if tf_num == 2: res["tokens"]["input"] = tf_val
                            elif tf_num == 3: res["tokens"]["output"] = tf_val
                            elif tf_num == 5: res["tokens"]["cached"] = tf_val
                            elif tf_num == 10: res["tokens"]["thought"] = tf_val
    return res

def inspect_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT idx, data FROM gen_metadata ORDER BY idx DESC LIMIT 1")
        row = c.fetchone()
        if not row:
            return None
        idx, blob = row
        info = parse_metadata_blob(blob)
        info["db_path"] = db_path
        info["conversation_id"] = os.path.splitext(os.path.basename(db_path))[0]
        info["last_idx"] = idx
        info["mtime"] = datetime.fromtimestamp(os.path.getmtime(db_path)).strftime("%Y-%m-%d %H:%M:%S")
        
        req_enum = info["requested_enum"]
        if req_enum in KNOWN_MODELS:
            info["ui_name"] = KNOWN_MODELS[req_enum]["ui_name"]
            info["ls_internal_name"] = KNOWN_MODELS[req_enum]["internal_name"]
            info["routed_profile"] = KNOWN_MODELS[req_enum]["routed_profile"]
        else:
            info["ui_name"] = info["model_enum"] or f"Enum_{req_enum}"
            info["ls_internal_name"] = "N/A"
            info["routed_profile"] = "N/A"
        return info
    except Exception as e:
        return {"error": str(e), "db_path": db_path}

def main():
    parser = argparse.ArgumentParser(description="Inspect Antigravity conversation models")
    parser.add_argument("--conv-id", help="Specific conversation ID to inspect")
    parser.add_argument("--all", action="store_true", help="Inspect all conversation databases")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--markdown", action="store_true", help="Output in Markdown table format")
    args = parser.parse_args()

    user_home = os.path.expanduser("~")
    conv_dir = os.path.join(user_home, ".gemini", "antigravity", "conversations")

    if not os.path.exists(conv_dir):
        print(f"Error: Conversations directory not found at {conv_dir}", file=sys.stderr)
        sys.exit(1)

    if args.conv_id:
        target = os.path.join(conv_dir, f"{args.conv_id}.db")
        if not os.path.exists(target):
            print(f"Error: Database {target} does not exist.", file=sys.stderr)
            sys.exit(1)
        dbs = [target]
    elif args.all:
        dbs = sorted(glob.glob(os.path.join(conv_dir, "*.db")), key=os.path.getmtime, reverse=True)
    else:
        dbs_list = glob.glob(os.path.join(conv_dir, "*.db"))
        if not dbs_list:
            print("No conversation databases found.", file=sys.stderr)
            sys.exit(1)
        dbs = [max(dbs_list, key=os.path.getmtime)]

    results = []
    for db_path in dbs:
        info = inspect_db(db_path)
        if info and "error" not in info:
            results.append(info)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.markdown:
        print("| 会话 ID | 最后更新时间 | 前端选择模型 | LS 内部映射 | 实际路由 Profile | 后端返回模型 (真实执行) |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in results:
            print(f"| `{r['conversation_id'][:8]}...` | {r['mtime']} | **{r['ui_name']}** | `{r['ls_internal_name']}` | `{r['routed_profile']}` | **`{r['returned_model']}`** |")
        return

    for res in results:
        print("=" * 70)
        print(f"会话 ID:        {res['conversation_id']}")
        print(f"最后修改时间:    {res['mtime']}")
        print(f"生成序号:        idx={res['last_idx']} (step={res['last_step_index']})")
        print("-" * 70)
        print(f"1. 前端选择模型:  {res['ui_name']} ({res['model_enum']} / {res['requested_enum']})")
        print(f"2. LS 内部定义:  {res['ls_internal_name']}")
        print(f"3. 实际路由配置:  {res['routed_profile']}")
        print(f"4. 后端返回模型:  {res['returned_model']}  <-- [服务端真实执行模型]")
        if res['tokens']:
            t = res['tokens']
            print(f"5. Token 消耗统计: 输入={t.get('input', 0)}, 输出={t.get('output', 0)}, 思考={t.get('thought', 0)}, 缓存={t.get('cached', 0)}")
        print("=" * 70)

if __name__ == "__main__":
    main()
