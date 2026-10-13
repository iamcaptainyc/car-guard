---
name: antigravity-session-model
description: >-
  Use this skill whenever the user asks about the model used in Antigravity conversations,
  checks the current or recent conversation's requested model, investigates the backend returned
  model (请求返回模型 / 实际执行模型 / 路由模型), queries token usage, or inquires about model discrepancies
  (e.g., "查一下会话模型", "当前用的什么模型", "返回模型是什么", "调查会话模型", "检查最近会话的模型").
  Provides an instant inspection script to extract exact protobuf response metadata from the
  conversation database.
---

# Antigravity 会话模型检查指南 (Session Model Inspector)

本 Skill 用于**快速、精准查询** Antigravity 最近或当前会话的**真实请求模型**与**后端返回模型（实际执行模型）**，避免重复进行繁琐的二进制逆向与数据库手动解码。

---

## 核心原理与全链路映射

当用户在 Antigravity 中提问时，请求与模型经过以下链路：

1. **前端 UI 选择**：用户在界面下拉菜单选择（例如 `Gemini 3.8 Flash (High)`）。
2. **状态与枚举**：保存于 `state.vscdb` 和 `antigravity_state.pbtxt` 中的枚举（如 `MODEL_PLACEHOLDER_M318`，枚举 ID `1318`）。
3. **Language Server (`language_server.exe`) 内部注册**：
   - 内部 `ModelInfo` 结构体表中，ID `1318` 注册的模型名为 `gemini-3.8-flash-medium`（原生 High 为 1059）。
   - 内置配置表 `model_config.json` 将 `MODEL_PLACEHOLDER_M318` 统一路由映射到 Profile `MODEL_PLACEHOLDER_M124`。
4. **后端 API 与返回响应**：
   - 调用 `daily-cloudcode-pa.googleapis.com` 或生产 API。
   - 每个生成步骤在本地数据库 `~/.gemini/antigravity/conversations/<uuid>.db` 的 `gen_metadata` 表中持久化保存。
   - 解码其 Protobuf 结构：
     - **Field 3**: 请求发送的枚举 ID（如 `1318`）。
     - **Field 19**: **后端实际路由执行并返回的模型标识**（例如 `gemini-3.8-flash-n`）。
     - **Field 20**: 包含 `model_enum`、`request_id` 等键值对。

---

## 快速调用方式

使用内置的 Python 检查工具：

```bash
# 1. 检查最近一次活跃会话（默认）
python "scripts\inspect_session_model.py"

# 2. 检查指定会话 ID
python "scripts\inspect_session_model.py" --conv-id <CONVERSATION_UUID>

# 3. 输出 Markdown 表格汇总所有历史会话
python "scripts\inspect_session_model.py" --all --markdown

# 4. 输出 JSON 结构化数据
python "scripts\inspect_session_model.py" --json
```

---

## 结果解读与回复规范

当向用户汇报时，应区分**请求模型**与**返回模型**：

1. **请求模型（前端与 LS 映射）**：
   - 前端显示名称（如 `Gemini 3.8 Flash (High)`）
   - Protobuf 枚举名称与 ID（如 `MODEL_PLACEHOLDER_M318` / `1318`）
   - LS 内部注册名称（如 `gemini-3.8-flash-medium`）
2. **返回模型（后端真实执行）**：
   - 数据库 Field 19 字段（当前版本为 **`gemini-3.8-flash-n`**）
3. **Token 与上下文消耗**：
   - 输入 tokens、输出 tokens、思考 tokens、缓存 tokens。
