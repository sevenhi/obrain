#!/usr/bin/env python3
# /data/cold/obrain/scripts/inject-to-honcho.py
# 将 obrain 知识库注入到 Honcho 记忆系统

import sys
import os
import json
import requests
from pathlib import Path

# 加载 .env 环境变量
env_file = Path.home() / ".hermes" / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip('"\'')
                os.environ[key] = value

HONCHO_URL = os.environ.get("HONCHO_API_URL", "http://localhost:8000")
WORKSPACE_ID = "hermes"
PEER_ID = "SevenZhan"  # 主用户 peer

def load_obrain_memory():
    """加载 obrain 核心记忆文件"""
    obrain_dir = Path("/data/cold/obrain")
    
    core_files = {
        "preferences": obrain_dir / "hermes-memory" / "preferences.md",
        "corrections": obrain_dir / "hermes-memory" / "corrections.md",
        "members": obrain_dir / "home-profile" / "members.md",
        "governance": obrain_dir / "automation-rules" / "governance.md",
    }
    
    memory_chunks = []
    for name, path in core_files.items():
        if path.exists():
            content = path.read_text(encoding='utf-8')
            memory_chunks.append(f"## {name.upper()}\n{content}")
        else:
            print(f"⚠️  Missing: {path}")
    
    return "\n\n---\n\n".join(memory_chunks)

def inject_conclusions(memory_text: str):
    """通过 Honcho Conclusions API 批量写入记忆"""
    url = f"{HONCHO_URL}/v3/workspaces/{WORKSPACE_ID}/conclusions"
    
    # 将内容分块 (每块最多 65535 字符，API 限制)
    # 按逻辑分块：每个核心文件作为一个 conclusion
    obrain_dir = Path("/data/cold/obrain")
    core_files = {
        "preferences": obrain_dir / "hermes-memory" / "preferences.md",
        "corrections": obrain_dir / "hermes-memory" / "corrections.md",
        "members": obrain_dir / "home-profile" / "members.md",
        "governance": obrain_dir / "automation-rules" / "governance.md",
    }
    
    conclusions = []
    for name, path in core_files.items():
        if path.exists():
            content = path.read_text(encoding='utf-8')
            conclusions.append({
                "content": f"[{name.upper()}]\n{content}",
                "observer_id": "hermes",
                "observed_id": PEER_ID,
                "session_id": None
            })
    
    if not conclusions:
        print("❌ No conclusions to inject")
        return False
    
    payload = {"conclusions": conclusions}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code in (200, 201):
            result = response.json()
            print(f"✅ Injected {len(conclusions)} conclusions to Honcho")
            return True
        else:
            print(f"❌ Honcho injection failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Honcho at {HONCHO_URL}")
        return False
    except Exception as e:
        print(f"❌ Honcho injection error: {e}")
        return False

def inject_to_builtin_memory(memory_text: str):
    """回退：注入到 Hermes 内置记忆 (MEMORY.md)"""
    try:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        memory_file = hermes_home / "memories" / "MEMORY.md"
        
        existing = ""
        if memory_file.exists():
            existing = memory_file.read_text(encoding='utf-8')
        
        separator = "\n§\n# obrain Knowledge Base\n"
        new_content = existing + separator + memory_text + "\n§\n"
        
        memory_file.write_text(new_content, encoding='utf-8')
        print(f"✅ Memory appended to {memory_file}")
        return True
    except Exception as e:
        print(f"❌ Built-in memory injection error: {e}")
        return False

def main():
    print("=== Injecting obrain memory to Honcho ===")
    print(f"Honcho URL: {HONCHO_URL}")
    print(f"Workspace: {WORKSPACE_ID}")
    print(f"Peer: {PEER_ID}")
    
    memory_text = load_obrain_memory()
    if not memory_text:
        print("❌ No memory content loaded")
        return 1
    
    print(f"Loaded {len(memory_text)} chars from obrain")
    
    # 尝试 Honcho Conclusions API
    if inject_conclusions(memory_text):
        return 0
    
    # 回退到内置记忆
    if inject_to_builtin_memory(memory_text):
        return 0
    
    print("❌ All injection methods failed")
    return 1

if __name__ == "__main__":
    sys.exit(main())