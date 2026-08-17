#!/usr/bin/env python3
# /data/cold/obrain/scripts/inject-to-hermes-memory.py
# 将 obrain 知识库注入到 Hermes 内置记忆系统
# 通过 Honcho/OpenViking 记忆提供者持久化

import sys
import os
from pathlib import Path

# 加载 .env 环境变量
env_file = Path.home() / ".hermes" / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # 去除引号
                value = value.strip('"\'')
                os.environ[key] = value

# 添加 Hermes 路径
sys.path.insert(0, str(Path.home() / ".hermes/hermes-agent"))

from hermes_constants import get_hermes_home

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

def inject_to_honcho(memory_text: str):
    """通过 Honcho/OpenViking 注入记忆"""
    try:
        import requests
        
        # Honcho 本地 API
        honcho_url = os.environ.get("OPENVIKING_ENDPOINT", "http://localhost:1933")
        api_key = os.environ.get("OPENVIKING_API_KEY", "")
        
        if not api_key:
            print("❌ OPENVIKING_API_KEY not set")
            return False
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 使用 conclusions API 写入记忆
        # 将内容分块写入
        chunks = [memory_text[i:i+4000] for i in range(0, len(memory_text), 4000)]
        
        for i, chunk in enumerate(chunks):
            payload = {
                "peer": "user",
                "conclusion": f"obrain knowledge base part {i+1}/{len(chunks)}"
            }
            
            response = requests.post(
                f"{honcho_url}/api/v1/memory/conclude",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ Honcho injection failed (chunk {i+1}): {response.status_code} - {response.text}")
                return False
        
        print("✅ Memory injected to Honcho/OpenViking")
        return True
            
    except ImportError:
        print("⚠️  requests not available, skipping Honcho injection")
        return False
    except Exception as e:
        print(f"❌ Honcho injection error: {e}")
        return False

def inject_to_builtin_memory(memory_text: str):
    """注入到 Hermes 内置记忆 (MEMORY.md)"""
    try:
        hermes_home = get_hermes_home()
        memory_file = hermes_home / "memories" / "MEMORY.md"
        
        # 读取现有记忆
        existing = ""
        if memory_file.exists():
            existing = memory_file.read_text(encoding='utf-8')
        
        # 追加 obrain 内容 (使用 § 分隔符)
        separator = "\n§\n# obrain Knowledge Base\n"
        new_content = existing + separator + memory_text + "\n§\n"
        
        # 写回
        memory_file.write_text(new_content, encoding='utf-8')
        print(f"✅ Memory appended to {memory_file}")
        return True
        
    except Exception as e:
        print(f"❌ Built-in memory injection error: {e}")
        return False

def main():
    print("=== Injecting obrain memory to Hermes ===")
    
    memory_text = load_obrain_memory()
    if not memory_text:
        print("❌ No memory content loaded")
        return 1
    
    print(f"Loaded {len(memory_text)} chars from obrain")
    
    # 尝试 Honcho (优先)
    if inject_to_honcho(memory_text):
        return 0
    
    # 回退到内置记忆
    if inject_to_builtin_memory(memory_text):
        return 0
    
    print("❌ All injection methods failed")
    return 1

if __name__ == "__main__":
    sys.exit(main())