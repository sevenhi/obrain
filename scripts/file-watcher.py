#!/usr/bin/env python3
# /data/cold/obrain/scripts/file-watcher.py
# 文件监听器：监控 obrain 目录变化，触发增量推送

import os
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime

OBRAIN_DIR = Path("/data/cold/obrain")
SCRIPTS_DIR = OBRAIN_DIR / "scripts"
WATCH_DIRS = [
    OBRAIN_DIR / "hermes-memory",
    OBRAIN_DIR / "device-logs", 
    OBRAIN_DIR / "home-profile",
    OBRAIN_DIR / "automation-rules",
    OBRAIN_DIR / "daily-notes",
    OBRAIN_DIR / "family-docs",
]

# 防抖延迟（秒）
DEBOUNCE_SECONDS = 10
# 最小推送间隔（秒）
MIN_PUSH_INTERVAL = 60

class FileWatcher:
    def __init__(self):
        self.last_event_time = 0
        self.last_push_time = 0
        self.pending_changes = False
        self.timer = None
        self.lock = threading.Lock()
        
    def on_change(self, path: Path):
        """文件变更回调"""
        with self.lock:
            self.last_event_time = time.time()
            self.pending_changes = True
            
            # 取消之前的定时器
            if self.timer:
                self.timer.cancel()
            
            # 设置新的防抖定时器
            self.timer = threading.Timer(DEBOUNCE_SECONDS, self._try_push)
            self.timer.start()
            
            print(f"[{datetime.now().strftime('%F %T')}] 检测到变更: {path}")
    
    def _try_push(self):
        """尝试推送"""
        with self.lock:
            now = time.time()
            
            # 检查防抖时间
            if now - self.last_event_time < DEBOUNCE_SECONDS:
                # 还有新事件，重新设置定时器
                self.timer = threading.Timer(DEBOUNCE_SECONDS, self._try_push)
                self.timer.start()
                return
            
            # 检查最小推送间隔
            if now - self.last_push_time < MIN_PUSH_INTERVAL:
                wait_time = MIN_PUSH_INTERVAL - (now - self.last_push_time)
                print(f"[{datetime.now().strftime('%F %T')}] 推送间隔未到，等待 {wait_time:.0f} 秒")
                self.timer = threading.Timer(wait_time, self._try_push)
                self.timer.start()
                return
            
            if not self.pending_changes:
                return
                
            self.pending_changes = False
            self.last_push_time = now
        
        # 执行推送（在锁外执行避免阻塞）
        self._execute_push()
    
    def _execute_push(self):
        """执行推送脚本"""
        print(f"[{datetime.now().strftime('%F %T')}] 执行增量推送...")
        try:
            result = subprocess.run(
                ["bash", str(SCRIPTS_DIR / "sync-push.sh")],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                print(f"[{datetime.now().strftime('%F %T')}] ✅ 推送成功")
            else:
                print(f"[{datetime.now().strftime('%F %T')}] ❌ 推送失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"[{datetime.now().strftime('%F %T')}] ⏱️ 推送超时")
        except Exception as e:
            print(f"[{datetime.now().strftime('%F %T')}] ❌ 推送异常: {e}")

def main():
    # 确保监听目录存在
    for d in WATCH_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"[{datetime.now().strftime('%F %T')}] 启动 obrain 文件监听器...")
    print(f"监听目录: {[str(d) for d in WATCH_DIRS]}")
    
    watcher = FileWatcher()
    
    # 使用 inotifywait 监听
    watch_dirs = [str(d) for d in WATCH_DIRS if d.exists()]
    if not watch_dirs:
        print("❌ 没有可监听的目录")
        return 1
        
    cmd = [
        "inotifywait", "-m", "-r", "-e", "modify,create,delete,move",
        "--format", "%w%f %e",
        *watch_dirs
    ]
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            # 格式: /path/to/file EVENT
            parts = line.split(" ", 1)
            if len(parts) >= 1:
                path = Path(parts[0])
                # 忽略临时文件和日志文件
                if any(skip in str(path) for skip in [".swp", "~", ".log", "sync-", "conflicts-"]):
                    continue
                watcher.on_change(path)
                
    except FileNotFoundError:
        print("❌ inotifywait 未安装，请运行: apt-get install inotify-tools")
        return 1
    except KeyboardInterrupt:
        print("\n停止监听...")
        if watcher.timer:
            watcher.timer.cancel()
        return 0
    except Exception as e:
        print(f"❌ 监听异常: {e}")
        return 1

if __name__ == "__main__":
    exit(main())