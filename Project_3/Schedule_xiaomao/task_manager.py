"""
Module2: 任务台账管理器
职责：解析用户指令、权限校验、JSON持久化、返回标准化变更记录
"""
import json
import os
import threading
from datetime import datetime
from constants import BUILTIN_TASKS


class TaskManager:
    def __init__(self, data_dir="./data"):
        """
        初始化任务台账管理器
        :param data_dir: 数据文件目录
        """
        self.data_dir = data_dir
        self.task_status_file = os.path.join(data_dir, "task_status.json")
        self.lock = threading.Lock()  # 简易文件锁
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 初始化或加载任务状态
        self._init_task_status()
    
    def _init_task_status(self):
        """初始化任务状态文件"""
        if not os.path.exists(self.task_status_file):
            # 创建默认状态
            default_status = {
                "builtin_tasks": {},
                "custom_tasks": [],
                "completion_records": {}
            }
            # 初始化内置任务状态
            for task in BUILTIN_TASKS:
                default_status["builtin_tasks"][task["name"]] = {
                    "paused": False,
                    "original_config": task
                }
            self._save_task_status(default_status)
        else:
            # 加载现有状态
            self._load_task_status()
    
    def _load_task_status(self):
        """加载任务状态文件"""
        with self.lock:
            try:
                with open(self.task_status_file, 'r', encoding='utf-8') as f:
                    self.task_status = json.load(f)
            except (json.JSONDecodeError, IOError):
                # 文件损坏，重新初始化
                self._init_task_status()
    
    def _save_task_status(self, status=None):
        """保存任务状态文件"""
        with self.lock:
            if status is None:
                status = self.task_status
            with open(self.task_status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
    
    def get_all_task_status(self):
        """获取所有任务状态"""
        self._load_task_status()
        return self.task_status.get("builtin_tasks", {})
    
    def get_custom_tasks(self):
        """获取自定义任务列表"""
        self._load_task_status()
        return self.task_status.get("custom_tasks", [])

    def is_task_paused(self, task_name):
        """判断任务是否暂停（内置与自定义统一查询）"""
        self._load_task_status()
        if task_name in self.task_status["builtin_tasks"]:
            return self.task_status["builtin_tasks"][task_name].get("paused", False)
        for task in self.task_status["custom_tasks"]:
            if task["name"] == task_name:
                return task.get("paused", False)
        return False
    
    def pause_task(self, task_name):
        """
        暂停任务
        :param task_name: 任务名称
        :return: (success: bool, message: str, change_record: dict)
        """
        self._load_task_status()
        
        # 检查是否为内置任务
        if task_name in self.task_status["builtin_tasks"]:
            self.task_status["builtin_tasks"][task_name]["paused"] = True
            self._save_task_status()
            return True, f"已暂停任务：{task_name}", {
                "action": "pause",
                "task_name": task_name,
                "task_type": "builtin"
            }
        
        # 检查是否为自定义任务
        for task in self.task_status["custom_tasks"]:
            if task["name"] == task_name:
                task["paused"] = True
                self._save_task_status()
                return True, f"已暂停任务：{task_name}", {
                    "action": "pause",
                    "task_name": task_name,
                    "task_type": "custom"
                }
        
        return False, f"未找到任务：{task_name}", None
    
    def resume_task(self, task_name):
        """
        恢复任务
        :param task_name: 任务名称
        :return: (success: bool, message: str, change_record: dict)
        """
        self._load_task_status()
        
        # 检查是否为内置任务
        if task_name in self.task_status["builtin_tasks"]:
            self.task_status["builtin_tasks"][task_name]["paused"] = False
            self._save_task_status()
            return True, f"已恢复任务：{task_name}", {
                "action": "resume",
                "task_name": task_name,
                "task_type": "builtin"
            }
        
        # 检查是否为自定义任务
        for task in self.task_status["custom_tasks"]:
            if task["name"] == task_name:
                task["paused"] = False
                self._save_task_status()
                return True, f"已恢复任务：{task_name}", {
                    "action": "resume",
                    "task_name": task_name,
                    "task_type": "custom"
                }
        
        return False, f"未找到任务：{task_name}", None
    
    def add_custom_task(self, task_name, week, time_str):
        """
        新增自定义任务
        :param task_name: 任务名称
        :param week: 星期配置（如[1,2,3]或"all"）
        :param time_str: 时间字符串
        :return: (success: bool, message: str, change_record: dict)
        """
        self._load_task_status()
        
        # 检查任务名是否已存在
        all_task_names = list(self.task_status["builtin_tasks"].keys())
        for task in self.task_status["custom_tasks"]:
            all_task_names.append(task["name"])
        
        if task_name in all_task_names:
            return False, f"任务名称已存在：{task_name}", None
        
        # 新增自定义任务
        new_task = {
            "name": task_name,
            "week": week,
            "time": time_str,
            "paused": False,
            "created_at": datetime.now().isoformat()
        }
        self.task_status["custom_tasks"].append(new_task)
        self._save_task_status()
        
        return True, f"已新增自定义任务：{task_name}", {
            "action": "add",
            "task_name": task_name,
            "task_type": "custom",
            "task_config": new_task
        }
    
    def delete_task(self, task_name):
        """
        删除任务（内置任务只能暂停，自定义任务可删除）
        :param task_name: 任务名称
        :return: (success: bool, message: str, change_record: dict, is_builtin: bool)
        """
        self._load_task_status()
        
        # 检查是否为内置任务
        if task_name in self.task_status["builtin_tasks"]:
            # 内置任务不能删除，只能暂停
            self.task_status["builtin_tasks"][task_name]["paused"] = True
            self._save_task_status()
            return True, f"内置任务不支持删除，已为你暂停：{task_name}", {
                "action": "pause",
                "task_name": task_name,
                "task_type": "builtin"
            }, True
        
        # 检查是否为自定义任务
        for i, task in enumerate(self.task_status["custom_tasks"]):
            if task["name"] == task_name:
                deleted_task = self.task_status["custom_tasks"].pop(i)
                self._save_task_status()
                return True, f"已删除自定义任务：{task_name}", {
                    "action": "delete",
                    "task_name": task_name,
                    "task_type": "custom",
                    "task_config": deleted_task
                }, False
        
        return False, f"未找到任务：{task_name}", None, False
    
    def mark_task_completed(self, task_name, date=None):
        """
        标记任务完成
        :param task_name: 任务名称
        :param date: 日期（默认今天）
        :return: (success: bool, message: str)
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self._load_task_status()
        
        # 初始化完成记录
        if date not in self.task_status["completion_records"]:
            self.task_status["completion_records"][date] = []
        
        # 检查是否已标记
        if task_name in self.task_status["completion_records"][date]:
            return False, f"任务今日已标记完成：{task_name}"
        
        # 标记完成
        self.task_status["completion_records"][date].append(task_name)
        self._save_task_status()
        
        return True, f"已标记任务完成：{task_name}"
    
    def get_today_completions(self, date=None):
        """获取今日完成的任务列表"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self._load_task_status()
        return self.task_status["completion_records"].get(date, [])
    
    def get_all_tasks_list(self):
        """获取所有任务列表（内置+自定义）"""
        self._load_task_status()
        
        tasks = []
        
        # 内置任务
        for task_name, status in self.task_status["builtin_tasks"].items():
            tasks.append({
                "name": task_name,
                "type": "builtin",
                "paused": status["paused"],
                "week": status["original_config"]["week"],
                "time": status["original_config"]["time"]
            })
        
        # 自定义任务
        for task in self.task_status["custom_tasks"]:
            tasks.append({
                "name": task["name"],
                "type": "custom",
                "paused": task["paused"],
                "week": task["week"],
                "time": task["time"]
            })
        
        return tasks
