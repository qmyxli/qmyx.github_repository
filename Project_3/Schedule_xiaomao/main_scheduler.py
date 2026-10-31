"""
Module0: 主控调度器 MainScheduler
职责：常驻主线程，启动定时任务调度、监听终端用户输入；
     收集所有结构化数据，调用渲染模块生成消息；
     调用天气模块获取天气与鼓励语，调用桌面通知和飞书通知输出。
约束：不进行业务计算、不拼接文案，只负责流程调度。
"""
import json
import re
import threading
import time
from datetime import datetime, timedelta

from constants import BUILTIN_TASKS, SCHEDULE_CONFIG
from time_calculator import TimeCalculator
from task_manager import TaskManager
from render import Render
from desktop_notifier import DesktopNotifier
from feishu_notifier import FeishuNotifier
from weather_service import WeatherService


class MainScheduler:
    def __init__(self, data_dir="./data"):
        """初始化主控调度器"""
        # 初始化各模块
        self.task_manager = TaskManager(data_dir)
        self.time_calculator = TimeCalculator(self.task_manager)
        self.render = Render()
        self.notifier = DesktopNotifier()
        self.feishu = FeishuNotifier()
        self.weather = WeatherService()

        self.running = True
        self._command_handlers = {}
        self._register_command_handlers()

        # 定时任务调度列表
        self._scheduled_jobs = []

        # 加载飞书配置
        self._load_feishu_config()

    def _load_feishu_config(self):
        """加载飞书配置"""
        import os
        config_file = os.path.join(self.task_manager.data_dir, "feishu_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if config.get("webhook_url"):
                        self.feishu.set_webhook(config["webhook_url"])
            except Exception:
                pass

    def _save_feishu_config(self):
        """保存飞书配置"""
        import os
        config_file = os.path.join(self.task_manager.data_dir, "feishu_config.json")
        config = {"webhook_url": self.feishu.webhook_url}
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    # ==================== 调度核心 ====================
    
    def _register_command_handlers(self):
        """注册终端指令处理函数"""
        self._command_handlers = {
            r"^暂停\s+(.+)$": self._handle_pause,
            r"^恢复\s+(.+)$": self._handle_resume,
            r"^新增\s+(.+)$": self._handle_add,
            r"^删除\s+(.+)$": self._handle_delete,
            r"^查看全部任务清单$|^查看任务$": self._handle_list_all,
            r"^完成【(.+)】$|^完成\[(.+)\]$": self._handle_complete,
            r"^今日任务$": self._handle_today_tasks,
            r"^设置飞书webhook\s+(.+)$": self._handle_set_feishu_webhook,
            r"^测试飞书$": self._handle_test_feishu,
            r"^退出程序$|^退出$": self._handle_exit,
        }
    
    def start(self):
        """启动主控调度器"""
        print("=" * 50)
        print("  生活节律小锚 - 已启动")
        print("  输入 '退出程序' 可安全退出")
        print("  输入 '查看全部任务清单' 查看所有任务")
        print("=" * 50)
        
        # 注册定时任务
        self._setup_schedule()
        
        # 启动终端输入监听线程
        input_thread = threading.Thread(target=self._listen_input, daemon=True)
        input_thread.start()
        
        # 主循环运行定时任务检查
        while self.running:
            self._check_scheduled_jobs()
            time.sleep(1)
    
    def _setup_schedule(self):
        """根据 SCHEDULE_CONFIG 注册定时任务"""
        for key, config in SCHEDULE_CONFIG.items():
            task_time = config["time"]
            week_days = config["week"]
            action_type = config["type"]
            
            # 提取可选的月/周参数
            day_of_month = config.get("day_of_month")
            week_of_month = config.get("week_of_month")
            
            if action_type == "morning_report":
                self._add_job(task_time, "all", self._trigger_morning_report)
            elif action_type == "evening_review":
                self._add_job(task_time, "all", self._trigger_evening_review)
            elif action_type == "merge_reminder":
                self._add_job(task_time, week_days, self._trigger_merge_reminder)
            elif action_type == "task_reminder":
                task_name = config["task"]
                self._add_job(task_time, week_days,
                             lambda tn=task_name: self._trigger_task_reminder(tn),
                             day_of_month=day_of_month,
                             week_of_month=week_of_month)

        # 注册自定义任务提醒（从持久化台账加载）
        for task in self.task_manager.get_custom_tasks():
            if task.get("paused", False):
                continue
            self._register_custom_task_job(task)
    
    def _add_job(self, time_str, week_days, func, day_of_month=None, week_of_month=None, task_name=None):
        """添加定时任务到调度列表"""
        job = {
            "time": time_str,
            "week": week_days,
            "func": func,
            "last_run": None
        }
        if day_of_month is not None:
            job["day_of_month"] = day_of_month
        if week_of_month is not None:
            job["week_of_month"] = week_of_month
        if task_name is not None:
            job["task_name"] = task_name
        self._scheduled_jobs.append(job)

    def _register_custom_task_job(self, task):
        """为单个自定义任务注册定时提醒 job"""
        name = task["name"]
        week = task.get("week", "all")
        time_str = task.get("time", "")
        if not time_str:
            return
        self._add_job(time_str, week,
                      lambda tn=name: self._trigger_task_reminder(tn),
                      task_name=name)

    def _remove_custom_task_job(self, task_name):
        """移除指定自定义任务的定时提醒 job"""
        self._scheduled_jobs = [
            j for j in self._scheduled_jobs if j.get("task_name") != task_name
        ]
    
    def _check_scheduled_jobs(self):
        """检查并执行到期的定时任务"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_weekday = now.isoweekday()  # 1=周一，7=周日
        today_str = now.strftime("%Y-%m-%d")
        day_of_month = now.day  # 今日几号
        
        # 计算今日是第几周（1=第1周，2=第2周...）
        week_of_month = (now.day - 1) // 7 + 1
        
        for job in self._scheduled_jobs:
            # 检查是否已执行过（同一分钟内不重复执行）
            if job["last_run"] == f"{today_str}_{current_time}":
                continue
            
            # 检查时间是否匹配
            if job["time"] != current_time:
                continue
            
            # 检查是否为月任务（特定日期）
            if "day_of_month" in job:
                if job["day_of_month"] != day_of_month:
                    continue
            # 检查是否为周任务（特定周）
            elif "week_of_month" in job:
                if week_of_month not in job["week_of_month"]:
                    continue
                # 还要检查星期是否匹配
                week_config = job["week"]
                if week_config != "all" and current_weekday not in week_config:
                    continue
            else:
                # 普通周任务
                week_config = job["week"]
                if week_config != "all" and current_weekday not in week_config:
                    continue
            
            # 执行任务
            try:
                job["func"]()
                job["last_run"] = f"{today_str}_{current_time}"
            except Exception as e:
                print(f"定时任务执行失败：{e}")
    
    # ==================== 定时触发事件 ====================

    def _output_message(self, message, title="小锚提醒"):
        """统一输出：终端打印 + 桌面弹窗 + 飞书通知"""
        print(f"\n{message}\n")
        self.notifier.show_notification(title, message)
        # 发送到飞书
        if self.feishu.webhook_url:
            self.feishu.send_card(title, message)
    
    def _trigger_morning_report(self):
        """触发早安晨报"""
        # 调用计算模块
        calc_result = self.time_calculator.get_today_tasks()

        today_tasks = calc_result["today_task"]
        merge_tips = calc_result["merge_tips"]

        # 获取今日天气与鼓励语（失败时降级，不影响晨报）
        weather_line = self.weather.get_weather_line()

        message = self.render.render_morning_report(today_tasks, merge_tips, weather_line)
        self._output_message(message, "早安晨报")
    
    def _trigger_task_reminder(self, task_name):
        """触发单个任务提醒"""
        # 检查任务是否暂停（内置与自定义统一判断）
        if self.task_manager.is_task_paused(task_name):
            return

        # 查找任务时间
        task_info = self.time_calculator.get_task_by_name(task_name)
        task_time = task_info["time"] if task_info else ""

        message = self.render.render_task_reminder(task_name, task_time)
        self._output_message(message, f"节律提醒 - {task_name}")
    
    def _trigger_merge_reminder(self):
        """触发合并下楼任务提醒"""
        message = self.render.render_merge_reminder()
        self._output_message(message, "节律提醒 - 下楼事项")

    def _trigger_evening_review(self):
        """触发晚间复盘"""
        # 获取今日任务和完成情况
        calc_result = self.time_calculator.get_today_tasks()
        completed = self.task_manager.get_today_completions()
        today_tasks = calc_result["today_task"]

        # 计算完成/未完成
        today_task_names = [t["name"] for t in today_tasks]
        finish_list = [name for name in today_task_names if name in completed]
        unfinish_list = [name for name in today_task_names if name not in completed]

        message = self.render.render_evening_review(finish_list, unfinish_list)
        self._output_message(message, "晚间复盘")
    
    # ==================== 终端指令处理 ====================
    
    def _listen_input(self):
        """监听终端用户输入"""
        while self.running:
            try:
                user_input = input("")
                if not user_input.strip():
                    continue
                self._process_input(user_input.strip())
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
    
    def _process_input(self, user_input):
        """解析并处理用户输入"""
        # 匹配指令
        for pattern, handler in self._command_handlers.items():
            match = re.match(pattern, user_input)
            if match:
                handler(match)
                return

        # 未匹配指令
        print(f"未识别的指令：{user_input}")
        print("可用指令：暂停/恢复/新增/删除/查看全部任务清单/完成【任务名】/今日任务/设置飞书webhook/测试飞书/退出程序")
    
    def _handle_pause(self, match):
        """处理暂停指令"""
        task_name = match.group(1).strip()
        success, message, change_record = self.task_manager.pause_task(task_name)
        if success:
            output = self.render.render_task_update(message)
            print(f"\n{output}\n")
        else:
            print(f"\n{message}\n")
    
    def _handle_resume(self, match):
        """处理恢复指令"""
        task_name = match.group(1).strip()
        success, message, change_record = self.task_manager.resume_task(task_name)
        if success:
            output = self.render.render_task_update(message)
            print(f"\n{output}\n")
        else:
            print(f"\n{message}\n")
    
    def _handle_add(self, match):
        """处理新增指令"""
        task_info = match.group(1).strip()
        # 解析格式：任务名 周期xxx 时间xxx
        name_match = re.match(r"自定义任务\s*(.+?)[\s,，]周期(.+?)[\s,，]时间(.+)", task_info)
        if not name_match:
            print("\n格式示例：新增 自定义任务 早起喝水，周期all，时间08:00\n")
            return
        
        task_name = name_match.group(1).strip()
        week_str = name_match.group(2).strip()
        time_str = name_match.group(3).strip()

        # 解析周期
        if week_str in ("all", "每日", "每天"):
            week = "all"
        else:
            # 解析数字/汉字列表
            week_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7}
            week = []
            for ch in week_str:
                if ch.isdigit():
                    week.append(int(ch))
                elif ch in week_map:
                    week.append(week_map[ch])
            week = sorted(set(week))
            if not week:
                print("\n无法识别的周期，请使用 all/每日 或 数字(如135)/汉字(如一三五)\n")
                return

        # 校验时间格式并补零为 HH:MM
        if not re.match(r"^\d{1,2}:\d{2}$", time_str):
            print("\n时间格式应为 HH:MM，例如 08:00\n")
            return
        hour, minute = time_str.split(":")
        time_str = f"{int(hour):02d}:{minute}"

        success, message, change_record = self.task_manager.add_custom_task(task_name, week, time_str)
        if success:
            # 立即注册定时提醒
            self._register_custom_task_job({"name": task_name, "week": week, "time": time_str})
            output = self.render.render_task_update(message)
            print(f"\n{output}\n")
        else:
            print(f"\n{message}\n")
    
    def _handle_delete(self, match):
        """处理删除指令"""
        task_name = match.group(1).strip()
        success, message, change_record, is_builtin = self.task_manager.delete_task(task_name)
        if success:
            if not is_builtin:
                self._remove_custom_task_job(task_name)
            output = self.render.render_task_update(message, is_builtin_delete=is_builtin)
            print(f"\n{output}\n")
        else:
            print(f"\n{message}\n")
    
    def _handle_list_all(self, match):
        """处理查看全部任务清单"""
        tasks = self.task_manager.get_all_tasks_list()
        print("\n📋 全部任务清单：")
        print("-" * 40)
        for task in tasks:
            status = "⏸ 已暂停" if task["paused"] else "✅ 启用中"
            week_str = self._format_week(task["week"])
            task_type = "内置" if task["type"] == "builtin" else "自定义"
            print(f"  [{task_type}] {task['name']}  |  {week_str}  |  {task['time']}  |  {status}")
        print("-" * 40)

    def _format_week(self, week):
        """将周期配置格式化为可读文本"""
        if week == "all":
            return "每日"
        if isinstance(week, list):
            if "all" in week:
                return "每日"
            names = ["一", "二", "三", "四", "五", "六", "日"]
            parts = [names[w - 1] for w in week if 1 <= w <= 7]
            return "周" + "、".join(parts) if parts else "每周"
        return f"周{week}"
    
    def _handle_complete(self, match):
        """处理完成指令"""
        task_name = match.group(1) or match.group(2)
        task_name = task_name.strip()
        success, message = self.task_manager.mark_task_completed(task_name)
        if success:
            output = self.render.render_completion()
            print(f"\n{message}")
            print(f"{output}\n")
        else:
            print(f"\n{message}\n")
    
    def _handle_today_tasks(self, match):
        """处理今日任务查询"""
        calc_result = self.time_calculator.get_today_tasks()
        today_tasks = calc_result["today_task"]
        merge_tips = calc_result["merge_tips"]

        completed = self.task_manager.get_today_completions()

        print("\n📌 今日任务：")
        print("-" * 40)
        for task in today_tasks:
            done = "✅" if task["name"] in completed else "⭕"
            print(f"  {done} {task['name']}（{task['time']}）")
        if merge_tips:
            print(f"\n{merge_tips}")
        print("-" * 40)
    
    def _handle_exit(self, match):
        """处理退出指令"""
        print("\n小锚已退出，下次见～")
        self.running = False

    def _handle_set_feishu_webhook(self, match):
        """处理设置飞书 Webhook 指令"""
        webhook_url = match.group(1).strip()
        self.feishu.set_webhook(webhook_url)
        self._save_feishu_config()
        print("\n✅ 飞书 Webhook 已设置")
        print(f"地址: {webhook_url}\n")

    def _handle_test_feishu(self, match):
        """处理测试飞书指令"""
        if not self.feishu.webhook_url:
            print("\n❌ 未配置飞书 Webhook")
            print("请先使用指令: 设置飞书webhook <你的webhook地址>\n")
            return

        success, msg = self.feishu.send_text("🔔 飞书通知测试成功！小锚提醒已连接。")
        if success:
            print("\n✅ 飞书测试消息已发送\n")
        else:
            print(f"\n❌ 发送失败: {msg}\n")
