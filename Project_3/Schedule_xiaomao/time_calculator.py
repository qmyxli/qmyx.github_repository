"""
Module1: 时序计算器
职责：根据当日星期筛选匹配任务，输出结构化字典
"""
from datetime import datetime
from constants import BUILTIN_TASKS


class TimeCalculator:
    def __init__(self, task_manager):
        """
        初始化时序计算器
        :param task_manager: 任务台账管理器实例
        """
        self.task_manager = task_manager

    def get_today_tasks(self):
        """
        获取今日任务列表
        :return: 结构化字典
        """
        today = datetime.now()
        weekday = today.isoweekday()  # 1=周一，7=周日
        day_of_month = today.day  # 今日几号

        # 计算今日是第几周（1=第1周，2=第2周...）
        week_of_month = (today.day - 1) // 7 + 1

        # 获取所有任务状态
        task_status = self.task_manager.get_all_task_status()

        # 筛选今日任务
        today_tasks = []
        for task in BUILTIN_TASKS:
            task_name = task["name"]

            # 检查任务是否暂停
            if task_status.get(task_name, {}).get("paused", False):
                continue

            # 检查是否为月任务（特定日期）
            if "day_of_month" in task:
                if task["day_of_month"] != day_of_month:
                    continue
            # 检查是否为周任务（特定周）
            elif "week_of_month" in task:
                if week_of_month not in task["week_of_month"]:
                    continue
                # 还要检查星期是否匹配
                week_config = task["week"]
                if "all" not in week_config and weekday not in week_config:
                    continue
            else:
                # 普通周任务
                week_config = task["week"]
                if "all" not in week_config and weekday not in week_config:
                    continue

            today_tasks.append({
                "name": task_name,
                "time": task["time"]
            })

        # 追加自定义任务（按星期筛选，无月/周特殊规则）
        for task in self.task_manager.get_custom_tasks():
            if task.get("paused", False):
                continue
            week_config = task.get("week", "all")
            if "all" not in week_config and weekday not in week_config:
                continue
            today_tasks.append({
                "name": task["name"],
                "time": task.get("time", "")
            })

        # 判断是否需要合并倒垃圾+领快递
        merge_tips = ""
        if weekday in [2, 5, 7]:  # 周二、周五、周日
            # 检查这两个任务是否都未暂停
            trash_paused = task_status.get("倒垃圾", {}).get("paused", False)
            express_paused = task_status.get("领快递", {}).get("paused", False)
            if not trash_paused and not express_paused:
                merge_tips = "🌙 今晚20点前记得完成下楼事项：倒垃圾 + 领快递"

        return {
            "today_task": today_tasks,
            "merge_tips": merge_tips
        }
    
    def get_task_by_name(self, task_name):
        """
        根据任务名获取任务信息
        :param task_name: 任务名称
        :return: 任务字典或None
        """
        for task in BUILTIN_TASKS:
            if task["name"] == task_name:
                return task
        # 查找自定义任务
        custom_tasks = self.task_manager.get_custom_tasks()
        for task in custom_tasks:
            if task["name"] == task_name:
                return task
        return None
