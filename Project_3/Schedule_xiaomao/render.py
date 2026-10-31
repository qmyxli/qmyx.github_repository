"""
Module4: 消息模板渲染器
职责：严格按照预设模板填充变量，生成最终展示文本
"""
from datetime import datetime
from constants import TEMPLATES


class Render:
    def __init__(self):
        """初始化渲染器"""
        self.templates = TEMPLATES

    def render_morning_report(self, today_task, merge_tips, weather_line=""):
        """
        渲染早安晨报
        :param today_task: 今日任务列表 [{"name": "任务名", "time": "时间"}]
        :param merge_tips: 合并提示文本
        :param weather_line: 天气与鼓励语文本
        :return: 渲染后的文本
        """
        current_date = datetime.now().strftime("%Y年%m月%d日")

        # 格式化任务列表
        if today_task:
            task_lines = "\n".join([f"  • {task['name']}（{task['time']}）" for task in today_task])
        else:
            task_lines = "  • 今日无固定时段任务"

        template = self.templates["morning_report"]
        return template.format(
            current_date=current_date,
            weather_line=weather_line if weather_line else "",
            today_task_lines=task_lines,
            merge_tips=merge_tips if merge_tips else ""
        )

    def render_task_reminder(self, task_name, task_time):
        """
        渲染任务提醒
        :param task_name: 任务名称
        :param task_time: 任务时间
        :return: 渲染后的文本
        """
        template = self.templates["task_reminder"]
        return template.format(
            task_name=task_name,
            task_time=task_time
        )

    def render_merge_reminder(self):
        """
        渲染合并下楼任务提醒
        :return: 渲染后的文本
        """
        return self.templates["merge_reminder"]

    def render_task_update(self, change_record, is_builtin_delete=False):
        """
        渲染任务操作反馈
        :param change_record: 变更记录描述
        :param is_builtin_delete: 是否为删除内置任务
        :return: 渲染后的文本
        """
        result = self.templates["task_update"].format(change_record=change_record)

        if is_builtin_delete:
            result += "\n\n" + self.templates["task_update_delete_builtin"]

        return result

    def render_completion(self):
        """
        渲染任务完成反馈
        :return: 渲染后的文本
        """
        return self.templates["completion"]

    def render_evening_review(self, finish_list, unfinish_list):
        """
        渲染晚间小结
        :param finish_list: 已完成任务列表
        :param unfinish_list: 未完成任务列表
        :return: 渲染后的文本
        """
        # 格式化列表
        finish_text = "、".join(finish_list) if finish_list else "无"
        unfinish_text = "、".join(unfinish_list) if unfinish_list else "无"

        template = self.templates["evening_review"]
        return template.format(
            finish_list=finish_text,
            unfinish_list=unfinish_text
        )
