"""
常量定义模块
包含内置任务、情绪周期、消息模板等固定配置
"""

# 系统内置任务（禁止修改时间，仅允许启用/暂停）
BUILTIN_TASKS = [
    {"name": "洗澡", "week": ["all"], "time": "07:00–07:15"},
    {"name": "读资治通鉴", "week": [1, 2, 3, 4, 5, 7], "time": "07:20–08:00"},
    {"name": "洗衣服", "week": [2, 5], "time": "18:00"},
    {"name": "锻炼身体", "week": [1, 3, 4, 6], "time": "19:00"},
    {"name": "倒垃圾", "week": [2, 5, 7], "time": "20:00前"},
    {"name": "领快递", "week": [2, 5, 7], "time": "20:00前"},
    {"name": "练习SQL编程", "week": [1, 2, 3, 4, 5, 7], "time": "20:30–21:20"},
    {"name": "写日记", "week": ["all"], "time": "21:20–21:40"},
    {"name": "买菜", "week": [6], "time": "08:30"},
    {"name": "买甜点", "week": [6], "time": "11:00"},
    {"name": "逛服装店", "week": [6], "time": "14:00"},
    # 新增任务
    {"name": "写小说", "week": [7], "time": "09:00"},
    {"name": "发布微信公众号文章", "week": [7], "time": "21:00"},
    {"name": "去银行存钱", "week": ["all"], "time": "10:00", "day_of_month": 10},
    {"name": "看英语短剧", "week": [3], "time": "20:00"},
    {"name": "看纪录片", "week": [6], "time": "19:30"},
    {"name": "准备爬山", "week": [5], "time": "22:00", "week_of_month": [1, 3]},
]

# 消息模板
TEMPLATES = {
    "morning_report": """【小锚 · 早安节律播报】
📅 今日日期：{current_date}
{weather_line}
📌 今日固定时段待办：
{today_task_lines}
{merge_tips}
💡 提示：随时可以告诉我暂停/恢复/新增任务。""",

    "task_reminder": """【节律提醒】
⏰ 即将到达时段：{task_name}（{task_time}）""",

    "merge_reminder": """【节律提醒】
🌙 今晚20点前记得完成下楼事项：倒垃圾 + 领快递""",

    "task_update": """✅ 任务清单已更新
📝 本次变更：{change_record}""",

    "task_update_delete_builtin": """说明：内置任务不支持删除，已切换为「暂停提醒」状态。""",

    "completion": """✅ 已标记任务完成。""",

    "evening_review": """【小锚 · 晚间复盘】
📅 今日总结
✅ 已完成：{finish_list}
⭕ 未完成：{unfinish_list}"""
}

# 定时任务配置（时间、触发条件、关联任务）
SCHEDULE_CONFIG = {
    "morning_report": {"time": "09:00", "week": "all", "type": "morning_report"},
    "bath_prep": {"time": "06:55", "week": "all", "type": "task_reminder", "task": "洗澡"},
    "reading": {"time": "07:15", "week": [1, 2, 3, 4, 5, 7], "type": "task_reminder", "task": "读资治通鉴"},
    "laundry": {"time": "17:50", "week": [2, 5], "type": "task_reminder", "task": "洗衣服"},
    "exercise": {"time": "18:50", "week": [1, 3, 4, 6], "type": "task_reminder", "task": "锻炼身体"},
    "merge_errands": {"time": "19:50", "week": [2, 5, 7], "type": "merge_reminder"},
    "buy_groceries": {"time": "08:20", "week": [6], "type": "task_reminder", "task": "买菜"},
    "buy_dessert": {"time": "10:50", "week": [6], "type": "task_reminder", "task": "买甜点"},
    "shop_clothes": {"time": "13:50", "week": [6], "type": "task_reminder", "task": "逛服装店"},
    "sql_practice": {"time": "20:20", "week": [1, 2, 3, 4, 5, 7], "type": "task_reminder", "task": "练习SQL编程"},
    "diary": {"time": "21:10", "week": "all", "type": "task_reminder", "task": "写日记"},
    "evening_review": {"time": "21:00", "week": "all", "type": "evening_review"},
    # 新增任务调度
    "write_novel": {"time": "09:00", "week": [7], "type": "task_reminder", "task": "写小说"},
    "publish_wechat": {"time": "21:00", "week": [7], "type": "task_reminder", "task": "发布微信公众号文章"},
    "bank_deposit": {"time": "10:00", "week": "all", "type": "task_reminder", "task": "去银行存钱", "day_of_month": 10},
    "english_drama": {"time": "20:00", "week": [3], "type": "task_reminder", "task": "看英语短剧"},
    "documentary": {"time": "19:30", "week": [6], "type": "task_reminder", "task": "看纪录片"},
    "hiking_prep": {"time": "22:00", "week": [5], "type": "task_reminder", "task": "准备爬山", "week_of_month": [1, 3]},
}
