"""
天气服务模块
职责：获取今日天气信息，并匹配一句充满活力的鼓励语。
方案：使用 wttr.in 免费天气接口（无需 API Key）+ 内置离线鼓励语词库。
定位：默认查询深圳市宝安区（用中心坐标请求，比拼音更精确）。
约束：网络失败、超时、响应异常时一律降级为离线鼓励语，绝不让晨报中断。
"""
import json
import random
import urllib.request
import urllib.error

from datetime import datetime


class WeatherService:
    def __init__(self, location="22.55,113.88", display_name="深圳·宝安", timeout=8):
        """
        初始化天气服务
        :param location: wttr.in 位置参数（默认深圳宝安区中心坐标，比拼音更精确）
        :param display_name: 显示用的中文地名
        :param timeout: 请求超时秒数
        """
        self.location = location
        self.display_name = display_name
        self.timeout = timeout
        self._api = "https://wttr.in/{location}?format=j1"

    def get_weather_line(self):
        """
        获取"天气 + 鼓励语"整行文案
        :return: 形如 "☀️ 深圳·宝安今日晴，气温12~25℃，天清气朗宜出行。新的一天，元气满满出发吧！"
        """
        weather = self._fetch_weather()
        return self._compose_line(weather)

    # ==================== 天气获取 ====================

    def _fetch_weather(self):
        """
        从 wttr.in 拉取天气（返回 None 表示获取失败）
        :return: dict {"desc": "晴", "low": "12", "high": "25", "city": "北京"} 或 None
        """
        url = self._api.format(location=self.location)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'curl/8.0'})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            return self._parse_weather(data)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    def _parse_weather(self, data):
        """解析 wttr.in JSON 响应为结构化字段（描述统一转中文，地名用固定显示名）"""
        current = data["current_condition"][0]
        desc_raw = current.get("weatherDesc", [{}])[0].get("value", "")
        low = data.get("weather", [{}])[0].get("mintempC", "")
        high = data.get("weather", [{}])[0].get("maxtempC", "")
        return {
            "city": self.display_name,
            "desc": WeatherService._translate_desc(desc_raw.strip()),
            "low": low,
            "high": high,
        }

    @staticmethod
    def _translate_desc(desc):
        """wttr.in 英文天气描述转中文，未匹配时原样返回"""
        desc_map = {
            "sunny": "晴", "clear": "晴", "fine": "晴",
            "partly cloudy": "多云", "partlycloudy": "多云",
            "cloudy": "阴", "overcast": "阴",
            "mist": "薄雾", "fog": "雾", "haze": "霾", "smoky haze": "霾",
            "light rain": "小雨", "light drizzle": "小雨", "drizzle": "小雨",
            "moderate rain": "中雨", "heavy rain": "大雨",
            "rain": "雨", "showers": "阵雨", "light rain shower": "阵雨",
            "patchy rain": "零星小雨", "patchy rain nearby": "零星阵雨",
            "light snow": "小雪", "snow": "雪", "heavy snow": "大雪",
            "sleet": "雨夹雪", "ice pellets": "冰粒",
            "thunderstorm": "雷雨", "thundery outbreaks": "雷雨",
            "windy": "大风", "blizzard": "暴风雪",
        }
        key = desc.lower().strip()
        return desc_map.get(key, desc)

    # ==================== 文案合成 ====================

    def _compose_line(self, weather):
        """
        根据天气情况拼接整行文案
        :param weather: 天气字典（可能为 None）
        :return: 天气与鼓励语组成的单行文本
        """
        encouragement = self._pick_encouragement(weather)
        if weather is None:
            return f"🌟 {encouragement}"

        desc = weather["desc"] or "天气"
        temp = ""
        if weather.get("low") and weather.get("high"):
            temp = f"{weather['low']}~{weather['high']}℃"
        city = weather.get("city") or ""

        parts = []
        prefix = "☀️" if self._is_sunny(desc) else "🌤️"
        parts.append(f"{prefix} {city}今日{desc}")
        if temp:
            parts.append(f"气温{temp}")
        parts.append(encouragement)
        return "，".join(parts)

    def _pick_encouragement(self, weather):
        """根据天气关键词选择鼓励语，天气未知时随机返回"""
        if weather is None:
            return random.choice(self._WORDS["default"])

        desc = weather.get("desc", "")
        for keyword, words in self._WORDS["mapping"].items():
            if keyword in desc:
                return random.choice(words)
        return random.choice(self._WORDS["default"])

    @staticmethod
    def _is_sunny(desc):
        """判断是否晴天"""
        return any(k in desc for k in ("晴", "sunny", "Sunny"))

    # ==================== 离线鼓励语词库 ====================

    _WORDS = {
        "mapping": {
            "雨": [
                "雨天也挡不住好心情，记得带伞，从容出发！",
                "窗外有雨，心里有光，今天也要稳稳前进！",
                "一场雨洗去疲惫，清清爽爽迎接新的一天！",
            ],
            "雪": [
                "落雪的日子适合慢一点，也别忘了把今天过得滚烫！",
                "雪景虽美，注意保暖，暖乎乎地开启今天！",
            ],
            "晴": [
                "阳光正好，元气满满，出发去拥抱这一天吧！",
                "天清气朗，适合行动，把今天过成想要的样子！",
                "好天气配好状态，今天也要闪闪发光！",
            ],
            "云": [
                "多云的日子也别偷懒，稳稳做好每一件小事！",
                "云层挡不住向上的心，今天继续加油！",
            ],
            "阴": [
                "天色虽阴，心情要晴，把今天过得明亮一点！",
                "阴天也是适合专注的日子，安心做好手头的事！",
            ],
            "雾": [
                "雾会散，路会明，慢慢来，一切都刚刚好！",
                "雾天出行注意安全，也别忘了照顾好自己！",
            ],
            "雷": [
                "雷雨交加时宜待在室内，正好安心读书做事！",
                "雷声过后是清新的空气，今天也要清醒地努力！",
            ],
            "风": [
                "风起时更要站稳脚步，今天也要逆风向上！",
                "有风的日子，把烦心事都吹走吧，轻装出发！",
            ],
        },
        "default": [
            "新的一天，元气满满出发吧！",
            "今天也要做自己生活的小太阳！",
            "一步一步来，今天会越来越好！",
            "把今天过成值得记住的一天！",
            "别急，慢慢来，你今天已经很棒了！",
        ],
    }
