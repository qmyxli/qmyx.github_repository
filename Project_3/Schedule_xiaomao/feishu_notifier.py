"""
飞书通知模块
职责：通过飞书机器人发送任务提醒消息
"""
import json
import urllib.request
import urllib.error


class FeishuNotifier:
    def __init__(self, webhook_url=None):
        """
        初始化飞书通知器
        :param webhook_url: 飞书机器人 Webhook 地址
        """
        self.webhook_url = webhook_url

    def set_webhook(self, webhook_url):
        """
        设置 Webhook 地址
        :param webhook_url: 飞书机器人 Webhook 地址
        """
        self.webhook_url = webhook_url

    def send_text(self, text):
        """
        发送文本消息
        :param text: 消息文本
        :return: (success: bool, message: str)
        """
        if not self.webhook_url:
            return False, "未配置飞书 Webhook 地址"

        payload = {
            "msg_type": "text",
            "content": {
                "text": text
            }
        }

        return self._send_request(payload)

    def send_rich_text(self, title, content_lines):
        """
        发送富文本消息
        :param title: 消息标题
        :param content_lines: 内容行列表，每行为文本列表
        :return: (success: bool, message: str)
        """
        if not self.webhook_url:
            return False, "未配置飞书 Webhook 地址"

        content = []
        for line in content_lines:
            content.append(line)

        payload = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": content
                    }
                }
            }
        }

        return self._send_request(payload)

    def send_card(self, title, content, color="blue"):
        """
        发送卡片消息
        :param title: 卡片标题
        :param content: 卡片内容（Markdown 格式）
        :param color: 标题颜色（blue/green/orange/red/purple/indigo/turquoise/yellow/grey/wathet）
        :return: (success: bool, message: str)
        """
        if not self.webhook_url:
            return False, "未配置飞书 Webhook 地址"

        color_map = {
            "blue": "blue",
            "green": "green",
            "orange": "orange",
            "red": "red",
            "purple": "purple",
            "indigo": "indigo",
            "turquoise": "turquoise",
            "yellow": "yellow",
            "grey": "grey",
            "wathet": "wathet"
        }

        template_color = color_map.get(color, "blue")

        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": title
                    },
                    "template": template_color
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content
                    }
                ]
            }
        }

        return self._send_request(payload)

    def _send_request(self, payload):
        """
        发送 HTTP 请求
        :param payload: 请求体
        :return: (success: bool, message: str)
        """
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('code') == 0 or result.get('StatusCode') == 0:
                    return True, "发送成功"
                else:
                    return False, f"发送失败: {result.get('msg', '未知错误')}"
        except urllib.error.URLError as e:
            return False, f"网络错误: {str(e)}"
        except Exception as e:
            return False, f"发送异常: {str(e)}"
