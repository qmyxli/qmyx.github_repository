"""
附加模块：桌面通知工具
职责：调用操作系统API，在电脑右下角弹出原生通知弹窗
"""
import platform
import subprocess
import threading


class DesktopNotifier:
    def __init__(self):
        """初始化桌面通知工具，检测操作系统"""
        self.system = platform.system()
        self._init_notifier()
    
    def _init_notifier(self):
        """初始化对应平台的通知器"""
        if self.system == "Windows":
            try:
                # 尝试导入 win10toast
                from win10toast import ToastNotifier
                self.notifier = ToastNotifier()
                self.use_win10toast = True
            except ImportError:
                try:
                    # 尝试导入 winotify
                    from winotify import Notification
                    self.use_winotify = True
                except ImportError:
                    print("警告：未安装通知库，桌面通知功能不可用")
                    print("请安装：pip install win10toast 或 pip install winotify")
                    self.notifier = None
                    self.use_win10toast = False
                    self.use_winotify = False
        elif self.system == "Darwin":  # macOS
            self.use_macos = True
        else:
            print(f"警告：不支持的操作系统 {self.system}，桌面通知功能不可用")
            self.use_macos = False
    
    def show_notification(self, title, message, duration=10):
        """
        显示桌面通知
        :param title: 通知标题
        :param message: 通知内容
        :param duration: 显示时长（秒）
        """
        # 使用线程异步发送通知，避免阻塞主线程
        thread = threading.Thread(
            target=self._send_notification,
            args=(title, message, duration),
            daemon=True
        )
        thread.start()
    
    def _send_notification(self, title, message, duration):
        """实际发送通知"""
        try:
            if self.system == "Windows":
                self._send_windows_notification(title, message, duration)
            elif self.system == "Darwin":
                self._send_macos_notification(title, message)
            else:
                print(f"[通知] {title}: {message}")
        except Exception as e:
            print(f"发送通知失败：{e}")
            print(f"[通知] {title}: {message}")
    
    def _send_windows_notification(self, title, message, duration):
        """Windows 平台通知"""
        if hasattr(self, 'use_win10toast') and self.use_win10toast and self.notifier:
            # 使用 win10toast
            self.notifier.show_toast(
                title,
                message,
                duration=duration,
                threaded=True
            )
        elif hasattr(self, 'use_winotify') and self.use_winotify:
            # 使用 winotify
            from winotify import Notification
            toast = Notification(
                app_id="生活节律小锚",
                title=title,
                msg=message,
                duration="long"
            )
            toast.show()
        else:
            # 回退到 PowerShell 通知
            self._send_powershell_notification(title, message)
    
    def _send_powershell_notification(self, title, message):
        """使用 PowerShell 发送通知（备选方案）"""
        try:
            # 转义特殊字符
            message_escaped = message.replace('"', '`"').replace('\n', ' ')
            title_escaped = title.replace('"', '`"')
            
            ps_script = f'''
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
            
            $template = @"
            <toast>
                <visual>
                    <binding template="ToastGeneric">
                        <text>{title_escaped}</text>
                        <text>{message_escaped}</text>
                    </binding>
                </visual>
            </toast>
"@
            
            $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
            $xml.LoadXml($template)
            $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("生活节律小锚").Show($toast)
            '''
            
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True,
                timeout=5
            )
        except Exception:
            # 最终回退：只打印到控制台
            print(f"[通知] {title}: {message}")
    
    def _send_macos_notification(self, title, message):
        """macOS 平台通知"""
        try:
            # 使用 osascript 发送通知
            script = f'display notification "{message}" with title "{title}"'
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
        except Exception:
            print(f"[通知] {title}: {message}")
