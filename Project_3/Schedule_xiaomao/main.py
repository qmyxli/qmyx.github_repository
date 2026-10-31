"""
生活节律小锚 - 入口文件
启动程序，初始化主控调度器并运行
"""
import os
import sys

# 统一控制台编码为 UTF-8，避免 Windows 下 emoji/中文输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 确保当前目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_scheduler import MainScheduler


def main():
    """主函数：启动生活节律小锚"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # 自动检测并创建 data 目录
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"已创建数据目录：{data_dir}")
    
    print("正在初始化生活节律小锚...")
    
    # 创建主控调度器并启动
    scheduler = MainScheduler(data_dir)
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)


if __name__ == "__main__":
    main()
