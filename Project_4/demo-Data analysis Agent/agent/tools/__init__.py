"""Agent工具集"""


if __name__ == "__main__":
    # 测试用例：验证工具集可正常生成示例数据
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.tools.data_collector import DataCollector
    data = DataCollector().generate_sample_data(n_orders=5)
    assert 'orders' in data, "示例数据生成失败"
    print(f"测试通过: 工具集生成数据集 {list(data.keys())}")
