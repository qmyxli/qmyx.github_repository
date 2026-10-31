"""日志配置 - 控制台 + logs/app.log（按天滚动）"""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

_configured = False
_log_dir: Optional[Path] = None

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(log_dir: Optional[Path] = None) -> Path:
    """配置根日志器：控制台 + 文件输出。

    幂等：重复调用不会重复添加 handler。应用各模块的 `logging.getLogger(__name__)`
    日志都会传播到根日志器，从而同时输出到控制台和 logs/app.log。

    Returns:
        日志目录路径
    """
    global _configured, _log_dir
    if _configured:
        return _log_dir

    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = TimedRotatingFileHandler(
        log_dir / "app.log",
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True
    _log_dir = log_dir
    return log_dir


if __name__ == "__main__":
    # 测试用例：配置日志并确认 app.log 文件生成
    import tempfile
    log_dir = setup_logging(Path(tempfile.mkdtemp()))
    logging.getLogger(__name__).info("这是一条测试日志")
    assert (log_dir / "app.log").exists(), "日志文件未生成"
    print(f"测试通过: 日志文件已生成 {log_dir / 'app.log'}")
