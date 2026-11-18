import logging
import os
from typing import Iterable, Optional


class Logger:
    @staticmethod
    def close_logger(logger: Optional[logging.Logger]) -> None:
        """关闭并移除 logger 的所有 handler，释放日志文件句柄。"""
        if logger is None:
            return

        for handler in logger.handlers[:]:
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            finally:
                logger.removeHandler(handler)

    @staticmethod
    def format_duration(elapsed_seconds: float) -> str:
        """Format duration with adaptive Chinese units for readability."""
        total_seconds = max(float(elapsed_seconds), 0.0)
        if total_seconds < 60:
            return f"{total_seconds:.2f} 秒"

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours} 小时 {minutes} 分 {seconds:05.2f} 秒"

        return f"{minutes} 分 {seconds:05.2f} 秒"

    @staticmethod
    def log_duration(logger: logging.Logger, stage_name: str, elapsed_seconds: float) -> None:
        """Log a stage duration using adaptive formatting."""
        if logger is None:
            return
        logger.info("%s耗时: %s", stage_name, Logger.format_duration(elapsed_seconds))

    @staticmethod
    def setup_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
    ) -> logging.Logger:
        """Create or reuse a named logger."""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        Logger.close_logger(logger)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False
        return logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Return an existing named logger."""
        return logging.getLogger(name)

    @staticmethod
    def get_first_configured_logger(
        names: Iterable[str],
        fallback_name: Optional[str] = None,
    ) -> logging.Logger:
        """Return the first logger that already has handlers configured."""
        for name in names:
            logger = logging.getLogger(str(name))
            if logger.handlers:
                return logger
        if fallback_name:
            return logging.getLogger(fallback_name)
        names = list(names)
        return logging.getLogger(names[0] if names else "default")

    @staticmethod
    def get_default_logger() -> logging.Logger:
        """Return the default logger."""
        return logging.getLogger("default")
