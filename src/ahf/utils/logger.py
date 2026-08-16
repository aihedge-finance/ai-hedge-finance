import asyncio
import logging
import sys
from pathlib import Path

from loguru import logger

from ahf.utils.utils import get_project_root

log_rotation = "100 MB"
log_retention = "120 months"

# def log_server_info():
#     logger.info("Starting AI Art API Server")
#     logger.info(f"Version: {settings.version}")
#     logger.info(f"Baseurl: {settings.app_baseurl}")
#     logger.info(f"Host: {settings.host}")
#     logger.info(f"Port: {settings.port}")
#     logger.info(f"Debug: {settings.debug}")
#     logger.info(f"Site title: {settings.app_site_title}")
#     logger.info(f"Data folder: {settings.app_data_folder}")
#     logger.info(f"Database: Cassandra")
#     logger.info(f"Service fee: {settings.app_service_fee}")


# def initialize_server_websocket_logger() -> Callable:
#     super_user_hash = sha256(settings.super_user.encode("utf-8")).hexdigest()
#
#     server_log_queue: asyncio.Queue = asyncio.Queue()
#
#     async def update_websocket_server_log():
#         while settings.app_running:
#             msg = await server_log_queue.get()
#             await websocket_updater(super_user_hash, msg)
#
#     logger.add(
#         lambda msg: server_log_queue.put_nowait(msg),
#         format=Formatter().format,
#     )
#
#     return update_websocket_server_log


def configure_logger(app_data_folder: str, is_debug: bool, user_id:str = None, bot_id: str = None) -> None:
    add_extra = False
    if user_id and bot_id:
        add_extra = True
        user_id = f"{user_id[:4]}...{user_id[-4:]}"
        bot_id = f"{bot_id[:4]}...{bot_id[-4:]}"
        logger.configure(extra={"user_id": user_id, "bot_id": bot_id})
    project_root = get_project_root()
    logger.remove()
    log_level: str = "DEBUG" if is_debug else "INFO"
    formatter = Formatter(is_debug, add_extra)
    logger.add(sys.stdout, level=log_level, format=formatter.format, enqueue=True)

    # https://github.com/pyca/bcrypt/issues/684#issuecomment-1858400267
    # logging.getLogger('passlib').setLevel(logging.ERROR)

    # if settings.enable_log_to_file:
    logger.add(
        Path(project_root, "logs", app_data_folder, "app.log"),
        colorize=True,
        rotation=log_rotation,
        retention=log_retention,
        level="INFO",
        format=formatter.format,
        enqueue = True,
    )
    logger.add(
        Path(project_root, "logs", app_data_folder, "debug.log"),
        colorize=True,
        rotation=log_rotation,
        retention=log_retention,
        level="DEBUG",
        format=formatter.format,
        enqueue = True,
    )

    """logging.getLogger("uvicorn").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.error").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.error").propagate = False

    logging.getLogger("cassandra").handlers = [InterceptHandler()]

    logging.getLogger("sqlalchemy").handlers = [InterceptHandler()]
    logging.getLogger("sqlalchemy.engine.base").handlers = [InterceptHandler()]
    logging.getLogger("sqlalchemy.engine.base").propagate = False
    logging.getLogger("sqlalchemy.engine.base.Engine").handlers = [InterceptHandler()]
    logging.getLogger("sqlalchemy.engine.base.Engine").propagate = False"""


class Formatter:
    def __init__(self, is_debug: bool = False, add_extra: bool = False):
        self.padding = 0
        self.minimal_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SS} | {level} | "
            "{message}" + (" | {extra}\n" if add_extra else "\n")
        )

        if is_debug:
            self.fmt = (
                "{time:YYYY-MM-DD HH:mm:ss.SS} | "
                "{level: <4} | "
                "{name}:{function}:{line} | "
                "{message}" + (" | {extra}\n" if add_extra else "\n")
            )
        else:
            self.fmt = self.minimal_fmt


    def format(self, record):
        function = "{function}".format(**record)
        if function == "emit":  # uvicorn logs
            return self.minimal_fmt
        return self.fmt
2


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.log(level, record.getMessage())
