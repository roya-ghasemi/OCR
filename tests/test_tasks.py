# -*- coding: utf-8 -*-
"""Queue setup that has broken the worker before."""
import sys

from ocr_service.config import REPO_DIR
from ocr_service.tasks import _broker_and_backend, celery_app


def test_the_filesystem_queue_lives_in_the_project_whatever_the_working_folder():
    broker, backend, opts = _broker_and_backend("filesystem://.celery_fs_unit")
    try:
        assert broker == "filesystem://"
        assert opts["data_folder_in"].startswith(str(REPO_DIR))
        assert opts["control_folder"].startswith(str(REPO_DIR))
        # a drive letter right after file:// is parsed as host:port and stops the worker
        assert backend.startswith("file://localhost/")
    finally:
        import shutil
        shutil.rmtree(REPO_DIR / ".celery_fs_unit", ignore_errors=True)


def test_windows_workers_use_the_solo_pool():
    if sys.platform == "win32":
        assert celery_app.conf.worker_pool == "solo"
