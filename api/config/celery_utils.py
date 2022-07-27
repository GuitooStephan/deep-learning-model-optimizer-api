from celery import current_app as current_celery_app
from celery.result import AsyncResult

from .celery_config import settings


def create_celery():
    celery_app = current_celery_app
    celery_app.config_from_object(settings, namespace='CELERY')
    celery_app.conf.update(task_track_started=True)
    celery_app.conf.update(task_serializer='json')
    celery_app.conf.update(result_serializer='json')
    celery_app.conf.update(accept_content=['json'])
    celery_app.conf.update(result_expires=1000)
    celery_app.conf.update(result_persistent=True)
    celery_app.conf.update(worker_send_task_events=False)
    celery_app.conf.update(worker_prefetch_multiplier=1)
    celery_app.conf.update(task_soft_time_limit=10000)
    celery_app.conf.update(task_time_limit=10000)
    celery_app.conf.update(task_acks_late=True)
    # celery_app.conf.update(broker_heartbeat=0)
    celery_app.conf.update(broker_connection_timeout=100000)
    # celery_app.conf.update(broker_pool_limit=None)

    return celery_app


def get_task_info(task_id):
    """
    return task info for the given task_id
    """
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": 'DONE' if task_result.ready() else 'PENDING',
        "task_result": task_result.get() if task_result.ready() else None,
    }
    return result
