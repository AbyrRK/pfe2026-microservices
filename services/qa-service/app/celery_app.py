# app/celery_app.py
from celery import Celery
from app.core.config import settings

celery = Celery(
    "pfe2026",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.research.research_task"]
)
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)