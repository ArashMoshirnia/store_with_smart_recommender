from celery.schedules import crontab
from celery.task import periodic_task

from recommenders.neural_network import train_model


@periodic_task(name='Update product recommendation model', run_every=crontab(hour='*/3', minute=0))
def update_product_recommendation_model():
    train_model()
