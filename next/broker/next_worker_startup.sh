#!/bin/sh
#export i=1
#celery -A next.broker.celery_app worker -l info --loglevel=WARNING --concurrency=1 -n hash_worker_${i}@${HOSTNAME} -Q Hash@${HOSTNAME} -- celeryd.prefetch_multiplier=1 &
#celery -A next.broker.celery_app worker -l info --loglevel=WARNING --concurrency=1 -n features_worker_${i}@${HOSTNAME} -Q Features@${HOSTNAME} -- celeryd.prefetch_multiplier=1 &

echo AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

for i in `seq 1 $CELERY_ASYNC_WORKER_COUNT`
do
    celery -A next.broker.celery_app worker -l info --loglevel=WARNING --concurrency=${CELERY_THREADS_PER_ASYNC_WORKER} -n async_worker_${i}@${HOSTNAME} -Q async@${HOSTNAME} -- celeryd.prefetch_multiplier=${CELERY_ASYNC_WORKER_PREFETCH} &
done

for i in `seq 1 $CELERY_DASHBOARD_WORKER_COUNT`
do
    celery -A next.broker.celery_app worker -l info --loglevel=WARNING --concurrency=${CELERY_THREADS_PER_DASHBOARD_WORKER} -n dashboard_worker_${i}@${HOSTNAME} -Q dashboard@${HOSTNAME} -- celeryd.prefetch_multiplier=${CELERY_DASHBOARD_WORKER_PREFETCH} &
done

for i in `seq 1 $CELERY_SYNC_WORKER_COUNT`
do
    celery -A next.broker.celery_app worker -l info --loglevel=WARNING --concurrency=1 -n sync_worker_${i}@${HOSTNAME} -Q sync_queue_${i}@${HOSTNAME} -- celeryd.prefetch_multiplier=1  &
done

echo STARTING HASH WORKER
celery -A next.broker.celery_app worker -l info --loglevel=WARNING --concurrency=1 -n Hash_Worker@${HOSTNAME} -Q Hash_Queue@${HOSTNAME} -- celeryd.prefetch_multiplier=10 &

wait