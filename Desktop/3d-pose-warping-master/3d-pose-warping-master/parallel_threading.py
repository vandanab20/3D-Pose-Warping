import itertools
import queue
import threading
import multiprocessing
import ctypes
import signal
import os
import random
import time
import numpy as np
import tensorflow as tf


def identity(x):
    return x


def parallel_map_as_tf_dataset(
        fun, iterable, *, output_types=None, output_shapes=None, shuffle_before_each_epoch=False,
        extra_args=None, n_workers=10, n_epochs=None, deterministic=False):
    
    if deterministic:
        n_workers = 1
        shuffle_before_each_epoch = False

    if fun is None:
        fun = identity

    extra_args = extra_args or []

    pool = get_pool(n_workers, deterministic)
    semaphore = threading.Semaphore(32)
    q = queue.Queue()

    
    first_elem, iterable = peek(iterable)
    if output_types is None or output_shapes is None:
        sample_output = fun(first_elem, *extra_args)
        output_shapes, output_types = get_shapes_and_tf_dtypes(sample_output)

    if n_epochs is None:
        epoch_counter = itertools.count()
    else:
        epoch_counter = range(n_epochs)

    if shuffle_before_each_epoch:
        iterable = list(iterable)

    def producer():
        for _ in epoch_counter:
            if shuffle_before_each_epoch:
                random.shuffle(iterable)

            for item in iterable:
                semaphore.acquire()
                pool.apply_async(fun, (item, *extra_args), callback=q.put)
        q.put(None)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    def consumer():
        while True:
            result = q.get()
            if result is None:
                return
            else:
                semaphore.release()
                yield tuple(result)

    return tf.data.Dataset.from_generator(consumer, output_types, output_shapes)


def peek(iterable):
    iterator = iter(iterable)
    head = next(iterator)
    return head, itertools.chain([head], iterator)


def get_shapes_and_tf_dtypes(thing):
    if not isinstance(thing, (list, tuple)):
        thing = (thing,)

    arrays = [np.asanyarray(a) for a in thing]
    tf_types = [tf.as_dtype(a.dtype) for a in arrays]
    shapes = [tf.TensorShape(a.shape) for a in arrays]
    return tuple(shapes), tuple(tf_types)


_pool = None


def get_pool(n_workers_if_uninitialized, deterministic):
    global _pool
    if deterministic:
        _pool = None
    if _pool is None:
        ctx = multiprocessing.get_context('spawn')
        # important to use 'spawn', because 'fork' would mean the whole memory is (lazily) copied
        # then due to copy-on-write semantics, it gets duplicated when the parent changes anything
        if deterministic:
            _pool = ctx.Pool(n_workers_if_uninitialized, initializer=init_worker_process_det)
        else:
            _pool = ctx.Pool(n_workers_if_uninitialized, initializer=init_worker_process)

    return _pool


def init_worker_process():
    terminate_on_parent_death()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    seed = generate_seed()
    np.random.seed(seed)
    random.seed(seed)


def init_worker_process_det():
    terminate_on_parent_death()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    seed = 0
    np.random.seed(seed)
    random.seed(seed)


def generate_seed():
    pid = os.getpid()
    s = int(time.time())
    return abs(((s * 181) * ((pid - 83) * 359)) % 104729)


def terminate_on_parent_death():
    prctl = ctypes.CDLL("libc.so.6").prctl
    PR_SET_PDEATHSIG = 1
    result = prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    if result != 0:
        print('prctl failed with exit code', result)
