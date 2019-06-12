'''Simple framework for running dependent jobs.'''
import abc
import collections
import copy
import hashlib
import multiprocessing
import os
import subprocess


def params_to_tuple(params):
    pairs = []
    for k in sorted(params.keys()):
        v = params[k]
        if type(v) is dict:
            s = params_to_tuple(v)
        elif type(v) is list:
            s = ','.join('%s' % vv for vv in v)
        else:
            s = v
        pairs.append((k, s))
    return tuple(pairs)


def params_to_string(params):
    pairs = []
    for k in sorted(params.keys()):
        v = params[k]
        if type(v) is dict:
            s = params_to_string(v)
        else:
            s = '%s' % v
        pairs.append(('%s' % k, s))
    return ';'.join('%s=%s' % (k, v) for k, v in pairs)


def params_to_id(params):
    config_hash = hashlib.md5()
    config_hash.update(params_to_string(params).encode('utf-8'))
    return config_hash.hexdigest()


Job = collections.namedtuple('Job', ['task', 'params'])

# Based on https://stackoverflow.com/a/1151686.
Job.__key__ = lambda self: (self.task,) + params_to_tuple(self.params)
Job.__hash__ = lambda self: hash(self.__key__())
Job.__eq__ = lambda self, other: self.__key__() == other.__key__()


class Task(abc.ABC):

    @abc.abstractmethod
    def Outputs(self, params):
        raise NotImplementedError

    @abc.abstractmethod
    def Run(self, params):
        raise NotImplementedError

    @abc.abstractmethod
    def Dependencies(self, params):
        raise NotImplementedError


registry = {}


def registertask(name):
    def decorator(cls):
        registry[name] = cls()
        return cls
    return decorator


def Outputs(job):
    return registry[job.task].Outputs(job.params)


def Dependencies(job):
    return registry[job.task].Dependencies(job.params)


def Run(queue, job):
    try:
        registry[job.task].Run(job.params)
    except Exception as e:
        print('caught exception running job ', end='')
        print(job, end='')
        print(':')
        print(e)
        queue.put((-1, job))
    else:
        queue.put((0, job))


def RunWorker(queue, job):
    p = multiprocessing.Process(target=Run, args=(queue, job))
    p.start()


def RunJobs(root_jobs):
    print('registry:')
    print(registry)

    # Walk dependencies building a tree.
    leaf_jobs = set()

    def BuildJobTree(job):
        subtrees = [BuildJobTree(d) for d in Dependencies(job)]
        if not subtrees:
            leaf_jobs.add(job)
        return (job, subtrees)

    job_trees = [BuildJobTree(root_job) for root_job in root_jobs]

    # Invert the dependencies of the tree.
    dependants = collections.defaultdict(list)

    def FindDependants(job):
        for dep in Dependencies(job):
            dependants[dep].append(job)
            FindDependants(dep)

    for root_job in root_jobs:
        FindDependants(root_job)

    complete_jobs = set()

    def IsReady(job):
        for dep in Dependencies(job):
            if dep not in complete_jobs:
                return False
        return True

    queue = multiprocessing.Queue()

    max_jobs = max(1, multiprocessing.cpu_count() - 1)

    pending_jobs = set()
    failed = False
    while True:
        while leaf_jobs and len(pending_jobs) < max_jobs and not failed:
            leaf = list(leaf_jobs)[0]
            RunWorker(queue, leaf)
            leaf_jobs.remove(leaf)
            pending_jobs.add(leaf)

        if pending_jobs:
            print('waiting for jobs to complete...')
            status, complete_job = queue.get()
            if status:
                failed = True
                print('job failed:')
                print(complete_job)
            else:
                print('job complete:')
                print(complete_job)
                complete_jobs.add(complete_job)
                for dependant in dependants[complete_job]:
                    if IsReady(dependant):
                        # Newly ready job.
                        leaf_jobs.add(dependant)

            pending_jobs.remove(complete_job)

        if failed and not pending_jobs:
            print('failed to complete some jobs')
            break

        if not failed and not leaf_jobs and not pending_jobs:
            print('successfully completed all jobs')
            break

    print('completed %d jobs' % len(complete_jobs))
