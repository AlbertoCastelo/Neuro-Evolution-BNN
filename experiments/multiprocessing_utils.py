import sys
import pdb
from multiprocessing import Process, Queue


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class Worker(Process):
    def __init__(self, task_queue: Queue, exit_queue: Queue, exception_queue: Queue,
                 results_queue: Queue):
        super().__init__()
        self.task_queue = task_queue
        self.exit_queue = exit_queue
        self.exception_queue = exception_queue
        self.results_queue = results_queue

    def run(self) -> None:
        while self.exit_queue.empty():
            # try:
            # print(self.task_queue.empty())
            if not self.task_queue.empty():
                task = self.task_queue.get()
                print(f'Running task: {str(task)}')
                task.run()
                print(f'Results: {task.get_results()}')
                self.results_queue.put(task.get_results())
            # except Exception as e:
            #     print(e)
            #     self.exception_queue.put(e)
        print('Worker dying')
