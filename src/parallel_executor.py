# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import queue
import time
import logging
from typing import Callable, List, Dict, Any, Optional, Union
from threading import Thread
from src.config_manager import config

# Get timeout values from ConfigManager
QUEUE_TIMEOUT = config.get('QUEUE_TIMEOUT')

class TaskQueue(queue.Queue):
    """
    Thread-safe task queue for parallel execution.
    """

    def __init__(self, num_workers=1):
        super().__init__()
        self.num_workers = num_workers
        self.start_workers()

    def add_task(self, task, *args, **kwargs):
        self.put((task, args, kwargs))

    def start_workers(self):
        for _ in range(self.num_workers):
            t = Thread(target=self.worker)
            t.daemon = True
            t.start()

    def join_nowait(self):
        """
        Non-blocking variant of join that just checks if the queue is empty.
        This is used with a timeout mechanism for the periodic check approach.
        """
        # No blocking operation here, caller will check unfinished_tasks
        pass

    def worker(self):
        while True:
            item, args, kwargs = self.get()
            task_name = item.__name__ if hasattr(item, '__name__') else 'unknown'
            
            try:
                # Apply task-level timeout if configured
                TASK_TIMEOUT = config.get('TASK_TIMEOUT')
                if TASK_TIMEOUT:
                    # TODO: For now, skip timeout handling to avoid circular import
                    # The timeout should be handled at a higher level
                    try:
                        item(*args, **kwargs)
                    except Exception as e:
                        # Log other exceptions but don't re-raise them
                        logging.error(f"Error in task {task_name}: {str(e)}")
                        continue
                else:
                    # No timeout configured
                    try:
                        item(*args, **kwargs)
                    except Exception as e:
                        # Log other exceptions but don't re-raise them
                        logging.error(f"Error in task {task_name}: {str(e)}")
                        continue
                    
            except Exception as e:
                logging.error(f"Unexpected error in worker thread: {str(e)}")
            finally:
                self.task_done()


class ParallelExecutor:
    """
    Utility class to handle the common pattern of parallel execution used in dataset_processor.
    Handles task queuing, timeout management, result collection, and status reporting.
    """
    
    def __init__(self, num_workers: int = 1, phase_name: str = "Processing"):
        self.num_workers = num_workers
        self.phase_name = phase_name
    
    def execute_parallel_simple(self, 
                               task_func: Callable,
                               items: List[Any],
                               task_args: List[Any] = None,
                               task_kwargs: Dict[str, Any] = None) -> None:
        """
        Execute tasks in parallel without result collection (like all_prepare, all_agent).
        
        Args:
            task_func: The function to execute for each item (e.g., th_prepare, th_agent)
            items: List of items to process (e.g., list of IDs)
            task_args: Additional positional arguments to pass to task_func
            task_kwargs: Additional keyword arguments to pass to task_func
        """
        print(f"\n=== Starting {self.phase_name} Phase ===")
        
        task_args = task_args or []
        task_kwargs = task_kwargs or {}
        
        q = TaskQueue(num_workers=self.num_workers)

        # Add all tasks to the queue
        for item in items:
            q.add_task(task_func, item, *task_args, **task_kwargs)

        # Handle timeout and completion
        self._wait_for_completion(q, len(items))
        
        print(f"=== {self.phase_name} Phase Complete ===\n")
    
    def execute_parallel_with_results(self,
                                    task_func: Callable,
                                    items: List[Any],
                                    task_args: List[Any] = None,
                                    task_kwargs: Dict[str, Any] = None,
                                    failed_items: List[Any] = None,
                                    error_result_factory: Callable = None) -> Dict[str, Any]:
        """
        Execute tasks in parallel with result collection (like all_run).
        
        Args:
            task_func: The function to execute for each item (e.g., th_run)
            items: List of items to process (e.g., list of IDs)
            task_args: Additional positional arguments to pass to task_func
            task_kwargs: Additional keyword arguments to pass to task_func
            failed_items: List of items that failed in previous phase
            error_result_factory: Function to create error results for failed items
            
        Returns:
            Dictionary mapping items to their results
        """
        print(f"\n=== Starting {self.phase_name} Phase ===")
        
        task_args = task_args or []
        task_kwargs = task_kwargs or {}
        failed_items = failed_items or []
        
        # Verify prerequisites if there are failed items
        if failed_items:
            print(f"WARNING: {len(failed_items)} items failed in previous phase")
            for item in failed_items:
                print(f"  - {item}")

        result_queue = queue.Queue()
        task_queue = TaskQueue(num_workers=self.num_workers)

        # Only run tasks that weren't in the failed list
        successful_items = [item for item in items if item not in failed_items]
        
        for item in successful_items:
            # Add result_queue as an argument for th_ functions that need it
            task_queue.add_task(task_func, item, result_queue, *task_args, **task_kwargs)
        
        # Add error results for failed items
        for item in failed_items:
            if error_result_factory:
                error_result = error_result_factory(item)
                result_queue.put({item: error_result})

        # Handle timeout and completion
        self._wait_for_completion(task_queue, len(successful_items))
        
        # Collect results
        results = self._collect_results(result_queue, len(items))
        
        print(f"=== {self.phase_name} Phase Complete ===\n")
        return results
    
    def execute_parallel_with_custom_results(self,
                                           task_func: Callable,
                                           items: List[Any],
                                           result_processor: Callable,
                                           task_args: List[Any] = None,
                                           task_kwargs: Dict[str, Any] = None) -> Any:
        """
        Execute tasks in parallel with custom result processing (like all_refine).
        
        Args:
            task_func: The function to execute for each item
            items: List of items to process  
            result_processor: Function to process results as they come in
            task_args: Additional positional arguments to pass to task_func
            task_kwargs: Additional keyword arguments to pass to task_func
            
        Returns:
            Whatever the result_processor returns
        """
        print(f"\n=== Starting {self.phase_name} Phase ===")
        
        task_args = task_args or []
        task_kwargs = task_kwargs or {}
        
        result_queue = queue.Queue()
        task_queue = TaskQueue(num_workers=self.num_workers)
        
        # Add all tasks
        for item in items:
            task_queue.add_task(task_func, item, result_queue, *task_args, **task_kwargs)
        
        # Process results as they come in
        results = result_processor(result_queue, task_queue, len(items))
        
        print(f"=== {self.phase_name} Phase Complete ===\n")
        return results
    
    def _wait_for_completion(self, task_queue: TaskQueue, expected_tasks: int) -> None:
        """
        Wait for all tasks to complete, handling timeouts appropriately.
        """
        # If QUEUE_TIMEOUT is None, use traditional blocking join
        if QUEUE_TIMEOUT is None:
            task_queue.join()
        else:
            # Set a timeout for the join operation
            start_time = time.time()
            remaining_tasks = expected_tasks
            
            # Check periodically if the queue is done or if timeout has occurred
            while remaining_tasks > 0:
                # Check if we've reached the timeout
                if time.time() - start_time > QUEUE_TIMEOUT:
                    print(f"Queue join timeout after {QUEUE_TIMEOUT}s. {remaining_tasks} tasks may not have completed.")
                    break
                    
                # Wait for a short period and then check if tasks are done
                task_queue.join_nowait()  # This is a non-blocking version of join
                time.sleep(1)
                
                # Recalculate remaining tasks
                remaining_tasks = task_queue.unfinished_tasks

        # Verify all tasks are complete
        if task_queue.unfinished_tasks > 0:
            print(f"WARNING: {task_queue.unfinished_tasks} {self.phase_name.lower()} tasks did not complete successfully")
        else:
            print(f"All {self.phase_name.lower()} tasks completed")
    
    def _collect_results(self, result_queue: queue.Queue, expected_results: int) -> Dict[str, Any]:
        """
        Collect results from the result queue.
        """
        # Get results from the result queue - handle timeout case
        if QUEUE_TIMEOUT is not None:
            # Check how many results we actually have
            available_results = result_queue.qsize()
            if available_results < expected_results:
                print(f"Warning: Expected {expected_results} results but only got {available_results}")
            
            # Get available results
            results_list = []
            for _ in range(available_results):
                try:
                    results_list.append(result_queue.get(block=False))
                except queue.Empty:
                    break
        else:
            # Get all results from the result queue
            results_list = [result_queue.get() for _ in range(expected_results)]

        # Format to Dictionary
        results = {}
        for result_item in results_list:
            if isinstance(result_item, dict):
                for key, value in result_item.items():
                    results[key] = value
            else:
                # Handle unexpected result format
                logging.warning(f"Unexpected result format: {result_item}")
                
        return results
