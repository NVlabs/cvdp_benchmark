# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import time
import json
import re
import os
from . import repository
import logging
from .merge_in_memory import diff_apply
from .llm_lib.openai_llm import OpenAI_Instance
from .llm_lib.model_factory import ModelFactory
from threading import Thread, Timer, current_thread
from .create_jsonl import create_jsonl
import queue 
from .model_helpers import ModelHelpers
from .config_manager import config
from .constants import CODE_COMPREHENSION_CATEGORIES, LLM_RETRY_COUNT_DEFAULT
from dotenv import load_dotenv
import shutil
import subprocess
import yaml
from . import network_util
from .dir_monitor import DirectorySizeMonitor
from . import git_utils
import psutil
import sys
# Load environment variables from .env file
load_dotenv()

# ----------------------------------------
# - Global Configurations
# ----------------------------------------

# Get timeout values from ConfigManager
MODEL_TIMEOUT = config.get('MODEL_TIMEOUT')
TASK_TIMEOUT = config.get('TASK_TIMEOUT')
QUEUE_TIMEOUT = config.get('QUEUE_TIMEOUT')

# Register LLM retry count if not already configured
config.register_config("LLM_RETRY_COUNT", default=LLM_RETRY_COUNT_DEFAULT, type_cast=int,
                      description="Number of retries for LLM API calls when parsing fails")
LLM_RETRY_COUNT = config.get('LLM_RETRY_COUNT')

# Register subjective scoring model configuration
config.register_config("SUBJECTIVE_SCORING_MODEL", default="sbj_score", type_cast=str,
                      description="Model to use for subjective scoring evaluation")
config.register_config("SUBJECTIVE_SCORING_THRESHOLD", default=7.0, type_cast=float,
                      description="Threshold for LLM-based subjective scoring (1-10 scale)")

# Print timeout configuration for debugging
if QUEUE_TIMEOUT is not None:
    print(f"Queue timeout set to {QUEUE_TIMEOUT}s")
else:
    print("Queue timeout is disabled by default. Set QUEUE_TIMEOUT environment variable to enable it.")

# ----------------------------------------
# - Timeout Utilities
# ----------------------------------------

class ThreadingTimeout:
    """
    Context manager for timing out function calls using threading only.
    Works in all threads, not just the main thread.
    
    Usage:
        try:
            with ThreadingTimeout(seconds=30):
                # code that might hang
        except TimeoutError:
            # handle timeout
    """
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None
        self.timed_out = False
        self.original_thread = None
        self.exc_info = None
    
    def _timeout_function(self):
        """Function called when timeout occurs."""
        self.timed_out = True
        # Store the original thread to raise exception later
        self.original_thread = current_thread()
        # Set the exception info for later use
        self.exc_info = TimeoutError(f"Function call timed out after {self.seconds} seconds")
    
    def __enter__(self):
        # Start a timer that will call _timeout_function after seconds
        self.timer = Timer(self.seconds, self._timeout_function)
        self.timer.daemon = True
        self.timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cancel the timer if operation completed in time
        if self.timer:
            self.timer.cancel()
        
        # If we timed out, raise the exception
        if self.timed_out and current_thread() == self.original_thread:
            raise self.exc_info
        
        return False

# ----------------------------------------
# - Multi-Threading
# ----------------------------------------



# ----------------------------------------
# - JSONL 2 Repository
# ----------------------------------------

class DatasetProcessor():
    """
    Class to convert JSON to repository.
    """

    # ----------------------------------------
    # - Constructor
    # ----------------------------------------

    def __init__(self, filename : str, golden : bool = True, threads : int = 1, debug = False, host = False, prefix : str = None, network_name = None, manage_network = True):
        # Initialize the model
        self.model   = None
        self.context = {}

        # Update context
        self.golden = golden
        self.debug  = debug
        self.host   = host
        self.prefix = prefix if prefix is not None else config.get("BENCHMARK_PREFIX")
        self.disable_patch = False  # Default: patches are applied in golden mode

        # Network settings
        self.network_name = network_name
        self.manage_network = manage_network

        # Centralized subjective scoring model management
        self._model_factory = ModelFactory()
        self._sbj_model_cache = {}  # Cache for subjective models by configuration
        self._sbj_model_config = None  # Current subjective model configuration
        self._sbj_model_instance = None  # Current subjective model instance
        
        # Initialize subjective scoring configuration
        self._init_subjective_scoring()

        self.folders = """You are a helpful assistance.
Consider that you have a folder structure like the following:\n
    - rtl/*   : Contains files which are RTL code.
    - verif/* : Contains files which are used to verify the correctness of the RTL code.
    - docs/*  : Contains files used to document the project, like Block Guides, RTL Plans and Verification Plans.

When generating files, return the file name in the correct place at the folder structure.
"""
            
        self.schema = [
            '{ "code": [{ "<name>" : "<code>"}] }',
            '{ "response": "<response>" }'
        ]

        self.files    = {}
        self.runs     = {}
        self.threads  = threads
        self.filename = filename
        
        # Initialize the model helpers
        self.helpers = ModelHelpers(self.folders, self.schema)

    # ----------------------------------------
    # - Helper Functions for Model Interaction
    # ----------------------------------------
    
    def create_system_prompt(self, base_context=None, schema=None):
        """
        Create a system prompt for the model.
        
        Args:
            base_context: Base context to use (defaults to self.folders)
            schema: Optional JSON schema for structured output
            
        Returns:
            The formatted system prompt
        """
        return self.helpers.create_system_prompt(base_context, schema)
    
    def determine_schema(self, files):
        """
        Determine schema based on the number of expected files.
        
        Args:
            files: List of expected output files
            
        Returns:
            Tuple of (schema, no_schema flag)
        """
        return self.helpers.determine_schema(files)
        
    def parse_model_response(self, res, files=None, no_schema=False):
        """
        Parse the model's response based on schema and expected files.
        
        Args:
            res: Raw response from the model
            files: List of expected output files
            no_schema: Whether schema was used
            
        Returns:
            Parsed output as a dictionary and success flag
        """
        return self.helpers.parse_model_response(res, files, no_schema)

    # ----------------------------------------
    # - Centralized Subjective Scoring Model Management
    # ----------------------------------------
    
    def _init_subjective_scoring(self):
        """
        Initialize subjective scoring configuration from environment/config.
        This method can be overridden by subclasses to customize behavior.
        """
        self._sbj_model_config = self._get_subjective_model_config()
        self._sbj_scoring_threshold = config.get('SUBJECTIVE_SCORING_THRESHOLD')
        
    def _get_subjective_model_config(self):
        """
        Get the subjective model configuration.
        This method can be overridden by subclasses to provide custom model selection logic.
        
        Returns:
            str: Model configuration string (e.g., "sbj_score_gpt-o4-mini")
        """
        return config.get('SUBJECTIVE_SCORING_MODEL')
    
    def _create_subjective_model(self, model_config=None):
        """
        Create a subjective scoring model instance.
        This method handles model creation with error handling and fallback.
        
        Args:
            model_config: Optional model configuration override
            
        Returns:
            The created model instance or None if creation failed
        """
        if model_config is None:
            model_config = self._sbj_model_config
            
        if not model_config:
            logging.warning("No subjective scoring model configured")
            return None
            
        try:
            model_instance = self._model_factory.create_model(
                model_name=model_config,
                context=None  # Subjective scoring models don't need context
            )
            
            # Set debug mode if enabled
            if hasattr(model_instance, 'set_debug') and self.debug:
                model_instance.set_debug(True)
                
            logging.info(f"Created subjective scoring model: {model_config}")
            return model_instance
            
        except Exception as e:
            logging.error(f"Failed to create subjective scoring model '{model_config}': {str(e)}")
            return None
    
    def get_subjective_model(self, model_config=None):
        """
        Get or create a subjective scoring model instance with caching.
        This method implements the template method pattern for model access.
        
        Args:
            model_config: Optional model configuration override
            
        Returns:
            The model instance or None if unavailable
        """
        if model_config is None:
            model_config = self._sbj_model_config
            
        if not model_config:
            return None
            
        # Check cache first
        if model_config in self._sbj_model_cache:
            return self._sbj_model_cache[model_config]
            
        # Create new model instance
        model_instance = self._create_subjective_model(model_config)
        
        if model_instance:
            # Cache the model for reuse
            self._sbj_model_cache[model_config] = model_instance
            
            # Update current instance reference
            if model_config == self._sbj_model_config:
                self._sbj_model_instance = model_instance
                
        return model_instance
    
    def _clear_subjective_model_cache(self):
        """
        Clear the subjective model cache.
        Useful for cleanup or when model configuration changes.
        """
        self._sbj_model_cache.clear()
        self._sbj_model_instance = None
        
    def configure_subjective_scoring(self, model_config=None, threshold=None):
        """
        Configure subjective scoring parameters.
        This method allows runtime configuration of subjective scoring.
        
        Args:
            model_config: Optional new model configuration
            threshold: Optional new scoring threshold
        """
        if model_config is not None:
            self._sbj_model_config = model_config
            # Clear cache to force recreation with new config
            self._clear_subjective_model_cache()
            
        if threshold is not None:
            self._sbj_scoring_threshold = threshold
            
        logging.info(f"Subjective scoring configured: model={self._sbj_model_config}, threshold={self._sbj_scoring_threshold}")
    
    def set_model_factory(self, factory):
        """
        Set a custom model factory for the DatasetProcessor instance.
        This allows wrapper classes to provide their own factory with custom models.
        
        Args:
            factory: ModelFactory instance to use for model creation
        """
        if factory is not None:
            self._model_factory = factory
            # Clear cache to force recreation with new factory
            self._clear_subjective_model_cache()
            logging.info("Custom model factory set, subjective model cache cleared")
    
    @property
    def sbj_llm_model(self):
        """
        Property to provide backward compatibility and lazy model creation.
        This replaces the old pattern of accessing via getattr.
        
        Returns:
            The current subjective model instance
        """
        if self._sbj_model_instance is None:
            self._sbj_model_instance = self.get_subjective_model()
        return self._sbj_model_instance

    # ----------------------------------------
    # - Create Model
    # ----------------------------------------

    def create_model(self):
        self.model  = OpenAI_Instance(self.folders)

    # ----------------------------------------
    # - Process JSON File
    # ----------------------------------------

    def process_json (self, filename : str = None):

        if filename is None:
            filename = self.filename

        with open(filename, 'r') as file:
            content = file.readlines()

        contexts = []

        # Process each line
        for line in content:
            contexts.append(json.loads(line))
        
        self.context = {}

        # - Update Context Per ID
        for context in contexts:
            self.context[ context['id'] ] = context

    # ----------------------------------------
    # - Restore File after Patch
    # ----------------------------------------
    
    def apply_patch(self, diffs, initial_context : str = ''):

        return diff_apply(initial_context, diffs)

    # ----------------------------------------
    # - Create Context based on ID
    # ----------------------------------------

    def create_context(self, id: str, model=None):
        """
        Abstract method for creating context. Must be implemented by subclasses.
        """
        raise NotImplementedError("create_context must be implemented by subclasses")

    # ----------------------------------------
    # - Prepare Repository
    # ----------------------------------------

    def get_patch_keys(self, id):
        """
        Get the patch keys for a given datapoint ID.
        This method can be overridden by subclasses to access patches from different locations.
        """
        return self.context[id]['output']['context'].keys()

    def extract_datapoint (self, id):

        try:
            harness     = self.context[id]['harness']
            data_point  = id.split("_")
            name        = os.path.join(self.prefix, "cvdp_" + "_".join(data_point[1:-1]))
            issue       = int(data_point[-1])
            patches     = self.get_patch_keys(id)

        except:
            print(f"{id} not available in the provided dataset.")
            raise ValueError(f"{id} not available in the provided dataset.")

        return (harness, name, issue, patches)

    def create_repository (self, id : str, harness : {} = None, name : str = "", issue : str = "", patches : {} = None):
        # Determine if this specific datapoint requires EDA license network
        from src import commercial_eda
        datapoint = self.context.get(id, {})
        requires_eda_license = commercial_eda.datapoint_requires_eda_license(datapoint)
        
        repo = repository.Repository(name, issue, self.files [id], harness['files'] if harness and 'files' in harness else harness, patches, host=self.host, sbj_llm_model=self.sbj_llm_model, network_name=getattr(self, 'network_name', None), manage_network=getattr(self, 'manage_network', True), requires_eda_license=requires_eda_license)
        
        # Network configuration is now passed during construction, no need to set it after
        
        return (harness != None and harness != {}, repo)

    def get_context_for_repo(self, id, model):
        """
        Abstract method for getting context for repository creation. Must be implemented by subclasses.
        
        Args:
            id: The datapoint ID
            model: The model instance
            
        Returns:
            The context dictionary for the repository
        """
        raise NotImplementedError("get_context_for_repo must be implemented by subclasses")

    def create_repo (self, id : str, model : OpenAI_Instance = None):

        # Extract Context Information
        (harness, name, issue, patches) = self.extract_datapoint(id)

        if not id in self.files:
            print(f"Creating harness environment for datapoint: {name}_{str(issue).zfill(4)}")
            self.files [id] = self.get_context_for_repo(id, model)

        return self.create_repository(id, harness, name, issue, patches)

    # ----------------------------------------
    # - Repository From Context Dictionary
    # ----------------------------------------

    def set_repo (self, id : str, context = {}):

        # Extract Context Information
        (harness, name, issue, _) = self.extract_datapoint(id)

        if not id in self.files:
            print(f"Recreating harness environment for datapoint: {name}_{str(issue).zfill(4)}")
            self.files [id] = context ['input']

            # Replicate Results
            for file, content in context ['output'].items():
                self.files [file] = content

        return self.create_repository(id, harness, name, issue)

    # ----------------------------------------
    # - Harness Execution per ID
    # ----------------------------------------

    def get_id(self, issue):
        if len(issue.split("_")) == 1:
            assert False, f"Issue {issue} is not a valid ID"
            id = issue.zfill(4)
            return self.name + f"_{id}"
        else:
            return issue

    # ----------------------------------------
    # - Initial Context function
    # ----------------------------------------

    def get_context_result(self, context):
        """
        Get the context result from a datapoint context.
        This method can be overridden by subclasses to access context from different locations.
        """
        return context['input']['context']

    def initial_context (self, id : str):

        context = copy.deepcopy(self.context [ id ])
        result  = self.get_context_result(context)
        issue   = int(id.split("_")[-1])

        return result, issue

    # ----------------------------------------
    # - Split into preparation and Execution
    # ----------------------------------------

    def prepare(self, issue : str = "", model : OpenAI_Instance = None):

        # Create Context
        id  = self.get_id(issue)
        (obj, repo) = self.create_repo(id, model)

        return (id, obj, repo)

    # ----------------------------------------
    # - Shared Subjective Scoring
    # ----------------------------------------

    def run_subjective_scoring(self, id, repo, model=None, obj=False):
        """
        Run subjective (ROUGE/BLEU) scoring on text content.
        This function can be used by both Copilot and Agentic formats.
        
        Args:
            id: The datapoint ID
            repo: Repository instance
            model: Optional model for LLM evaluation
            obj: Whether to also run objective testing
            
        Returns:
            Tuple of (tests, errors)
        """
        # Get the expected answer/reference
        cat = int(self.context[id]['categories'][0][3:])
        
        # Get reference from output.response for both Copilot and Agentic formats
        reference = None
        if 'subjective_reference' in self.context[id]:
            reference = self.context[id]['subjective_reference']
        elif 'output' in self.context[id] and 'response' in self.context[id]['output']:
            reference = self.context[id]['output']['response']
        else:
            reference = ""
            logging.error(f"ERROR: No reference response found for {id}. Subjective scoring will be inaccurate and likely fail!")
        
        # Extract the original problem prompt
        problem_prompt = ""
        if 'prompt' in self.context[id]:
            problem_prompt = self.context[id]['prompt']
        elif 'input' in self.context[id] and 'prompt' in self.context[id]['input']:
            problem_prompt = self.context[id]['input']['prompt']
        elif 'input' in self.context[id] and 'text' in self.context[id]['input']:
            problem_prompt = self.context[id]['input']['text']
        
        # Find the actual content to score
        uut = None
        if 'subjective.txt' in self.files[id]:
            # Standard location for LLM response
            uut = self.files[id]['subjective.txt']
        elif 'docs/subjective.txt' in self.files[id]:
            # Agentic format location (in docs directory)
            uut = self.files[id]['docs/subjective.txt']
        else:
            # No suitable content found and not in golden mode
            print(f"[WARNING] No content found for subjective scoring of {id}. Skipping evaluation...")
            tests = [{"result": 1, "log": None, "error_msg": "No content for subjective scoring", "execution": 0.0}]
            return tests, 1
        
        # Ensure we have content to score
        if not uut or len(uut) == 0:
            print(f"[WARNING] Empty content for {id}. Skipping subjective evaluation...")
            tests = [{"result": 1, "log": None, "error_msg": "Empty content for subjective scoring", "execution": 0.0}]
            return tests, 1
        
        # Run the subjective tests with problem prompt
        (tests, errors) = repo.sbj(uut, reference, cat, problem_prompt)
        print(f"Finished executing subjective scoring for {id}...")
        
        # Optionally run objective tests if requested
        if obj:
            (obj_tests, obj_errors) = repo.obj(uut)
            tests.extend(obj_tests)
            errors += obj_errors
            
        return tests, errors
        
    def th_refine(self, id, q : queue.Queue = None, refine_model = None):
        """
        Thread function to refine a single datapoint.
        
        Args:
            id (str): The datapoint ID
            q (queue.Queue, optional): Queue to put results into
            refine_model: The refinement model instance to use (already created)
        """
        print(f"Refining datapoint {id}...")
        
        try:
            if not refine_model:
                print(f"Warning: No refinement model provided for {id}, skipping")
                if q:
                    q.put({"id": id, "refined": False, "error": "No refinement model provided"})
                return
                
            # Skip if we've already refined this datapoint
            if id in self.refined_datapoints:
                print(f"Datapoint {id} already refined, skipping")
                if q:
                    q.put({"id": id, "refined": True, "already_done": True})
                return
                
            # Get a repository instance to get harness info if needed
            repo = None
            if self.include_harness and id in self.runs and 'repo' in self.runs[id]:
                repo = self.runs[id]['repo']
                
            # Perform the actual refinement
            result = self._try_refine_datapoint(id, refine_model, repo)
            
            if q:
                q.put({"id": id, "refined": result})
                
        except Exception as e:
            print(f"Error in refinement thread for {id}: {str(e)}")
            if q:
                q.put({"id": id, "refined": False, "error": str(e)})
    
    def all_refine(self, model_factory = None):
        """
        Refine all datapoints using the specified model factory.
        This should be called before all_prepare() if refinement is needed.
        
        Args:
            model_factory: The model factory to use for creating refinement models
            
        Returns:
            dict: Results of the refinement process
        """
        if not self.refine_model:
            print("No refinement model specified, skipping all refinements")
            return {"refined": 0, "total": 0, "errors": 0}
        
        if not model_factory:
            print("Warning: No model factory provided for refinement, skipping")
            return {"refined": 0, "total": 0, "errors": 1, "error": "No model factory provided"}
            
        print(f"Starting refinement of all datapoints using model: {self.refine_model}")
        sys.stdout.flush()
        
        if not hasattr(self, 'context') or not self.context:
            print("No datapoints loaded, run process_json() first")
            return {"refined": 0, "total": 0, "errors": 1, "error": "No datapoints loaded"}
            
        # Create a single refinement model for all threads to use
        try:
            print(f"Creating refinement model: {self.refine_model}")
            refine_model = model_factory.create_model(model_name=self.refine_model, context=self.folders)
            
            if not hasattr(refine_model, 'refine') or not callable(refine_model.refine):
                print(f"Error: Model '{self.refine_model}' does not support the refine() method")
                return {"refined": 0, "total": 0, "errors": 1, "error": f"Model {self.refine_model} does not support refine()"}
        except Exception as e:
            print(f"Error creating refinement model: {str(e)}")
            return {"refined": 0, "total": 0, "errors": 1, "error": str(e)}
            
        # Track which datapoints need refinement
        to_refine = []
        for id in self.context.keys():
            if id not in self.refined_datapoints:
                to_refine.append(id)
                
        print(f"Found {len(to_refine)} datapoints to refine out of {len(self.context)} total")
        
        def process_refine_results(result_queue, task_queue, expected_count):
            """Process refinement results as they come in"""
            refined_count = 0
            error_count = 0
            
            # Wait for all tasks to be added to the queue first
            while task_queue.unfinished_tasks > 0:
                try:
                    # Block with timeout to avoid CPU spinning
                    res = result_queue.get(block=True, timeout=0.1)
                    
                    # Process result
                    id = res.get("id", "unknown")
                    refined = res.get("refined", False)
                    error = res.get("error", None)
                    
                    if refined:
                        refined_count += 1
                    elif error:
                        error_count += 1
                        print(f"Error refining datapoint {id}: {error}")
                        
                    # Mark this queue item as processed
                    result_queue.task_done()
                    
                except queue.Empty:
                    # Timeout occurred, just continue and try again
                    continue

            # Save all refined datapoints at once
            if self.refined_datapoints:
                self._save_refined_datapoints()
                
            print(f"Refinement complete: {refined_count} datapoints refined, {error_count} errors")
            
            return {"refined": refined_count, "total": len(to_refine), "errors": error_count}
        
        from .parallel_executor import ParallelExecutor
        executor = ParallelExecutor(num_workers=self.threads, phase_name="Refinement")
        results = executor.execute_parallel_with_custom_results(
            task_func=self.th_refine,
            items=to_refine,
            result_processor=process_refine_results,
            task_args=[refine_model]
        )
        
        return results
        
    def _try_refine_datapoint(self, id, model, repo):
        """
        Try to refine a datapoint using the model's refine method if available.
        
        Args:
            id (str): The datapoint ID
            model: The model instance
            repo: The repository instance
            
        Returns:
            bool: True if refinement was successful, False otherwise
        """
        # Skip if we've already refined this datapoint
        if id in self.refined_datapoints:
            return True
            
        try:
            # Create a deep copy of the original datapoint to preserve all fields
            original_datapoint = copy.deepcopy(self.context[id])
            
            # Store the original fields which must remain unchanged
            preserved_fields = {
                'output': original_datapoint.get('output', {}),
                'harness': original_datapoint.get('harness', {}),
                'input': original_datapoint.get('input', {}),
                'categories': original_datapoint.get('categories', [])
            }
            
            # Create the refinement context
            refinement_context = {
                'datapoint': original_datapoint,
                'id': id
            }
            
            # Add golden patch information if allowed
            if self.include_golden_patch and not self.golden:
                # Identify if there's a golden patch available for comparison
                golden_version = self._find_golden_version(id)
                if golden_version:
                    refinement_context['golden_patch'] = golden_version
            
            # Add harness information if allowed
            if self.include_harness and repo and repo.issue_path:
                harness_info = self._collect_harness_info(repo.issue_path)
                if harness_info:
                    refinement_context['harness_info'] = harness_info
            
            # Call the model's refine method
            print(f"Refining datapoint {id} using model's refine method...")
            refined_datapoint = model.refine(refinement_context)
            
            # Ensure all preserved fields remain unchanged
            for field, value in preserved_fields.items():
                if field in refined_datapoint:
                    refined_datapoint[field] = value
                
            # Validate the refined datapoint
            if self._validate_refined_datapoint(refined_datapoint, id):
                refined_datapoint['original_prompt'] = original_datapoint['input']['prompt']
                refined_datapoint['input']['prompt'] = refined_datapoint['prompt']
                del refined_datapoint['prompt']
                # Store the refined datapoint
                self.refined_datapoints[id] = refined_datapoint
                print(f"Successfully refined datapoint {id}")
                
                # Update the context with the refined datapoint
                self.context[id] = refined_datapoint
                
                return True
            else:
                print(f"Refined datapoint for {id} has invalid format, using original")
                return False
        except Exception as e:
            print(f"Error refining datapoint {id}: {str(e)}")
            return False

    def th_prepare(self, id, model):

        print(f"Starting {id} repository execution...")

        try:

            # DeepCopy Initial Context
            (input, _) = self.initial_context(id)

            (_, obj, repo) = self.prepare(issue = id, model = model)

            output = self.files.get(id, {})
            diff   = dict(set(output.items()) ^ set(input.items()))

            self.runs [id] = {
                'obj' : obj,
                'repo' : repo,
                'input' : input,
                'output' : diff
            }

        except Exception as e:
            # Log the error
            error_msg = str(e)
            logging.error(f"Error in repository preparation for {id}: {error_msg}")
            
            # Mark this ID as having an error so we can skip it in the run phase
            self.runs[id] = {
                'obj': False,
                'repo': None,
                'input': {},
                'output': {},
                'error_msg': "th_prepare: " + error_msg
            }
            
            # If this is not our expected error type, log it for debugging
            if error_msg != f"Unable to process harness for id {id}":
                print(f"Unexpected error in preparation for {id}: {error_msg}")

    def th_run(self, id, q : queue.Queue = None, model : OpenAI_Instance = None):
        """
        Override th_run to check for agent errors.
        """
        print(f"Starting {id} harness execution...")

        try:
            # Check if this task had an error during preparation
            if id in self.runs and 'error_msg' in self.runs[id]:
                error_msg = self.runs[id]['error_msg']
                logging.warning(f"Skipping {id} due to preparation error: {error_msg}")
                
                # Use the real category and difficulty from the context
                category = self.context[id]['categories'][0]
                difficulty = self.context[id]['categories'][1]
                
                # Return an error result with the actual category and difficulty
                error_result = {
                    "category": category,
                    "difficulty": difficulty,
                    "tests": [{"result": 1, "log": None, "error_msg": "th_run: " + error_msg, "execution": 0.0}],
                    "errors": 1
                }
                q.put({id: error_result})
                return
            
            # Check if there was an agent error
            if id in self.runs and 'agent_error' in self.runs[id]:
                error_msg = self.runs[id]['agent_error']
                logging.warning(f"Found agent error for {id}: {error_msg}")
                # We still continue with execution as we may have partial results
            
            # Run the normal execution
            res = self.run(
                id = id,
                obj = self.runs[id]['obj'],
                repo = self.runs[id]['repo'],
                model = model
            )
            
            # If there was an agent error, add it to the result
            if id in self.runs and 'agent_error' in self.runs[id]:
                if 'tests' in res:
                    # Add agent error to test results if not already present
                    agent_error_found = False
                    for test in res['tests']:
                        if 'error_msg' in test and test['error_msg'] and 'agent_error' in test['error_msg']:
                            agent_error_found = True
                            break
                    
                    if not agent_error_found:
                        # Include agent logfile if available
                        agent_logfile = self.runs[id].get('agent_logfile', None)
                        res['tests'].append({
                            "result": 1,
                            "log": agent_logfile, 
                            "error_msg": f"Agent error: {self.runs[id]['agent_error']}",
                            "execution": 0.0
                        })
                        res['errors'] += 1
            
            # Add agent logfile to result metadata if available
            if id in self.runs and 'agent_logfile' in self.runs[id]:
                res['agent_logfile'] = self.runs[id]['agent_logfile']

            q.put({id: res})

        except Exception as e:
            # Log the error
            error_msg = str(e)
            logging.error(f"Error in harness execution for {id}: {error_msg}")
            
            # Use the real category and difficulty from the context
            category = self.context[id]['categories'][0]
            difficulty = self.context[id]['categories'][1]
            
            # Always put something in the queue so the task is marked as complete
            error_result = {
                "category": category,
                "difficulty": difficulty,
                "tests": [{"result": 1, "log": None, "error_msg": "th_run: " + error_msg, "execution": 0.0}],
                "errors": 1
            }
            
            # Add agent error if there was one
            if id in self.runs and 'agent_error' in self.runs[id]:
                # Include agent logfile if available
                agent_logfile = self.runs[id].get('agent_logfile', None)
                error_result['tests'].append({
                    "result": 1,
                    "log": agent_logfile,
                    "error_msg": f"Agent error: {self.runs[id]['agent_error']}",
                    "execution": 0.0
                })
                error_result['errors'] += 1
            
            # Add agent logfile to result metadata if available
            if id in self.runs and 'agent_logfile' in self.runs[id]:
                error_result['agent_logfile'] = self.runs[id]['agent_logfile']
            
            # Put the error result in the queue so the main thread knows this task is done
            q.put({id: error_result})
            
            # If this is not our expected error type, we can still raise it for debugging
            if error_msg != f"Unable to process harness for id {id}":
                print(f"Unexpected error in task {id}: {error_msg}")

    def all_prepare(self, model : OpenAI_Instance = None):
        from .parallel_executor import ParallelExecutor
        
        executor = ParallelExecutor(num_workers=self.threads, phase_name="Preparation")
        executor.execute_parallel_simple(
            task_func=self.th_prepare,
            items=list(self.context.keys()),
            task_args=[model]
        )

        print("Writing preparation results...")
        content = {key: value for key, value in self.runs.items()}
        jsonl   = []

        for key, value in content.items():
            # Format Dictionary
            dct = {}
            dct['input']  = value ['input']
            dct['output'] = value ['output']
            dct['obj']    = value ['obj']

            jsonl.append({key : dct})

        # Create prefix directory if it doesn't exist
        os.makedirs(self.prefix, exist_ok=True)
        
        # Write Prepared JSONL in prefix directory
        create_jsonl(os.path.join(self.prefix, 'prompt_response.jsonl'), jsonl)

    def run(self, id : str = "", obj : bool = False, repo : repository.Repository = None, model : OpenAI_Instance = None):

        cat = int(self.context[id]['categories'][0][3:])

        # Check if this is a subjective category
        if cat in CODE_COMPREHENSION_CATEGORIES:
            # Run subjective scoring for any mode - the method handles golden vs non-golden internally
            (tests, errors) = self.run_subjective_scoring(id, repo, model, obj)
        else:
            # Objective category - run objective tests
            repo.debug = self.debug
            (tests, errors) = repo.obj()

        result = {}
        result['category']   = self.context[id]['categories'][0]
        result['difficulty'] = self.context[id]['categories'][1]
        result['tests']      = tests
        result['errors']     = errors

        return result


    def all_run(self, model : OpenAI_Instance = None):
        from .parallel_executor import ParallelExecutor
        
        # Verify preparation is complete
        if not self.runs:
            raise RuntimeError("Cannot start execution phase: Preparation phase has not been completed")
            
        # Check if any tasks failed during preparation
        failed_prep = [id for id, run in self.runs.items() if 'error_msg' in run]
        
        def create_error_result(id):
            """Create error result for failed preparation tasks"""
            category = self.context[id]['categories'][0]
            difficulty = self.context[id]['categories'][1]
            return {
                "category": category,
                "difficulty": difficulty,
                "tests": [{"result": 1, "log": None, "error_msg": self.runs[id]['error_msg'], "execution": 0.0}],
                "errors": 1
            }
        
        executor = ParallelExecutor(num_workers=self.threads, phase_name="Execution")
        result = executor.execute_parallel_with_results(
            task_func=self.th_run,
            items=list(self.context.keys()),
            task_args=[model],  # Note: result_queue will be inserted as second arg by executor
            failed_items=failed_prep,
            error_result_factory=create_error_result
        )
        
        return result

class CopilotProcessor (DatasetProcessor):
    def __init__(self, filename : str = "", golden : bool = True, threads : int = 1, debug = False, host = False, prefix : str = None, network_name=None, manage_network=True, include_golden_patch=False, include_harness=False, refine_model=None):
        super().__init__(filename, golden, threads, debug, host, prefix, network_name, manage_network)
        self.include_golden_patch = include_golden_patch
        self.include_harness = include_harness
        self.refined_datapoints = {}
        self.refined_filename = None
        self.refine_model = refine_model

    def get_context_for_repo(self, id, model):
        """
        Get the context for creating a repository using standard context creation.
        
        Args:
            id: The datapoint ID
            model: The model instance
            
        Returns:
            The context dictionary for the repository
        """
        return self.create_context(id, model)
        
    def create_context (self, id : str, model = None):

        context = copy.deepcopy(self.context [ id ])
        result  = self.get_context_result(context)
        issue   = int(id.split("_")[-1])

        # Identify repository naming
        issue_dir = context['id'].split("_")
        issue_dir = os.path.join(self.prefix, "cvdp_" + "_".join(issue_dir[1:-1]))

        if self.golden:

            if not self.disable_patch:
                # ----------------------------------------
                # - Patch Files
                # ----------------------------------------

                if len(context['output']['context']) and not self.disable_patch:

                    # Process Patches
                    for file in context ['output']['context'].keys():

                        patches = context ['output']['context'][ file ]
                        result [file] = patches
                                        
                # ----------------------------------------
                # - Add subjective.txt for response if available in code comprehension categories
                # ----------------------------------------
                
                # Check if this is a subjective category
                cat = int(context['categories'][0][3:])
                if cat in CODE_COMPREHENSION_CATEGORIES:
                    # Add response to subjective.txt for evaluation if it exists
                    if 'response' in context['output'] and context['output']['response']:
                        # Add response to subjective.txt for evaluation
                        result['subjective.txt'] = context['output']['response']
                        
        # ----------------------------------------
        # - Ask for the LLM the response of the prompt input
        # ----------------------------------------

        else:

            prompt = ""

            for file, cxt in context['input']['context'].items():
                prompt += f"\nConsider the following content for the file {file}:\n```\n{cxt}\n```"

            prompt += f"\nProvide me one answer for this request: {context['input']['prompt']}\n"
            files   = list(context['output']['context'].keys())

            # ----------------------------------------
            # - Special handling for code comprehension categories
            # ----------------------------------------
            # For code comprehension categories, we want the response to go into subjective.txt
            # rather than into the patch files listed in output context
            cat = None
            is_code_comprehension = False
            if 'categories' in context and context['categories'] and isinstance(context['categories'][0], str) and context['categories'][0].startswith('cid'):
                cat = int(context['categories'][0][3:])
                is_code_comprehension = cat in CODE_COMPREHENSION_CATEGORIES
            
            if is_code_comprehension:
                # Override files list to empty - this will trigger response mode
                # and the response will be captured in subjective.txt
                files = []

            # Determine schema based on the number of files
            schema_to_use, no_schema = self.determine_schema(files)
            
            # Add additional instructions based on expected files
            if len(files) == 1:
                prompt += f"Please provide your response as plain text without any JSON formatting. Your response will be saved directly to: {files[0]}.\n"
            elif len(files) > 1:
                prompt += f"Name the files as: {files}.\n"
            else:
                # For code comprehension, explicitly request the response format
                if is_code_comprehension:
                    prompt += "Provide your response using the format: { \"response\": \"<your answer here>\" }\n"
                else:
                    prompt += "Provide your response below.\n"

            folders = { 'prompts' : os.path.join(issue_dir, "prompts") }

            # Create all necessary directories first
            try:
                os.makedirs(issue_dir, exist_ok=True)
                os.makedirs(folders['prompts'], exist_ok=True)
            except Exception as e:
                logging.error(f"Failed to create directories for {id}: {str(e)}")
                raise

            llm_retry_count = LLM_RETRY_COUNT
            while 1:
                logfile = os.path.join(issue_dir, "prompts", f"{issue}.md")
                os.makedirs(os.path.dirname(logfile), exist_ok=True)
                print(f"Requesting valid response to model...")

                try:
                    if model != None:
                        cat = None
                        if 'categories' in self.context[id] and self.context[id]['categories'] and isinstance(self.context[id]['categories'][0], str) and self.context[id]['categories'][0].startswith('cid'):
                            cat = int(self.context[id]['categories'][0][3:])
                        output, success = model.prompt(prompt, schema=schema_to_use, prompt_log=logfile, files=files, timeout=MODEL_TIMEOUT, category=cat)
                    elif self.model != None:
                        cat = None
                        if 'categories' in self.context[id] and self.context[id]['categories'] and isinstance(self.context[id]['categories'][0], str) and self.context[id]['categories'][0].startswith('cid'):
                            cat = int(self.context[id]['categories'][0][3:])
                        output, success = self.model.prompt(prompt, schema=schema_to_use, prompt_log=logfile, files=files, timeout=MODEL_TIMEOUT, category=cat)
                    else:
                        raise ValueError("Unable to execute harness without an model assigned.")
                    
                    # Handle parsing failures
                    if not success:
                        if llm_retry_count > 0:
                            print("Exception occurred. Retrying LLM API.")
                            llm_retry_count -= 1
                        else:
                            logging.error("Failed to parse JSON response after multiple retries")
                            output = {}
                            break
                    else:
                        break
                except Exception as e:
                    logging.error(f"Error during model prompt for {id}: {str(e)}")
                    if llm_retry_count > 0:
                        print("Exception occurred. Retrying LLM API.")
                        llm_retry_count -= 1
                    else:
                        raise

            # ----------------------------------------
            # - Increment Context
            # ----------------------------------------

            if len(files) == 1 and 'direct_text' in output:
                # If we're using direct text mode with a single expected file
                result[files[0]] = output['direct_text']
            elif len(files) == 1 and 'response' in output:
                # If we're using the simplified schema with a single expected file
                result[files[0]] = output['response']
            elif 'code' in output:
                # If using the default schema with code array
                for cxt in output['code']:
                    for filename in cxt:
                        result[filename] = cxt[filename]
            else:
                # Fallback - add response into context
                if 'direct_text' in output:
                    result['subjective.txt'] = output['direct_text']
                elif 'response' in output:
                    result['subjective.txt'] = output['response']
                else:
                    result['subjective.txt'] = str(output)

        return result
        
        
    def _find_golden_version(self, id):
        """
        Find the golden version (expected output) for a datapoint.
        
        Args:
            id (str): The datapoint ID
            
        Returns:
            dict: The golden patch information or None if not found
        """
        if 'output' in self.context[id] and 'context' in self.context[id]['output']:
            # Create a patch-like structure from the expected output
            result = {}
            for file_path, content in self.context[id]['output']['context'].items():
                if file_path in self.context[id]['input']['context']:
                    # Create a diff between input and expected output
                    input_content = self.context[id]['input']['context'][file_path]
                    if input_content != content:
                        # Generate a proper diff
                        import difflib
                        diff = '\n'.join(difflib.unified_diff(
                            input_content.splitlines(), 
                            content.splitlines(),
                            fromfile=f'a/{file_path}',
                            tofile=f'b/{file_path}',
                            lineterm=''
                        ))
                        result[file_path] = diff
                else:
                    # New file
                    result[file_path] = content
            return result
        return None
    
    def _collect_harness_info(self, issue_path):
        """
        Collect harness information from the issue path.
        
        Args:
            issue_path (str): The path to the issue directory
            
        Returns:
            dict: The harness information or None if not found
        """
        harness_info = {}
        
        # Check for docker-compose.yml
        docker_compose_yml = os.path.join(issue_path, "docker-compose.yml")
        if os.path.exists(docker_compose_yml):
            try:
                with open(docker_compose_yml, 'r') as f:
                    harness_info['docker_compose'] = f.read()
            except:
                pass
                
        # Check for test files in src directory
        src_dir = os.path.join(issue_path, "src")
        if os.path.exists(src_dir):
            harness_info['test_files'] = {}
            for root, _, files in os.walk(src_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, issue_path)
                    try:
                        with open(file_path, 'r') as f:
                            harness_info['test_files'][rel_path] = f.read()
                    except:
                        pass
        
        return harness_info if harness_info else None
    
    def _validate_refined_datapoint(self, refined_datapoint, original_id):
        """
        Validate that a refined datapoint has the necessary structure.
        
        Args:
            refined_datapoint: The refined datapoint to validate
            original_id: The ID of the original datapoint
            
        Returns:
            bool: True if the refined datapoint is valid, False otherwise
        """
        # Check that the refined datapoint has the necessary structure
        if not isinstance(refined_datapoint, dict):
            print("Refined datapoint is not a valid dictionary")
            return False
            
        # Check for minimal field format (just prompt and scores)
        if 'prompt' in refined_datapoint:
            # This appears to be a minimal response with just the modified fields
            # We'll merge it with the original datapoint later
            print(f"Detected minimal refinement format with just the modified fields")
            
            # Ensure we have the required fields for the minimal format
            if 'reasoning' not in refined_datapoint:
                print("Minimal refinement format missing required 'reasoning' field")
                # Try to continue anyway
            
            # The scores are informational but not required for validation
            if 'ambiguity_score' not in refined_datapoint or 'consistency_score' not in refined_datapoint:
                print("Warning: Refinement missing one or both score fields")
                
            # Create a full datapoint structure by copying the original and updating with the new prompt
            original_datapoint = copy.deepcopy(self.context[original_id])
            
            # Preserve the original 'id', 'input', 'output', 'categories', and 'harness' fields
            # But update with the new prompt and add the new fields
            if 'prompt' in refined_datapoint:
                if 'input' not in original_datapoint:
                    original_datapoint['input'] = {}
                original_datapoint['input']['prompt'] = refined_datapoint['prompt']
            
            if 'reasoning' in refined_datapoint:
                original_datapoint['reasoning'] = refined_datapoint['reasoning']
                
            if 'ambiguity_score' in refined_datapoint:
                original_datapoint['ambiguity_score'] = refined_datapoint['ambiguity_score']
                
            if 'consistency_score' in refined_datapoint:
                original_datapoint['consistency_score'] = refined_datapoint['consistency_score']
                
            # Replace the refined datapoint with our merged version
            for key, value in original_datapoint.items():
                refined_datapoint[key] = value
                
            return True
            
        # If not a minimal response, check for required fields in a full response
        required_fields = ['id', 'input', 'output', 'categories']
        for field in required_fields:
            if field not in refined_datapoint:
                print(f"Refined datapoint missing required field: {field}")
                return False
                
        # Make sure ID matches original
        if refined_datapoint['id'] != self.context[original_id]['id']:
            print("Warning: Refined datapoint has different ID than original")
            # Fix the ID to match original
            refined_datapoint['id'] = self.context[original_id]['id']
            
        # Check input structure
        if not isinstance(refined_datapoint['input'], dict) or 'context' not in refined_datapoint['input']:
            print("Refined datapoint has invalid input structure")
            return False
            
        # Check output structure
        if not isinstance(refined_datapoint['output'], dict):
            print("Refined datapoint has invalid output structure")
            return False
            
        return True
    
    def _save_refined_datapoints(self):
        """
        Save all refined datapoints to a file.
        
        Each refined datapoint will include the original fields plus these refinement fields:
        - prompt: The improved prompt text
        - reasoning: The explanation for the refinements made
        - ambiguity_score: Rating from 1-10 of how ambiguous the original prompt was
        - consistency_score: Rating from 1-10 of how consistent the prompt/input/output/harness were
        """
        if not self.refined_datapoints:
            return
            
        # Create the refined filename if not already done
        if not self.refined_filename:
            # Base the refined filename on the original filename
            basename = os.path.basename(self.filename)
            dirname = os.path.dirname(self.filename)
            self.refined_filename = os.path.join(dirname, f"{os.path.splitext(basename)[0]}_refined.jsonl")
        
        # Write all refined datapoints to the file
        refinement_stats = {
            'total': len(self.refined_datapoints),
            'with_scores': 0,
            'avg_ambiguity': 0,
            'avg_consistency': 0
        }
        
        ambiguity_scores = []
        consistency_scores = []
        
        with open(self.refined_filename, 'w') as f:
            for id, datapoint in self.refined_datapoints.items():
                # Track statistics about scores if they exist
                if 'ambiguity_score' in datapoint and 'consistency_score' in datapoint:
                    refinement_stats['with_scores'] += 1
                    ambiguity_scores.append(datapoint['ambiguity_score'])
                    consistency_scores.append(datapoint['consistency_score'])
                    
                f.write(json.dumps(datapoint) + '\n')
        
        # Calculate average scores if available
        if ambiguity_scores:
            refinement_stats['avg_ambiguity'] = sum(ambiguity_scores) / len(ambiguity_scores)
        if consistency_scores:
            refinement_stats['avg_consistency'] = sum(consistency_scores) / len(consistency_scores)
            
        print(f"Saved {refinement_stats['total']} refined datapoints to {self.refined_filename}")
        
        if refinement_stats['with_scores'] > 0:
            print(f"Refinement statistics:")
            print(f"  - Datapoints with scores: {refinement_stats['with_scores']}/{refinement_stats['total']}")
            print(f"  - Average ambiguity score: {refinement_stats['avg_ambiguity']:.2f}/10")
            print(f"  - Average consistency score: {refinement_stats['avg_consistency']:.2f}/10")

class AgenticProcessor (DatasetProcessor):

    # ----------------------------------------
    # - Process JSON File
    # ----------------------------------------

    def __init__(self, filename : str, golden : bool = True, threads : int = 1, debug = False, host = False, prefix : str = None, network_name=None, manage_network=True):
        super().__init__(filename, golden, threads, debug, host, prefix, network_name, manage_network)
        self.agent_results = {}
        # Directory size monitor
        self.dir_monitor = DirectorySizeMonitor()
        # Initialize include flags to False by default
        self.include_golden_patch = False
        self.include_harness = False

        # Ensure patch_image Docker image exists for agentic heavy processing
        result = subprocess.run(["docker", "images", "-q", "patch_image"],
                                capture_output=True,
                                text=True
        )

        if not result.stdout.strip():

            # Ensure prefix directory exists
            os.makedirs(self.prefix, exist_ok=True)
            dockerfile = os.path.join(self.prefix, "Dockerfile.patch_image")

            print(f"[INFO] Docker image 'patch_image' not found, building it...")
            with open(dockerfile, "w") as f:
                f.write("FROM ubuntu:22.04\nRUN apt update && apt install -y git")

            # Build image
            subprocess.run(["docker", "build", "-t", "patch_image", "-f", dockerfile, "."],
                            check=True)

        else:
            print(f"[INFO] Docker image 'patch_image' already exists...")

    # ----------------------------------------
    # - Process JSON File
    # ----------------------------------------

    def get_patch_keys(self, id):
        """
        Override to access patches from 'patch' key instead of 'output.context'.
        """
        return self.context[id]['patch'].keys()

    def get_context_result(self, context):
        """
        Override to access context directly instead of from 'input.context'.
        """
        return context['context']

    def create_repository (self, id : str, harness = None, name : str = "", issue : str = "", patches = None):
        # Determine if this specific datapoint requires EDA license network
        from src import commercial_eda
        datapoint = self.context.get(id, {})
        requires_eda_license = commercial_eda.datapoint_requires_eda_license(datapoint)

        if 'cvdp_agentic_heavy' in id:
            # For agentic heavy datapoints, create minimal context with just prompt.json
            # This avoids the space overhead of full context restoration while still providing the prompt
            import json
            minimal_context = {
                'prompt.json': json.dumps({"prompt": self.context[id]['prompt']})
            }
            repo = repository.AgenticRepository(name, issue, minimal_context, harness, patches, host=self.host, network_name=getattr(self, 'network_name', None), manage_network=getattr(self, 'manage_network', True), requires_eda_license=requires_eda_license)
        else:
            repo = repository.Repository(name, issue, self.files [id], harness, patches, host=self.host, network_name=getattr(self, 'network_name', None), manage_network=getattr(self, 'manage_network', True), requires_eda_license=requires_eda_license)
        
        # Network configuration is now passed during construction, no need to set it after
        
        return (True, repo)

    def create_context (self, id : str, model : OpenAI_Instance = None):

        # Setup initial context files
        (context, _) = self.initial_context(id)

        # Identify repository naming
        name      = self.context[id]['id'].split("_")
        issue_dir = os.path.join(self.prefix, "cvdp_" + "_".join(name[1:-1]))

        # add prompt.json to the context
        context['prompt.json'] = json.dumps({"prompt" : self.context[id]['prompt']})

        if id in self.agent_results:
            return self.agent_results[id]
        else:
            return self.result_context(int(name [-1]), context, self.context [id]['patch'])

    def agent_run (self, issue_path : str = '', agent : str = '', monitor_size=True, **kargs):
        # Create docker-compose-agent.yml file
        docker_compose_path = os.path.join(issue_path, "docker-compose-agent.yml")

        # If requested, create a golden patch file for the issue
        if hasattr(self, 'include_golden_patch') and self.include_golden_patch:
            # First, find the issue ID from the issue_path
            issue_id = os.path.basename(issue_path)
            
            # Find the corresponding ID in our context data
            data_id = None
            for id in self.context:
                if str(issue_id) in id:
                    data_id = id
                    break
            
            if data_id and 'patch' in self.context[data_id]:
                # Get the patch data
                patch_data = self.context[data_id]['patch']
                
                # Create a unified patch file
                golden_patch_file = os.path.join(issue_path, "golden_ref_solution.patch")
                print(f"Creating golden patch file at {golden_patch_file}")
                
                with open(golden_patch_file, 'w', encoding='utf-8') as f:
                    # Write a header comment
                    f.write("# Golden Patch File - Reference Solution\n\n")
                    
                    # Write each file's patch
                    for file_path, patch_content in patch_data.items():
                        f.write(f"# File: {file_path}\n")
                        f.write(patch_content)
                        f.write("\n\n")
        
        # Check if this is a context-heavy datapoint that needs git repository handling
        issue_id = os.path.basename(issue_path)
        data_id = None
        for id in self.context:
            if str(issue_id) in id:
                data_id = id
                break
        
        use_git_workspace = False
        workspace_volume = None
        
        # Determine if we should use git-based workspace
        if data_id:
            # Check for context-heavy ID pattern
            is_context_heavy = 'agentic_heavy' in data_id
            
            # Check for repo info in CLI args or datapoint context
            repo_url = getattr(self, 'repo_url', None)
            commit_hash = getattr(self, 'commit_hash', None)
            context_data = self.context[data_id].get('context', {})
            
            # CLI args take precedence over datapoint context
            if not repo_url and 'repo' in context_data:
                repo_url = context_data['repo']
            if not commit_hash and 'commit' in context_data:
                commit_hash = context_data['commit']
            
            # Check if this is a context-heavy datapoint
            # The git workspace will be created by the repository's create_repo method
            if (repo_url and commit_hash) or is_context_heavy:
                use_git_workspace = True
                print(f"[INFO] Detected context-heavy datapoint: {data_id}")
                
                # For context heavy datapoints, determine the workspace volume name
                # This should match the volume naming in create_repo
                if is_context_heavy and (repo_url and commit_hash):
                    workspace_volume = f"{data_id}_workspace"
                    print(f"[INFO] Using workspace volume: {workspace_volume}")
                else:
                    # Try to get from existing repository if available
                    for _, data in self.runs.items():
                        if 'repo' in data and data['repo'] is not None:
                            if hasattr(data['repo'], 'issue_path') and data['repo'].issue_path == issue_path:
                                if hasattr(data['repo'], 'volume_name') and data['repo'].volume_name:
                                    workspace_volume = data['repo'].volume_name
                                    break
        
        # Create docker-compose configuration for the agent
        if use_git_workspace and workspace_volume:
            # Use volume-based mounting for git workspaces
            docker_compose = {
                'services': {
                    'agent': {
                        'image': agent,
                        'volumes': [
                            f'{workspace_volume}:/code',
                            './prompt.json:/code/prompt.json',
                            './rundir:/code/rundir'
                        ],
                        'working_dir': '/code',
                        'environment': {
                            'OPENAI_USER_KEY': config.get('OPENAI_USER_KEY', '')
                        }
                    }
                },
                'volumes': {
                    workspace_volume: {
                        'external': True,
                        'name': workspace_volume
                    }
                }
            }
        else:
            # Use traditional directory-based mounting
            docker_compose = {
                'services': {
                    'agent': {
                        'image': agent,
                        'volumes': [
                            './docs:/code/docs',
                            './rtl:/code/rtl',
                            './verif:/code/verif',
                            './rundir:/code/rundir',
                            './prompt.json:/code/prompt.json'
                        ],
                        'working_dir': '/code',
                        'environment': {
                            'OPENAI_USER_KEY': config.get('OPENAI_USER_KEY', '')
                        }
                    }
                }
            }
            print("[INFO] Using traditional directory-based mounting")
        
        # Add golden patch to volumes if it exists
        if hasattr(self, 'include_golden_patch') and self.include_golden_patch:
            golden_patch_file = os.path.join(issue_path, "golden_ref_solution.patch")
            if os.path.exists(golden_patch_file):
                docker_compose['services']['agent']['volumes'].append('./golden_ref_solution.patch:/code/golden_ref_solution.patch')
                print("Golden patch file will be mounted in the agent container at /code/golden_ref_solution.patch")
        
        # Add harness files if requested
        if hasattr(self, 'include_harness') and self.include_harness:
            # The issue_path already contains the test harness necessary files
            # created by the Repository.prepare() method
            
            # First, make sure src directory exists and is populated
            src_dir = os.path.join(issue_path, "src")
            if os.path.exists(src_dir):
                docker_compose['services']['agent']['volumes'].append('./src:/code/src')
                print("Test harness src directory will be mounted in the agent container at /code/src")
                            
            # Add the docker-compose.yml file that was used to run the tests
            docker_compose_yml = os.path.join(issue_path, "docker-compose.yml")
            if os.path.exists(docker_compose_yml):
                docker_compose['services']['agent']['volumes'].append('./docker-compose.yml:/code/docker-compose.yml')
                print("Original docker-compose.yml will be mounted in the agent container at /code/docker-compose.yml")
        
        # Add network configuration if we have a network name
        if hasattr(self, 'network_name') and self.network_name:
            docker_compose['networks'] = {
                'default': {
                    'name': self.network_name,
                    'external': True
                }
            }
            # Ensure agent service uses the network
            docker_compose['services']['agent']['networks'] = ['default']
        
        # Add license network configuration for commercial EDA datapoints
        if data_id:
            from src import commercial_eda
            datapoint = self.context.get(data_id, {})
            if commercial_eda.datapoint_requires_eda_license(datapoint):
                license_network_name = config.get('LICENSE_NETWORK')
                if license_network_name:
                    # Ensure networks section exists
                    if 'networks' not in docker_compose:
                        docker_compose['networks'] = {}
                    
                    # Add the license network
                    docker_compose['networks'][license_network_name] = {
                        'name': license_network_name,
                        'external': True
                    }
                    
                    # Add license network to agent service networks
                    if 'networks' not in docker_compose['services']['agent']:
                        docker_compose['services']['agent']['networks'] = []
                    elif isinstance(docker_compose['services']['agent']['networks'], list):
                        # Networks is already a list, append to it
                        pass
                    else:
                        # Networks might be in dict format, convert to list
                        docker_compose['services']['agent']['networks'] = ['default']
                    
                    # Add license network to the list if not already present
                    if license_network_name not in docker_compose['services']['agent']['networks']:
                        docker_compose['services']['agent']['networks'].append(license_network_name)
                    
                    print(f"Added license network '{license_network_name}' to agent Docker Compose for commercial EDA datapoint")
        
        # Write the Docker Compose file
        with open(docker_compose_path, 'w') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)
        
        # Create before snapshot volume for git workspace volumes (for change detection)
        before_volume = None
        if use_git_workspace and workspace_volume:
            before_volume = self._create_before_snapshot_volume(workspace_volume, issue_path)
        
        print(f"Running agent inside {issue_path}...")
        
        # Note: Network configuration for standard docker-compose.yml will be handled 
        # by create_agent_script or repository methods when shell scripts are generated
        
        # Extract issue ID and repo name for a unique project name
        issue_id = os.path.basename(issue_path)
        harness_dir = os.path.dirname(issue_path)
        repo_name = os.path.basename(os.path.dirname(harness_dir))
        
        # Format only the repo_name to comply with Docker naming requirements
        formatted_repo = ''.join(c.lower() if c.isalnum() or c == '-' or c == '_' else '_' for c in repo_name)
        if not formatted_repo[0].isalnum():
            formatted_repo = 'p' + formatted_repo
        
        # Find repository instance if it exists for this issue path
        repo_instance = None
        for _, data in self.runs.items():
            if 'repo' in data and data['repo'] is not None:
                if data['repo'].issue_path == issue_path:
                    repo_instance = data['repo']
                    break
        
        # Workspace volume should already be set above if available
        
        # Create project name with formatted repo name
        # Use simple naming convention consistent with volume naming
        project_name = f"agent_{formatted_repo}_{issue_id}_{int(time.time())}"
        
        # Create/ensure reports directory exists
        if repo_instance is not None:
            # Use the repository's predefined reports path directly
            reports_dir = repo_instance.report_path
        else:
            # Fall back to manual path calculation
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(issue_path)), "reports")
        
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create logfile path for agent output
        logfile = os.path.join(reports_dir, f"{issue_id}_agent.txt")
        print(f"Agent output will be saved to: {logfile}")
        
        # Create the run_docker_agent.sh script
        if repo_instance is not None:
            # Use Repository's method if we have a Repository instance
            repo_instance.create_agent_script(docker_compose_path, agent)
        else:
            # Otherwise use our local method
            self.create_agent_script(docker_compose_path, project_name)
        
        # Create path to the shell script
        script_path = os.path.join(os.path.dirname(docker_compose_path), 'run_docker_agent.sh')
        
        try:
            # Run the shell script and redirect output to logfile
            print(f"Executing agent script: {script_path}")
            
            # Define kill command for monitoring
            kill_cmd = f"docker compose -f {docker_compose_path} -p {project_name} kill agent"
            
            # Execute the script in a subprocess
            with open(logfile, 'w') as log_file:
                # Use our own exec_timeout style function for consistency with src/repository.py
                p = subprocess.Popen(script_path, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
                pid = p.pid
                
                # Start directory size monitoring if enabled
                if monitor_size:
                    # Start monitoring thread for the issue directory
                    self.dir_monitor.start_monitoring(
                        directory=issue_path,
                        process_id=pid,
                        kill_cmd=kill_cmd
                    )
                
                # Wait for the process to complete with timeout
                try:
                    # Use communicate with timeout instead of wait()
                    p.communicate(timeout=repository.DOCKER_TIMEOUT)
                    returncode = p.returncode
                except subprocess.TimeoutExpired:
                    print(f'Timeout for {script_path} ({repository.DOCKER_TIMEOUT}s) expired')
                    # Kill the process tree
                    if hasattr(repository, 'kill_process_tree'):
                        repository.kill_process_tree(p.pid)
                    else:
                        # Fallback if kill_process_tree is not available
                        try:
                            p.kill()
                        except:
                            pass
                    
                    # Execute kill command
                    subprocess.run(kill_cmd, shell=True)
                    returncode = 1  # Non-zero return code indicating failure
            
            # Legacy network cleanup - only runs if no shared network is configured
            # Since we now always use shared networks, this cleanup is effectively disabled
            if (not hasattr(self, 'network_name') or not self.network_name) and (not hasattr(self, 'manage_network') or self.manage_network):
                try:
                    # Extract project name prefix for filtering (remove timestamp)
                    project_prefix = "_".join(project_name.split("_")[:-1])
                    # Use more robust filtering approach
                    cleanup_cmd = f"docker network ls --filter name={project_prefix} -q | xargs -r docker network rm 2>/dev/null || true"
                    subprocess.run(cleanup_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    # Suppress even the Python exception messages
                    pass
            
            # Generate agent_changes.patch directly from git workspace volume
            if use_git_workspace and workspace_volume and before_volume:
                try:
                    self._generate_volume_changes_patch(workspace_volume, before_volume, issue_path)
                except Exception as e:
                    print(f"[WARNING] Failed to generate volume changes patch: {e}")
                finally:
                    # Clean up the before snapshot volume
                    try:
                        subprocess.run(["docker", "volume", "rm", "-f", before_volume], 
                                     check=False, capture_output=True)
                    except Exception as cleanup_e:
                        print(f"[WARNING] Failed to cleanup before volume {before_volume}: {cleanup_e}")
            
            # Note: Git workspace volume cleanup is handled by the repository's cleanup method
            # to ensure it persists for both harness and agent execution
            
            return returncode, logfile
        except Exception as e:
            logging.error(f"Error running agent script: {str(e)}")
            
            # Log the error to the logfile if possible
            try:
                with open(logfile, 'a') as log_file:
                    log_file.write(f"\nError running agent script: {str(e)}\n")
            except Exception:
                pass
            
            # Legacy network cleanup in exception handler - only runs if no shared network is configured
            # Since we now always use shared networks, this cleanup is effectively disabled
            if (not hasattr(self, 'network_name') or not self.network_name) and (not hasattr(self, 'manage_network') or self.manage_network):
                try:
                    # Extract project name prefix for filtering (remove timestamp)
                    project_prefix = "_".join(project_name.split("_")[:-1])
                    # Use more robust filtering approach
                    cleanup_cmd = f"docker network ls --filter name={project_prefix} -q | xargs -r docker network rm 2>/dev/null || true"
                    subprocess.run(cleanup_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    # Completely suppress errors from cleanup
                    pass
            
            # Note: Git workspace volume cleanup is handled by the repository's cleanup method
            
            return 1, logfile

    def create_agent_script(self, docker_compose_path, project_name):
        """
        Creates a run_docker_agent.sh script to run the agent in a Docker container.
        
        Args:
            docker_compose_path (str): Path to the docker-compose-agent.yml file
            project_name (str): Docker Compose project name
        """
        # Ensure docker_compose_path is absolute
        docker_compose_path = os.path.abspath(docker_compose_path)
        docker_dir = os.path.dirname(docker_compose_path)
        
        # Ensure docker-compose file has network configuration
        # This is the correct place to configure networks - when generating the shell script
        if hasattr(self, 'network_name') and self.network_name:
            print(f"Ensuring {docker_compose_path} has correct network configuration")
            try:
                network_util.add_network_to_docker_compose(docker_compose_path, self.network_name)
            except Exception as e:
                print(f"Warning: Failed to add network configuration to {docker_compose_path}: {str(e)}")
        
        # Create the script path
        script_path = os.path.join(docker_dir, 'run_docker_agent.sh')
        
        # Extract project name prefix for filtering (remove timestamp)
        project_prefix = "_".join(project_name.split("_")[:-1])
        
        # Write the script with better error handling
        with open(script_path, 'w') as script_file:
            script_file.write("#!/bin/bash\n\n")
            script_file.write(f"# Auto-generated script to run agent Docker container\n")
            script_file.write(f"# Usage: {os.path.basename(script_path)} [-d] (where -d enables debug mode with bash entrypoint)\n")
            script_file.write(f"set -e\n\n")
            
            # Parse command line arguments for debug mode
            script_file.write(f"# Parse command line arguments\n")
            script_file.write(f"DEBUG_MODE=false\n")
            script_file.write(f"while getopts 'd' flag; do\n")
            script_file.write(f"  case \"${{flag}}\" in\n")
            script_file.write(f"    d) DEBUG_MODE=true ;;\n")
            script_file.write(f"  esac\n")
            script_file.write(f"done\n\n")
            
            # Add network handling if we have a network name
            if hasattr(self, 'network_name') and self.network_name:
                script_file.write(f"# Use shared bridge network: {self.network_name}\n")
                script_file.write(f"NETWORK_CREATED=0\n\n")
                
                script_file.write(f"# Check if network exists, create if needed\n")
                script_file.write(f"if ! docker network inspect {self.network_name} &>/dev/null; then\n")
                script_file.write(f"  echo \"Creating Docker network {self.network_name}...\"\n")
                script_file.write(f"  docker network create --driver bridge {self.network_name}\n")
                script_file.write(f"  NETWORK_CREATED=1\n")
                script_file.write(f"fi\n\n")
            
            script_file.write(f"# Function to clean up resources\n")
            script_file.write(f"cleanup() {{\n")
            script_file.write(f"  echo \"Cleaning up Docker resources...\"\n")

            # Cleanup image
            script_file.write(f"  docker rmi {project_name}-agent 2>/dev/null || true\n")

            # Only clean up network if we created it or if we're using default networks
            if hasattr(self, 'network_name') and self.network_name:
                script_file.write(f"  if [ $NETWORK_CREATED -eq 1 ]; then\n")
                script_file.write(f"    echo \"Removing Docker network {self.network_name}...\"\n")
                script_file.write(f"    docker network rm {self.network_name} 2>/dev/null || true\n")
                script_file.write(f"  fi\n")
            else:
                # Use more robust filtering approach for default networks
                script_file.write(f"  docker network ls --filter name={project_prefix} -q | xargs -r docker network rm 2>/dev/null || true\n")
            
            script_file.write(f"}}\n\n")
            script_file.write(f"# Set up cleanup trap\n")
            script_file.write(f"trap cleanup EXIT\n\n")
            
            # Run the container with or without debug entrypoint
            script_file.write(f"# Run the agent container\n")
            script_file.write(f"echo \"Running agent with project name: {project_name}\"\n")
            script_file.write(f"# Get current user and group IDs\n")
            script_file.write(f"USER_ID=$(id -u)\n")
            script_file.write(f"GROUP_ID=$(id -g)\n")
            
            # Add memory limits if configured
            agent_memory_limit = config.get("DOCKER_AGENT_MEMORY_LIMIT")
            memory_flags = ""
            if agent_memory_limit:
                memory_flags = f" --memory={agent_memory_limit} --memory-swap={agent_memory_limit}"
                script_file.write(f"# Memory limit: {agent_memory_limit}\n")
            script_file.write(f"\n")
            
            script_file.write(f"if [ \"$DEBUG_MODE\" = true ]; then\n")
            script_file.write(f"  echo \"DEBUG MODE: Starting container with bash entrypoint\"\n")
            script_file.write(f"  docker compose -f {docker_compose_path} -p {project_name} run --rm{memory_flags} --user $USER_ID:$GROUP_ID --entrypoint bash agent\n")
            script_file.write(f"else\n")
            script_file.write(f"  docker compose -f {docker_compose_path} -p {project_name} run --rm{memory_flags} --user $USER_ID:$GROUP_ID agent\n")
            script_file.write(f"fi\n")
            script_file.write(f"exit_code=$?\n\n")
            script_file.write(f"# Exit with the same code as the docker command\n")
            script_file.write(f"exit $exit_code\n")
        
        # Make the script executable
        os.chmod(script_path, 0o755)

        # Ensure script file is flushed
        os.sync()
        time.sleep(0.1)
        
        print(f"Created agent script: {script_path}")

    def th_agent(self, id):
        """
        Process an ID through the agent, similar to th_prepare and th_run.
        This function runs the agent and processes the results.
        """
        print(f"Starting agent processing for {id}...")
        
        try:
            # Setup initial context
            (context, _) = self.initial_context(id)
            
            # Identify repository naming
            name = self.context[id]['id'].split("_")
            repo_name = os.path.join(self.prefix, "cvdp_" + "_".join(name[1:-1]))
            issue_id = int(name[-1])
            issue_path = os.path.join(repo_name, "harness", f"{issue_id}")
            
            # Add prompt.json to the context
            context['prompt.json'] = json.dumps({"prompt": self.context[id]['prompt']})
            
            result = context.copy()
            
            # Check if this is a context-heavy datapoint that uses git workspace volumes
            has_agentic_heavy = 'agentic_heavy' in id
            context_data = self.context[id].get('context', {})
            repo_url = context_data.get('repo')
            commit_hash = context_data.get('commit')
            is_context_heavy = (has_agentic_heavy and bool(repo_url) and bool(commit_hash))
            
            if is_context_heavy:
                
                if not self.golden:
                    # For context-heavy datapoints, just run the agent - volume-based patch generation
                    # is handled in agent_run method
                    agent_status, agent_logfile = self.agent_run(issue_path, self.agent)
                    
                    # Store agent log file info
                    result['agent_logfile'] = agent_logfile
                    
                    if agent_status != 0:
                        error_msg = f"Agent process exited with non-zero status: {agent_status}"
                        logging.error(error_msg)
                        result['agent_error'] = error_msg
                
            else:
                # Traditional agent processing for non-context-heavy datapoints
                # Prepare directories for agent
                dir_names = ["docs", "rundir", "rtl", "verif"]
                before_dir = os.path.join(issue_path, "before")
                os.makedirs(before_dir, exist_ok=True)
                for d in dir_names:
                    src = os.path.join(issue_path, d)
                    bak = os.path.join(before_dir, d)
                    os.makedirs(src, exist_ok=True)
                    if os.path.exists(bak): shutil.rmtree(bak)
                    shutil.copytree(src, bak)
                
                if not self.golden:
                    # Run agent
                    agent_status, agent_logfile = self.agent_run(issue_path, self.agent)
                
                    # Store agent log file info
                    result['agent_logfile'] = agent_logfile
                    
                    if agent_status != 0:
                        error_msg = f"Agent process exited with non-zero status: {agent_status}"
                        logging.error(error_msg)
                        # Store error but also continue to process any files that might have been created
                        result['agent_error'] = error_msg
                    
                    # Clean up any stray Docker resources (with error suppression)
                    try:
                        # Extract issue ID and repo name for project name pattern
                        issue_id_str = str(issue_id)
                        harness_dir = os.path.dirname(issue_path)
                        repo_name_base = os.path.basename(os.path.dirname(harness_dir))
                        
                        # Format repo_name to comply with Docker naming requirements
                        formatted_repo = ''.join(c.lower() if c.isalnum() or c == '-' or c == '_' else '_' for c in repo_name_base)
                        if not formatted_repo[0].isalnum():
                            formatted_repo = 'p' + formatted_repo
                        
                        # Clean up any networks with similar pattern
                        cleanup_cmd = f"docker network ls --filter name=agent_{formatted_repo}_{issue_id_str} -q | xargs -r docker network rm 2>/dev/null || true"
                        subprocess.run(cleanup_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        # Completely suppress errors from cleanup
                        pass
                    
                    # Process differences even if agent had errors, in case partial results were created
                    # Track all changes in a single unified patch
                    unified_patch = []
                
                    for d in dir_names:
                        orig_dir = os.path.join(before_dir, d)
                        mod_dir = os.path.join(issue_path, d)
                        orig_files = {f: os.path.join(orig_dir, f) for f in self._get_files(orig_dir)}
                        mod_files = {f: os.path.join(mod_dir, f) for f in self._get_files(mod_dir)}
                    
                        # Handle added and modified files
                        for rel_path, mod_path in mod_files.items():
                            context_path = os.path.join(d, rel_path)
                            try:
                                with open(mod_path, 'r', encoding='utf-8', errors='replace') as f:
                                    mod_content = f.read()
                                
                                # Added or modified file
                                if rel_path not in orig_files:
                                    # New file - use directly
                                    result[context_path] = mod_content
                                    # Add new file to unified patch
                                    unified_patch.append(f"--- /dev/null\n+++ b/{context_path}\n@@ -0,0 +1,{len(mod_content.splitlines())} @@")
                                    for line in mod_content.splitlines():
                                        unified_patch.append(f"+{line}")
                                    unified_patch.append("")  # Empty line between files
                                else:
                                    with open(orig_files[rel_path], 'r', encoding='utf-8', errors='replace') as f:
                                        orig_content = f.read()
                                    if orig_content != mod_content:
                                        # Add diff to unified patch
                                        diff = self._diff(orig_content, mod_content, context_path)
                                        unified_patch.append(diff)
                                        unified_patch.append("")  # Empty line between files
                                        # Use modified content directly
                                        result[context_path] = mod_content
                            except Exception as file_e:
                                file_error = f"Error processing {mod_path}: {str(file_e)}"
                                logging.error(file_error)
                                # Store the file-specific error but continue processing other files
                                result[f'file_error_{context_path}'] = file_error
                    
                        # Handle deleted files
                        for rel_path, orig_path in orig_files.items():
                            if rel_path not in mod_files:
                                context_path = os.path.join(d, rel_path)
                                with open(orig_path, 'r', encoding='utf-8', errors='replace') as f:
                                    orig_content = f.read()
                                # Add deletion to unified patch
                                unified_patch.append(f"--- a/{context_path}\n+++ /dev/null\n@@ -1,{len(orig_content.splitlines())} +0,0 @@")
                                for line in orig_content.splitlines():
                                    unified_patch.append(f"-{line}")
                                unified_patch.append("")  # Empty line between files
            
                    # Write the unified patch file
                    if unified_patch:
                        patch_file = os.path.join(issue_path, "agent_changes.patch")
                        with open(patch_file, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(unified_patch))
                        result['agent_patch_file'] = patch_file
            
            # Store the results for later use
            self.agent_results[id] = result
            
            # If this ID is already in the runs dict from preparation phase, update it to include agent status
            if id in self.runs:
                if 'agent_error' in result:
                    self.runs[id]['agent_error'] = result['agent_error']
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error in agent processing for {id}: {error_msg}")
            # Create basic context with error information
            self.agent_results[id] = context.copy() if 'context' in locals() else {}
            self.agent_results[id]['agent_error'] = f"th_agent: {error_msg}"
            
            # If this ID is already in the runs dict from preparation phase, update it with error
            if id in self.runs:
                self.runs[id]['agent_error'] = f"th_agent: {error_msg}"

    def all_agent(self):
        """
        Process all IDs through the agent in parallel, similar to all_prepare and all_run.
        """
        from .parallel_executor import ParallelExecutor
        
        executor = ParallelExecutor(num_workers=self.threads, phase_name="Agent Processing")
        executor.execute_parallel_simple(
            task_func=self.th_agent,
            items=list(self.context.keys())
        )

    def result_context (self, id = 0, context = {}, patch : dict = {}):
        """
        Process context and apply patches. For non-golden mode, 
        this now uses the pre-processed results from th_agent.
        """
        result = context.copy()
        
        # Apply patches (for golden mode)
        if self.golden and not self.disable_patch and patch:
            for file, patches in patch.items():
                result[file] = self.apply_patch(patches, context.get(file, ''))
        # assert(False)

        return result
        
    def _get_files(self, directory):
        """Get all files in directory recursively, returning relative paths."""
        return [] if not os.path.exists(directory) else [
            os.path.relpath(os.path.join(root, f), directory)
            for root, _, files in os.walk(directory) for f in files
        ]
        
    def _diff(self, original, modified, path):
        """Generate unified diff between original and modified content."""
        import difflib
        return '\n'.join(difflib.unified_diff(
            original.splitlines(), modified.splitlines(),
            fromfile=f'a/{path}', tofile=f'b/{path}', lineterm=''
        ))

    def all_prepare(self, model : OpenAI_Instance = None):
        """
        Override all_prepare for Agentic to include agent processing after preparation.
        """
        # First, call the parent class implementation for standard preparation
        super().all_prepare(model)
        
        # Then, if not in golden mode, process all with agent
        if not self.golden and self.agent:
            self.all_agent()

    # ----------------------------------------
    # - Git Context and Patch Container Methods
    # ----------------------------------------





    def _generate_volume_changes_patch(self, workspace_volume: str, before_volume: str, issue_path: str):
        """
        Generate agent_changes.patch by comparing the current workspace volume content
        with the before snapshot volume, without extracting files to disk.
        
        For context-heavy agentic datapoints, compares the entire /code directory
        excluding hidden files (.* ) and prompt.json.
        """
        
        try:
            # Generate diff between the two volumes
            # For context-heavy datapoints, we need to go through all /code directory and subdirectories
            # while excluding .*, prompt.json, and rundir/ (since rundir is mounted from host)
            diff_cmd = [
                "docker", "run", "--rm",
                "-v", f"{before_volume}:/before:ro",
                "-v", f"{workspace_volume}:/after:ro",
                "ubuntu:22.04",
                "bash", "-c",
                # Generate diff between before and after volumes, with proper exclusions
                "cd / && "
                "diff -ruN before after "
                "--exclude='.*' --exclude='prompt.json' --exclude='rundir' "
                "2>/dev/null || true"  # diff returns non-zero when differences found
            ]
            
            result = subprocess.run(diff_cmd, capture_output=True, text=True, check=False)
            
            if result.stdout.strip():
                # Process the diff output to clean it up
                diff_output = result.stdout
                # Replace /before and /after paths to make it a proper patch
                diff_output = diff_output.replace("before/", "a/")
                diff_output = diff_output.replace("after/", "b/")
                
                # Write the patch to agent_changes.patch
                patch_file = os.path.join(issue_path, "agent_changes.patch")
                with open(patch_file, 'w', encoding='utf-8') as f:
                    f.write(diff_output)
                
            else:
                # Create empty patch file to indicate no changes
                patch_file = os.path.join(issue_path, "agent_changes.patch")
                with open(patch_file, 'w', encoding='utf-8') as f:
                    f.write("# No changes detected\n")
                
        except Exception as e:
            print(f"[WARNING] Failed to generate patch from volumes: {e}")
            
            # Fallback: create simple status file
            try:
                status_file = os.path.join(issue_path, "agent_changes.patch")
                with open(status_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Error generating patch: {e}\n")
                print(f"[INFO] Created error status in agent_changes.patch")
            except Exception as e2:
                print(f"[ERROR] Failed to create error status file: {e2}")

    def _create_before_snapshot_volume(self, workspace_volume: str, issue_path: str):
        """
        Create a snapshot Docker volume of the workspace volume before the agent runs,
        for later comparison to generate agent_changes.patch.
        
        This captures the state AFTER initial context patches are applied,
        which is the starting point for the agent.
        
        Returns the name of the created before volume.
        """
        # Generate a unique name for the before volume
        before_volume = f"{workspace_volume}_before"
        
        try:
            # Create the before volume
            create_vol_cmd = ["docker", "volume", "create", before_volume]
            subprocess.run(create_vol_cmd, check=True, capture_output=True)
            
            # Copy current state from workspace volume to before volume
            snapshot_cmd = [
                "docker", "run", "--rm",
                "-v", f"{workspace_volume}:/source:ro",
                "-v", f"{before_volume}:/target",
                "ubuntu:22.04",
                "bash", "-c",
                # Copy all content except hidden files and prompt.json
                "cd /source && "
                "find . -maxdepth 1 -type f ! -name '.*' ! -name 'prompt.json' -exec cp {} /target/ \\; && "
                "find . -maxdepth 1 -type d ! -name '.*' ! -name '.' -exec cp -r {} /target/ \\;"
            ]
            
            subprocess.run(snapshot_cmd, check=True, capture_output=True, text=True)
            return before_volume
            
        except Exception as e:
            print(f"[WARNING] Failed to create before snapshot volume: {e}")
            # Try to clean up the volume if it was created
            try:
                subprocess.run(["docker", "volume", "rm", "-f", before_volume], 
                             check=False, capture_output=True)
            except:
                pass
            return None

    def create_repo (self, id : str, model : OpenAI_Instance = None):
        """
        Clone & checkout a real git repo (via CLI flags or JSON fields),
        read only the `external/` folder into memory, then fall back to
        JSON‐dump logic if no git info is present.
        """

        # 1) Unpack the datapoint
        (harness, name, issue, patches) = self.extract_datapoint(id)
        ctx = self.context[id]

        # 2) Pick up the repo URL & commit—CLI flags win over JSON fields
        repo_url   = getattr(self, "repo_url", None) or ctx.get("context", {}).get("repo")
        commit_sha = getattr(self, "commit_hash", None) or ctx.get("context", {}).get("commit")
        data_point = id.split("_")

        try:
            if not os.getenv("CLONE_HTTP") and "github.com/" in repo_url if repo_url else False:
                repo_url   = repo_url.split("github.com/")[-1]
                repo_url   = f"git@github.com:{repo_url}.git"
        except:
            pass

        if repo_url and commit_sha:
            print(f"[DEBUG:create_repo-Agentic] Starting datapoint preparation for id={id}")

            # Use GitRepositoryManager for consistent volume management
            git_manager = git_utils.get_git_manager(self.prefix)
            volume_name = f"{id}_workspace"
            
            # Get patches if available
            patches = ctx.get('patch', {}) if not self.disable_patch else {}
            
            # Determine root directory (extract only external/ folder for CVDP repos)
            root_dir = "external" if 'cvdp_' in repo_url or 'cvdp-' in repo_url else None
            
            # Create the git workspace using the consolidated approach
            success = git_manager.create_volume_with_checkout(
                repo_url=repo_url,
                commit_hash=commit_sha,
                volume_name=volume_name,
                patches=patches,
                root_dir=root_dir
            )
            
            if not success:
                print(f"[ERROR] Failed to create git workspace for {id}, falling back to regular mode")
                # Fall through to the regular mode below
            else:
                (_, repo) = self.create_repository(id, harness, name, issue, {})
                repo.volume_name = volume_name
                
                # Register automatic cleanup for the volume
                import atexit
                def cleanup_volume():
                    try:
                        print(f"[INFO] Cleaning up workspace volume: {volume_name}")
                        import subprocess
                        subprocess.run(
                            ["docker", "volume", "rm", "-f", volume_name],
                            check=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    except Exception as e:
                        print(f"[WARNING] Failed to cleanup workspace volume {volume_name}: {e}")
                
                # Only register cleanup once per volume
                cleanup_attr = f"_cleanup_volume_{id}_registered"
                if not hasattr(atexit, cleanup_attr):
                    atexit.register(cleanup_volume)
                    setattr(atexit, cleanup_attr, True)
                    print(f"[INFO] Registered automatic cleanup for volume: {volume_name}")
                
                # Create workspace volume script for context heavy datapoints
                if 'agentic_heavy' in id:
                    # Create script in the same directory as other scripts (issue_path)
                    repo.create_workspace_volume_script(
                        docker_dir=repo.issue_path,
                        repo_url=repo_url,
                        commit_hash=commit_sha,
                        patches=patches,
                        root_dir=root_dir
                    )
                
                return (True, repo)

        else:
            # 5) Fallback: original JSON‐dump into self.files[id]
            if id not in self.files:
                print(f"Creating harness environment for datapoint: {name}_{issue:04d}")
            if not self.golden and id in self.agent_results:
                self.files[id] = self.agent_results[id]
            else:
                self.files[id] = self.create_context(id, model)

        return self.create_repository(id, harness, name, issue, patches)

    def get_context_for_repo(self, id, model):
        """
        Override to use agent_results when available for non-golden mode.
        """
        # Use agent_results if available (for non-golden mode after agent processing)
        if not self.golden and id in self.agent_results:
            return self.agent_results[id]
        else:
            return self.create_context(id, model)
