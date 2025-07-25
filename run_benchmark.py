#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

import argparse
from src import report
from src import wrapper
import json
import os
from src import network_util
import atexit
import sys
from dotenv import load_dotenv
from src.config_manager import config
from src.argparse_common import add_common_arguments, add_validation_checks, clean_filename
from src.logging_util import setup_logging, cleanup_logging

# Load environment variables from .env file
load_dotenv()

# Get subjective scoring flag from ConfigManager
ENABLE_SUBJECTIVE_SCORING = config.get("ENABLE_SUBJECTIVE_SCORING")

def detect_dataset_format(filename, force_agentic=False, force_copilot=False):
    """Detect if a dataset is agentic or copilot type, with option to force agentic or copilot mode."""
    if force_agentic:
        return True
    if force_copilot:
        return False
        
    with open(filename, 'r') as file:
        content = file.readlines()
    
    data = json.loads(content[0])
    id   = data['id'].split('_')[1]
    return id == 'agentic'

class CopilotBenchmark(wrapper.CopilotWrapper):
    def benchmark(self, runs_file = None):
        raw_result_path = os.path.join(self.repo.prefix, "raw_result.json")
        report_path = os.path.join(self.repo.prefix, "report.json")
        
        # If raw_result.json exists, load it instead of rerunning tests
        if os.path.exists(raw_result_path):
            print(f"Using existing raw_result.json from {raw_result_path}")
            with open(raw_result_path, 'r') as f:
                res = json.load(f)
        else:
            if runs_file is None:
                self.repo.process_json()
                
                # If refinement is enabled, run it before preparation
                if hasattr(self.repo, 'refine_model') and self.repo.refine_model:
                    print(f"Refining datapoints using model: {self.repo.refine_model}")
                    refine_results = self.repo.all_refine(model_factory=self.factory)
                    print(f"Refinement completed: {refine_results['refined']} datapoints refined")
                    sys.stdout.flush()

                # Prepare all repositories
                self.repo.all_prepare(self.model)

                # Run all tests
                res = self.repo.all_run(self.model)
            else:
                self.repo.process_json()

                with open (runs_file, 'r+') as runs_f:
                    runs = runs_f.readlines()

                # Replicate repositories
                for run in runs:
                    # From String to Dictionary
                    cxt = json.loads(run)
                    id  = list(cxt.keys())[0]
                    vlt = list(cxt.values())[0]

                    (obj, repo)         = self.repo.set_repo(id=id, context=vlt)
                    self.repo.runs [id] = {'obj' : obj, 'repo' : repo, 'input' : vlt ['input'], 'output' : vlt ['output']}

                res = self.repo.all_run(self.model)

            # Create prefix directory if it doesn't exist
            os.makedirs(self.repo.prefix, exist_ok=True)
            
            # Write results to prefix directory
            with open(raw_result_path, "w+") as f:
                f.write(json.dumps(res))

        return res

    def execute_single(self, issue, runs_file=None):
        """Execute a single issue - similar to harness functionality."""
        raw_result_path = os.path.join(self.repo.prefix, "raw_result.json")
        report_path = os.path.join(self.repo.prefix, "report.json")
        
        # Create directories if they don't exist
        os.makedirs(self.repo.prefix, exist_ok=True)

        # Check if we're using --regenerate-report flag only
        if hasattr(self, 'regenerate_report_only') and self.regenerate_report_only:
            # In this case, we should load from raw_result.json
            if os.path.exists(raw_result_path):
                print(f"Using existing raw_result.json from {raw_result_path} due to --regenerate-report flag")
                with open(raw_result_path, 'r') as f:
                    all_results = json.load(f)
                    if issue in all_results:
                        print(f"Found result for issue {issue} in raw_result.json")
                        return all_results[issue]
                    else:
                        raise Exception(f"Issue {issue} not found in existing raw_result.json")
        
        # Always process the issue, even if it exists in raw_result.json
        if runs_file is None:
            self.repo.process_json()
            
            # If refinement is enabled, run it before preparation
            if hasattr(self.repo, 'refine_model') and self.repo.refine_model:
                print(f"Refining datapoint {issue} using model: {self.repo.refine_model}")
                # First prepare the repository so we have issue_path available
                (_, obj, repo) = self.repo.prepare(issue, self.model)
                # Create model for refinement
                refine_model = self.factory.create_model(model_name=self.repo.refine_model, context=self.repo.folders)
                # Only refine the specific datapoint
                self.repo._try_refine_datapoint(issue, refine_model, repo)
                # Save the refined datapoint
                if hasattr(self.repo, 'refined_datapoints') and issue in self.repo.refined_datapoints:
                    print(f"Saving refined datapoint for {issue}")
                    self.repo._save_refined_datapoints()
                sys.stdout.flush()
            else:
                # If not refining, prepare the repository now
                (_, obj, repo) = self.repo.prepare(issue, self.model)
            
            # For agentic mode, ensure agent processing is done
            if hasattr(self.repo, 'agent') and self.repo.agent and not self.repo.golden:
                # Process this specific issue with the agent
                print(f"Processing issue {issue} with agent {self.repo.agent}")
                self.repo.th_prepare(issue, self.model)
                if hasattr(self.repo, 'th_agent'):
                    self.repo.th_agent(issue)
            
            result = self.repo.run(issue, obj, repo, self.model)
        else:
            self.repo.process_json()

            with open(runs_file, 'r+') as runs_f:
                runs = runs_f.readlines()

            # Replicate repositories
            for run in runs:
                # From String to Dictionary
                cxt = json.loads(run)
                id  = list(cxt.keys())[0]
                vlt = list(cxt.values())[0]

                (obj, repo) = self.repo.set_repo(id=id, context=vlt)
                self.repo.runs[id] = {'obj': obj, 'repo': repo, 'input': vlt['input'], 'output': vlt['output']}

            result = self.repo.run(issue, self.repo.runs[issue]['obj'], self.repo.runs[issue]['repo'], self.model)
        
        # Store this single result in a raw_result file
        with open(raw_result_path, "a+") as f:
            # Try to load existing file first
            try:
                f.seek(0)
                content = f.read()
                if content:
                    all_results = json.loads(content)
                else:
                    all_results = {}
            except json.JSONDecodeError:
                all_results = {}
            
            # Update with new result and write back
            all_results[issue] = result
            f.seek(0)
            f.truncate()
            f.write(json.dumps(all_results, indent=2))
        
        return result

class AgenticBenchmark(wrapper.AgenticWrapper):
    def execute_single(self, issue, runs_file=None):
        """Execute a single issue - similar to harness functionality for agentic mode."""
        raw_result_path = os.path.join(self.repo.prefix, "raw_result.json")
        report_path = os.path.join(self.repo.prefix, "report.json")
        
        # Create directories if they don't exist
        os.makedirs(self.repo.prefix, exist_ok=True)

        # Check if we're using --regenerate-report flag only
        if hasattr(self, 'regenerate_report_only') and self.regenerate_report_only:
            # In this case, we should load from raw_result.json
            if os.path.exists(raw_result_path):
                print(f"Using existing raw_result.json from {raw_result_path} due to --regenerate-report flag")
                with open(raw_result_path, 'r') as f:
                    all_results = json.load(f)
                    if issue in all_results:
                        print(f"Found result for issue {issue} in raw_result.json")
                        return all_results[issue]
                    else:
                        raise Exception(f"Issue {issue} not found in existing raw_result.json")
        
        # Always process the issue, even if it exists in raw_result.json
        if runs_file is None:
            self.repo.process_json()
            
            # If refinement is enabled, run it before preparation
            if hasattr(self.repo, 'refine_model') and self.repo.refine_model:
                print(f"Refining datapoint {issue} using model: {self.repo.refine_model}")
                # First prepare the repository so we have issue_path available
                (_, obj, repo) = self.repo.prepare(issue, self.model)
                # Create model for refinement
                refine_model = self.factory.create_model(model_name=self.repo.refine_model, context=self.repo.folders)
                # Only refine the specific datapoint
                self.repo._try_refine_datapoint(issue, refine_model, repo)
                # Save the refined datapoint
                if hasattr(self.repo, 'refined_datapoints') and issue in self.repo.refined_datapoints:
                    print(f"Saving refined datapoint for {issue}")
                    self.repo._save_refined_datapoints()
                sys.stdout.flush()
            else:
                # If not refining, prepare the repository now
                (_, obj, repo) = self.repo.prepare(issue, self.model)
            
            # For agentic mode, ensure agent processing is done
            if hasattr(self.repo, 'agent') and self.repo.agent and not self.repo.golden:
                # Process this specific issue with the agent
                print(f"Processing issue {issue} with agent {self.repo.agent}")
                # Note: commented out th_prepare as it was in original AgenticHarness
                # self.repo.th_prepare(issue, self.model)
                if hasattr(self.repo, 'th_agent'):
                    self.repo.th_agent(issue)
            
            result = self.repo.run(issue, obj, repo, self.model)
            
            # Store this single result in a raw_result file
            with open(raw_result_path, "a+") as f:
                # Try to load existing file first
                try:
                    f.seek(0)
                    content = f.read()
                    if content:
                        all_results = json.loads(content)
                    else:
                        all_results = {}
                except json.JSONDecodeError:
                    all_results = {}
                
                # Update with new result and write back
                all_results[issue] = result
                f.seek(0)
                f.truncate()
                f.write(json.dumps(all_results, indent=2))
            
            return result
        else:
            with open(runs_file, 'r+') as runs_f:
                runs = runs_f.readlines()

            # Replicate repositories
            for run in runs:
                # From String to Dictionary
                cxt = json.loads(run)
                id  = list(cxt.keys())[0]
                vlt = list(cxt.values())[0]

                (obj, repo) = self.repo.set_repo(id=id, context=vlt)
                self.repo.runs[id] = {'obj': obj, 'repo': repo, 'input': vlt['input'], 'output': vlt['output']}

            return self.repo.run(issue, self.repo.runs[issue]['obj'], self.repo.runs[issue]['repo'], self.model)

def benchmark_main():
    """Main function for the benchmark module."""
    # Parse Creation
    parser = argparse.ArgumentParser(description="Parser for harness evaluation.")
    
    # Add common arguments shared with run_samples.py
    add_common_arguments(parser)

    args = parser.parse_args()

    # Apply common validation checks
    add_validation_checks(args)

    # Apply subjective scoring flag from arguments or environment variable
    # TODO: Temporarily hardcoded to True - previously: args.enable_sbj_scoring or ENABLE_SUBJECTIVE_SCORING
    use_sbj_scoring = True
    if use_sbj_scoring:
        os.environ["ENABLE_SUBJECTIVE_SCORING"] = "true"
        print("LLM-based subjective scoring is enabled")

    # Clean up filename
    filename = clean_filename(args.filename)
    
    return args, filename, use_sbj_scoring

if __name__ == "__main__":

    args, filename, use_sbj_scoring = benchmark_main()

    # Handle dataset transformation if needed
    if args.force_agentic or args.force_copilot:
        # Create a temporary wrapper just for transformation
        transform_wrapper = wrapper.AgenticWrapper(
            filename=filename, 
            golden=(not args.llm), 
            host=getattr(args, 'host', False), 
            prefix=args.prefix,
            force_agentic=args.force_agentic,
            force_agentic_include_golden=args.force_agentic_include_golden,
            force_agentic_include_harness=args.force_agentic_include_harness,
            force_copilot=args.force_copilot,
            copilot_refine=args.copilot_refine
        )
        # Get the transformed filename (if any transformation was applied)
        if args.force_agentic:
            filename = transform_wrapper.transform_dataset_to_agentic(filename) or filename
        elif args.force_copilot:
            filename = transform_wrapper.transform_dataset_to_copilot(filename) or filename
        
        print(f"Using transformed dataset: {filename}")

    # Validate commercial EDA tool setup
    from src import commercial_eda
    eda_validation = commercial_eda.validate_commercial_eda_setup(filename)
    commercial_eda.print_commercial_eda_info(eda_validation)
    
    # Exit if EDA tool validation failed
    if eda_validation['required'] and not eda_validation['validation_passed']:
        print("\nCommercial EDA tool validation failed. Cannot proceed with EDA tool workflows.")
        sys.exit(1)

    # Handle Docker network setup
    shared_network_name = None
    if not args.regenerate_report:
        if args.network_name:
            # Use the specified network name
            shared_network_name = args.network_name
            print(f"Using specified Docker network: {shared_network_name}")
        else:
            # Auto-generate a network name based on the dataset
            shared_network_name = network_util.generate_network_name(filename, shared=True)
            print(f"Generated Docker network name: {shared_network_name}")
    
        # Create the network if we're not using an external network manager
        if not args.external_network:
            print("Creating Docker network for all Docker containers in this run...")
            network_util.create_docker_network(shared_network_name)
            
            # Register cleanup function to remove the network on exit
            def cleanup_network():
                print(f"Cleaning up Docker network: {shared_network_name}")
                network_util.remove_docker_network(shared_network_name)
            
            # Only register cleanup if we're creating the network
            atexit.register(cleanup_network)
            # Mark that we've registered a network cleanup handler
            setattr(atexit, "_network_cleanup_registered", True)

    # Set queue timeout if specified
    if args.queue_timeout is not None:
        os.environ['QUEUE_TIMEOUT'] = str(args.queue_timeout)
        print(f"Queue timeout set to {args.queue_timeout}s")

    # Benchmark Object
    agentic = detect_dataset_format(filename, force_agentic=args.force_agentic, force_copilot=args.force_copilot)

    # Set up shared network info
    network_args = {}
    if shared_network_name:
        network_args = {
            'network_name': shared_network_name,
            'manage_network': not args.external_network
        }
    
    # Add wrapper constructor arguments
    wrapper_args = {
        'filename': filename,
        'golden': (not args.llm),
        'host': getattr(args, 'host', False),
        'prefix': args.prefix,
        'custom_factory_path': args.custom_factory,
        'copilot_refine': args.copilot_refine,
        **network_args
    }

    if agentic:
        # Add agentic-specific arguments
        wrapper_args.update({
            'force_agentic': args.force_agentic,
            'force_agentic_include_golden': args.force_agentic_include_golden,
            'force_agentic_include_harness': args.force_agentic_include_harness,
            'force_copilot': args.force_copilot,
        })
        obj = AgenticBenchmark(**wrapper_args)
        obj.agent = args.agent
        obj.repo.agent = args.agent  # Fix: Also set agent on the repo object
    else:
        obj = CopilotBenchmark(**wrapper_args)

    # Set regenerate_report_only flag if needed
    if args.regenerate_report:
        obj.regenerate_report_only = True

    # Set up logging to save output to run.log in the prefix directory
    setup_logging(args.prefix)
    
    # Register cleanup function to restore original streams on exit
    atexit.register(cleanup_logging)

    obj.repo.threads = args.threads
    obj.repo.disable_patch = args.no_patch  # Pass the disable_patch flag to the repo
    
    # Only create model if we're in LLM mode and either:
    # 1. Not using an agent (non-agentic mode), OR
    # 2. Subjective scoring is enabled (model needed for scoring)
    if args.llm and (not args.agent):
        # Use default model if none was specified
        model_to_use = args.model if args.model is not None else config.get("DEFAULT_MODEL")
        obj.create_model(version=model_to_use)

    try:
        # If --regenerate-report flag is used and raw_result.json exists, load it directly
        raw_result_path = os.path.join(args.prefix, "raw_result.json")
        if args.regenerate_report and os.path.exists(raw_result_path):
            print(f"Loading existing raw_result.json from {raw_result_path}")
            with open(raw_result_path, 'r') as f:
                res = json.load(f)
        else:
            # Check if single issue mode is requested
            if args.id:
                # Single issue execution (both Benchmark and AgenticBenchmark have execute_single)
                exec_result = obj.execute_single(args.id, runs_file=args.answers)
                
                # Add execution information for single issue
                if isinstance(exec_result, dict):
                    model_agent_value = None
                    if args.llm:
                        if agentic and args.agent is not None:
                            model_agent_value = args.agent
                        else:
                            model_agent_value = args.model if args.model is not None else config.get("DEFAULT_MODEL")
                    
                    # Add metadata if not present
                    if 'metadata' not in exec_result:
                        exec_result['metadata'] = {}
                    
                    # Update metadata with current run info
                    exec_result['metadata'].update({
                        'golden_mode': (not args.llm),
                        'disable_patch': args.no_patch,
                        'model_agent': model_agent_value,
                        'force_agentic': args.force_agentic,
                        'force_agentic_include_golden': args.force_agentic_include_golden,
                        'force_agentic_include_harness': args.force_agentic_include_harness,
                        'force_copilot': args.force_copilot,
                        'copilot_refine': args.copilot_refine
                    })
                    
                    # Read the entire raw_result.json to generate a complete report
                    raw_result_path = os.path.join(args.prefix, "raw_result.json")
                    with open(raw_result_path, 'r') as f:
                        all_results = json.load(f)
                    
                    # Create a report using all available results
                    rpt = report.Report(all_results, prefix=args.prefix, dataset_path=filename, 
                                      golden_mode=(not args.llm), 
                                      disable_patch=args.no_patch,
                                      model_agent=model_agent_value,
                                      force_agentic=args.force_agentic,
                                      force_agentic_include_golden=args.force_agentic_include_golden,
                                      force_agentic_include_harness=args.force_agentic_include_harness,
                                      force_copilot=args.force_copilot,
                                      copilot_refine=args.copilot_refine)
                    rpt.report_header()
                    rpt.report_categories()
                    rpt.report_timers()
                    
                    # Check for agent logfile
                    if 'agent_logfile' in exec_result:
                        print(f"Agent logfile: {exec_result['agent_logfile']}")
                    if 'agent_patch_file' in exec_result:
                        print(f"Agent patch file: {exec_result['agent_patch_file']}")
                
                print(json.dumps(exec_result, indent=2))
                
            else:
                # Full benchmark mode (original functionality)
                res = obj.benchmark(runs_file=args.answers)

                # Create report and print results
                rpt = report.Report(
                    res, 
                    prefix=args.prefix, 
                    dataset_path=filename, 
                    golden_mode=(not args.llm), 
                    disable_patch=args.no_patch, 
                    model_agent=args.agent if args.agent else (args.model if args.model is not None else config.get("DEFAULT_MODEL")),
                    force_agentic=args.force_agentic,
                    force_agentic_include_golden=args.force_agentic_include_golden,
                    force_agentic_include_harness=args.force_agentic_include_harness,
                    force_copilot=args.force_copilot,
                    copilot_refine=args.copilot_refine
                )
                rpt.report_header()  # Add header with metadata
                rpt.report_categories()
                rpt.report_timers()

    except Exception as e:
        raise Exception(f"Unable to process the JSON file: {filename}. Error: {str(e)}")

    # If we're managing networks (not external), clean up the network
    # This cleanup is handled by the atexit handler, but add it here explicitly
    # only if we haven't registered the atexit handler for some reason
    if shared_network_name and not args.external_network and not args.regenerate_report and not hasattr(atexit, "_network_cleanup_registered"):
        print(f"Cleaning up Docker network: {shared_network_name}")
        network_util.remove_docker_network(shared_network_name)
