# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import time
import subprocess
import yaml
import psutil
from src import subjective
from src import network_util
import dotenv
from src.dir_monitor import DirectorySizeMonitor
from src.config_manager import config
from src.constants import (
    SCORING_CONFIG, 
    LLM_SUBJECTIVE_CATEGORIES,
    BLEU_SCORING_CATEGORIES
)
import re

dotenv.load_dotenv()

# ----------------------------------------
# - Global Configurations
# ----------------------------------------

# Get Docker timeout values from ConfigManager
DOCKER_TIMEOUT = config.get('DOCKER_TIMEOUT')
DOCKER_TIMEOUT_AGENT = config.get('DOCKER_TIMEOUT_AGENT')

def kill_process_tree(pid):

    try:
        parent = psutil.Process(pid)

        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()

    except psutil.NoSuchProcess:
        pass


def apply_template_substitution(content: str) -> str:
    """
    Apply template variable substitution for EDA tool infrastructure.
    
    Replaces placeholders in the format __VARIABLE__ with configured values:
    - __VERIF_EDA_IMAGE__ -> VERIF_EDA_IMAGE config value
    - __LICENSE_NETWORK__ -> LICENSE_NETWORK config value  
    - __OSS_SIM_IMAGE__ -> OSS_SIM_IMAGE config value
    - __OSS_PNR_IMAGE__ -> OSS_PNR_IMAGE config value
    
    Args:
        content: String content with potential template variables
        
    Returns:
        String with template variables substituted
    """
    # Define template mappings
    substitutions = {
        '__VERIF_EDA_IMAGE__': config.get('VERIF_EDA_IMAGE'),
        '__LICENSE_NETWORK__': config.get('LICENSE_NETWORK'),
        '__OSS_SIM_IMAGE__': config.get('OSS_SIM_IMAGE'),
        '__OSS_PNR_IMAGE__': config.get('OSS_PNR_IMAGE')
    }
    
    # Apply substitutions for any placeholders found
    for placeholder, value in substitutions.items():
        if value and placeholder in content:
            content = content.replace(placeholder, value)
    
    return content


def add_license_network_to_docker_compose(content: str, license_network_name: str) -> str:
    """
    Add or update license network configuration in Docker Compose YAML content.
    
    This function ensures that commercial EDA datapoints have the correct license
    network configuration, handling both hardcoded 'licnetwork' references and
    adding the license network as an additional external network.
    
    Args:
        content: Docker Compose YAML content as string
        license_network_name: Name of the license network to add/configure
        
    Returns:
        Updated Docker Compose YAML content as string
    """
    import yaml
    
    try:
        # Parse the YAML content
        data = yaml.safe_load(content)
        
        # If the file is empty or invalid, skip processing
        if not data or 'services' not in data:
            return content
        
        # Ensure networks section exists
        if 'networks' not in data:
            data['networks'] = {}
        
        # Add/update the license network configuration
        # Replace any existing 'licnetwork' with the configured name
        if 'licnetwork' in data['networks'] and license_network_name != 'licnetwork':
            # Move licnetwork config to the new name
            data['networks'][license_network_name] = data['networks'].pop('licnetwork')
        
        # Ensure license network is configured as external
        if license_network_name not in data['networks']:
            data['networks'][license_network_name] = {}
        
        data['networks'][license_network_name].update({
            'name': license_network_name,
            'external': True
        })
        
        # Update service network references
        for service_name, service_config in data['services'].items():
            if 'networks' in service_config:
                # Replace 'licnetwork' references in service networks
                if isinstance(service_config['networks'], list):
                    # Handle list format: [default, licnetwork]
                    networks = service_config['networks']
                    for i, network in enumerate(networks):
                        if network == 'licnetwork' and license_network_name != 'licnetwork':
                            networks[i] = license_network_name
                elif isinstance(service_config['networks'], dict):
                    # Handle dict format: {default: {}, licnetwork: {}}
                    if 'licnetwork' in service_config['networks'] and license_network_name != 'licnetwork':
                        service_config['networks'][license_network_name] = service_config['networks'].pop('licnetwork')
        
        # Convert back to YAML string
        return yaml.dump(data, default_flow_style=False)
        
    except Exception as e:
        # If YAML parsing fails, return original content
        print(f"Warning: Failed to process license network in YAML: {e}")
        return content

class Repository:

    def __init__(self,
            repo         = str,
            id           = int,
            context      = list[list[dict()]],
            harness      = list[list[dict()]],
            patches      = list[str],
            debug        = False,
            host         = False,
            sbj_llm_model    = None,
            network_name = None,
            manage_network = True,
            requires_eda_license = False
        ):

        self.name         = repo
        self.id           = id
        self.context      = context
        self.harness      = harness
        self.patches      = patches
        self.issue_path   = None
        self.logfile      = None
        self.host         = host
        self.debug        = debug
        self.sbj_llm_model    = sbj_llm_model  # Added LLM model parameter
        self.requires_eda_license = requires_eda_license  # Flag indicating if this datapoint requires EDA license network
        # Network info
        self.network_name = network_name
        self.manage_network = manage_network
        # Directory size monitor
        self.dir_monitor = DirectorySizeMonitor()

        # Automatically prepare environment
        self.prepare()

        # Harness Properties
        self.n_gram   = SCORING_CONFIG['N_GRAM_DEFAULT']
        self.rouge_th = SCORING_CONFIG['ROUGE_THRESHOLD']
        self.bleu_th  = SCORING_CONFIG['BLEU_THRESHOLD']
        self.llm_score_th = SCORING_CONFIG['SUBJECTIVE_THRESHOLD_DEFAULT']

    # ----------------------------------------
    # - Create Folders
    # ----------------------------------------

    def safely_create_dir(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

    def create_folders(self, path):

        self.safely_create_dir(f"{path}")
        self.safely_create_dir(f"{path}/rtl")
        self.safely_create_dir(f"{path}/verif")
        self.safely_create_dir(f"{path}/docs")
        self.safely_create_dir(f"{path}/src")
        self.safely_create_dir(f"{path}/rundir")

    def write_file(self, filename, content):

        file = os.path.join(self.name, filename)
        dir  = os.path.dirname(file)

        if not os.path.exists(dir):
            os.makedirs(dir)

        try:
            with open(f"{file}", 'w+', encoding="utf-8") as file:
                file.write(content)

        except:
            print(f"Failed to write file: {filename}")

    # ----------------------------------------
    # - Restore Harness Context
    # ----------------------------------------

    def restore_files(self, files):

        for file, content in files.items():

            # Apply centralized template substitution for EDA infrastructure
            content = apply_template_substitution(content)
            
            # Add license network configuration for commercial EDA datapoints
            if self.requires_eda_license and file.endswith('docker-compose.yml'):
                license_network_name = config.get('LICENSE_NETWORK')
                if license_network_name:
                    content = add_license_network_to_docker_compose(content, license_network_name)
                           
            # Filter out rundir volumes from docker-compose.yml
            if file.endswith('docker-compose.yml'):
                
                try:
                    # Parse the YAML content
                    data = yaml.safe_load(content)
                    
                    # Check if it's a valid docker-compose file with services
                    if data and 'services' in data:
                        # For each service, filter the volumes
                        for service_name, service_config in data['services'].items():
                            if 'volumes' in service_config:
                                # Filter out any volumes with '/code' path
                                filtered_volumes = []
                                for volume in service_config['volumes']:
                                    if '/code' in volume:
                                        print(f"Warning: Removing '/code' volume mapping: {volume} in issue '{self.name}' (id {self.id})")
                                    else:
                                        filtered_volumes.append(volume)
                                service_config['volumes'] = filtered_volumes
                                
                        # Add network configuration if not already present
                        if self.network_name and 'networks' not in data:
                            data['networks'] = {
                                'default': {
                                    'name': self.network_name,
                                    'external': True
                                }
                            }

                        # Convert back to YAML
                        content = yaml.dump(data, default_flow_style=False)
                except Exception as e:
                    print(f"Error processing docker-compose.yml: {str(e)}")

            self.write_file(f"harness/{self.id}/{file}", content)

    def try_create_dir(self, path):

        try:
            os.makedirs(path, exist_ok=True)

        except:
            print(f"Failed to create {path}.")

    def create_issue_dirs(self, issue_path):

        self.try_create_dir(self.name)
        self.try_create_dir(f"{self.name}/harness")
        self.try_create_dir(f"{self.name}/reports")
        self.create_folders(issue_path)

    # ----------------------------------------
    # - Docker Command Line
    # ----------------------------------------

    def docker_cmd(self, issue_path):

        cwd  = os.getcwd()
        key  = config.get('OPENAI_USER_KEY')
        path = os.path.abspath(issue_path)

        # Running docker services
        volumes = [f'-v "{path}/{vol}:/code/{vol}"' for vol in ["docs", "rundir", "rtl", "verif", "src"]]
        volumes.extend([f'-v "{cwd}/src/llm_lib:/pysubj"'])

        cmd     = " ".join(volumes)
        cmd    += f" --rm -w /code/rundir"

        # Adding OpenAI Key to the command line
        if key:
            cmd    += f" --env OPENAI_USER_KEY={key}"

        return cmd

    def exec_timeout(self, cmd, kill = None, out = None, monitor_dir = None, monitor_kill_cmd = None):

        # ----------------------------------------
        # - Args
        # ----------------------------------------

        kargs = {'shell' : True}

        if out is not None:
            kargs['stdout'] = out

        # Start the process
        p = subprocess.Popen(f"{cmd}", **kargs)
        pid = p.pid

        # Start directory size monitoring if enabled
        if monitor_dir is not None and hasattr(self, 'dir_monitor'):
            # Start monitoring thread for the directory
            self.dir_monitor.start_monitoring(
                directory=monitor_dir,
                process_id=pid,
                kill_cmd=monitor_kill_cmd or kill
            )

        # Define the timeout
        try:
            p.communicate(timeout=DOCKER_TIMEOUT)

        except subprocess.TimeoutExpired:
            print(f'Timeout for {cmd} ({DOCKER_TIMEOUT}s) expired')
            kill_process_tree(p.pid)

            # If kill command is defined
            if kill:
                subprocess.run(kill, shell = True)

            return p.returncode, pid

        return p.returncode, pid

    def log_run(self, cmd, kill = None, logfile = "", monitor_dir = None, monitor_kill_cmd = None):

        if (self.debug):
            print(f"{cmd}")

        start_time = time.time()

        if logfile != "":
            with open(logfile, 'w+') as out:
                returncode, pid = self.exec_timeout(cmd, kill, out, monitor_dir, monitor_kill_cmd)
        else:
            returncode, pid = self.exec_timeout(cmd, kill, None, monitor_dir, monitor_kill_cmd)

        return {"result" : returncode, "log" : logfile, "error_msg" : None, "execution" : time.time() - start_time, "pid": pid}

    def log_docker(self, docker : str = "", cmd : str = "", service : str = "", logfile : str = "", 
                  monitor_size=True):
        # Ensure docker variable is absolute path
        docker = os.path.abspath(docker)
        
        # Ensure docker-compose file has network configuration before proceeding
        # This is the correct place to configure networks - when generating the shell script
        if self.network_name:
            try:
                print(f"Ensuring {docker} has correct network configuration")
                network_util.add_network_to_docker_compose(docker, self.network_name)
            except Exception as e:
                print(f"Warning: Failed to add network configuration to {docker}: {str(e)}")
        
        # Generate a unique project name using repo name, ID and timestamp
        docker_dir = os.path.dirname(docker)
        harness_dir = os.path.dirname(docker_dir)
        repo_name = os.path.basename(os.path.dirname(harness_dir))
        
        # Format only the repo_name to comply with Docker naming requirements
        formatted_repo = ''.join(c.lower() if c.isalnum() or c == '-' or c == '_' else '_' for c in repo_name)
        if not formatted_repo[0].isalnum():
            formatted_repo = 'p' + formatted_repo
        
        # Create project name with formatted repo name
        project_name = f"{formatted_repo}_{self.id}_{int(time.time())}"

        # Extract project name prefix for filtering (remove timestamp)
        project_prefix = "_".join(project_name.split("_")[:-1])

        # Create the docker command with project name
        # line_cmd = f"docker compose -f {docker} -p {project_name} run {cmd} {service}"
        kill_cmd = f"docker compose -f {docker} -p {project_name} kill {service}"
        
        # Save the command with absolute paths to a shell script
        # Get the directory where the docker-compose file is located
        script_path = os.path.join(docker_dir, f'run_docker_harness_{service}.sh')
        line_cmd = script_path
        
        # Write the script with better error handling
        with open(script_path, 'w') as script_file:
            script_file.write("#!/bin/bash\n\n")
            script_file.write(f"# Auto-generated script to run harness Docker container\n")
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
            if self.network_name:
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
            script_file.write(f"  docker compose -f {docker} -p {project_name} kill {service} 2>/dev/null || true\n")
            
            # Cleanup image
            script_file.write(f"  docker rmi {project_name}-{service} 2>/dev/null || true\n")

            # Only clean up network if we created it
            if self.network_name:
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
            
            # Run the harness container with or without debug entrypoint
            script_file.write(f"# Run the harness container\n")
            script_file.write(f"echo \"Running harness with project name: {project_name}\"\n")
            script_file.write(f"# Get current user and group IDs\n")
            script_file.write(f"USER_ID=$(id -u)\n")
            script_file.write(f"GROUP_ID=$(id -g)\n")
            
            # Add memory limits if configured
            harness_memory_limit = config.get("DOCKER_HARNESS_MEMORY_LIMIT")
            memory_flags = ""
            if harness_memory_limit:
                memory_flags = f" --memory={harness_memory_limit} --memory-swap={harness_memory_limit}"
                script_file.write(f"# Memory limit: {harness_memory_limit}\n")
            script_file.write(f"\n")
            
            script_file.write(f"if [ \"$DEBUG_MODE\" = true ]; then\n")
            script_file.write(f"  echo \"DEBUG MODE: Starting container with bash entrypoint\"\n")
            script_file.write(f"  docker compose -f {docker} -p {project_name} run --rm{memory_flags} --user $USER_ID:$GROUP_ID -e HOME=/code/rundir --entrypoint bash {cmd} {service}\n")
            script_file.write(f"else\n")
            script_file.write(f"  docker compose -f {docker} -p {project_name} run --rm{memory_flags} --user $USER_ID:$GROUP_ID -e HOME=/code/rundir {cmd} {service}\n")
            script_file.write(f"fi\n")
            script_file.write(f"exit_code=$?\n\n")
            script_file.write(f"# Exit with the same code as the docker command\n")
            script_file.write(f"exit $exit_code\n")

        # Make the script executable
        os.chmod(script_path, 0o755)

        # Ensure script file is flushed
        os.sync()
        time.sleep(0.1)

        # Start the Docker process
        if self.debug:
            print(f"Running command: {line_cmd}")
        
        # Run using log_run, which uses exec_timeout
        # Pass the directory to monitor if monitoring is enabled
        monitor_dir = docker_dir if monitor_size else None
        result = self.log_run(line_cmd, kill_cmd, logfile, monitor_dir, kill_cmd)
        
        # Only clean up the specific network created for this run if we're managing networks
        # and not using our shared network
        if not self.network_name and self.manage_network:
            try:
                # Use more robust filtering approach
                cleanup_cmd = f"docker network ls --filter name={project_prefix} -q | xargs -r docker network rm 2>/dev/null || true"
                subprocess.run(cleanup_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                # Suppress all cleanup errors
                pass
        
        return result

    def create_workspace_volume_script(self, docker_dir, repo_url=None, commit_hash=None, patches=None, root_dir=None):
        """
        Creates create_workspace_volume.sh and destroy_workspace_volume.sh scripts for context heavy datapoints.
        These scripts leverage existing git cache mirrors and use the same approach as GitRepositoryManager.
        
        Args:
            docker_dir (str): Directory where the scripts will be created
            repo_url (str): Git repository URL
            commit_hash (str): Commit hash to checkout
            patches (dict): Optional patches to apply
            root_dir (str): Optional subdirectory to extract (e.g., "external")
        """
        volume_name = self.volume_name or f"{self.id}_workspace"
        
        # Create the create_workspace_volume.sh script
        create_script_path = os.path.join(docker_dir, 'create_workspace_volume.sh')
        with open(create_script_path, 'w') as script_file:
            script_file.write("#!/bin/bash\n")
            script_file.write("# Auto-generated script to create workspace volume for context heavy datapoint\n")
            script_file.write(f"# Datapoint ID: {self.id}\n")
            script_file.write(f"# Volume name: {volume_name}\n")
            script_file.write(f"# Repository: {repo_url}\n")
            script_file.write(f"# Commit: {commit_hash}\n\n")
            
            script_file.write("set -e\n\n")
            
            script_file.write("echo \"Creating workspace volume for context heavy datapoint...\"\n")
            script_file.write(f"echo \"Volume name: {volume_name}\"\n")
            if repo_url:
                script_file.write(f"echo \"Repository: {repo_url}\"\n")
            if commit_hash:
                script_file.write(f"echo \"Commit: {commit_hash}\"\n")
            script_file.write("\n")
            
            # Check if volume already exists
            script_file.write("# Check if volume already exists\n")
            script_file.write(f"if docker volume inspect {volume_name} &>/dev/null; then\n")
            script_file.write(f"  echo \"Volume {volume_name} already exists. Removing it first...\"\n")
            script_file.write(f"  docker volume rm -f {volume_name}\n")
            script_file.write("fi\n\n")
            
            # Create the volume
            script_file.write("# Create Docker volume\n")
            script_file.write(f"echo \"Creating Docker volume: {volume_name}\"\n")
            script_file.write(f"docker volume create {volume_name}\n\n")
            
            if repo_url and commit_hash:
                # Find existing git cache mirror
                script_file.write("# Look for existing git cache mirror\n")
                script_file.write("SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n")
                script_file.write("WORK_DIR=\"$(dirname \"$(dirname \"$(dirname \"$SCRIPT_DIR\")\")\")\"  # Go up to work directory\n")
                script_file.write("GIT_CACHE_DIR=\"$WORK_DIR/git_cache/mirrors\"\n")
                script_file.write("MIRROR_DIR=\"\"\n\n")
                
                script_file.write("# Find the mirror directory (should be named with repo hash)\n")
                script_file.write("if [ -d \"$GIT_CACHE_DIR\" ]; then\n")
                script_file.write("  for mirror in \"$GIT_CACHE_DIR\"/*.git; do\n")
                script_file.write("    if [ -d \"$mirror\" ]; then\n")
                script_file.write("      MIRROR_DIR=\"$mirror\"\n")
                script_file.write("      echo \"Found git cache mirror: $MIRROR_DIR\"\n")
                script_file.write("      break\n")
                script_file.write("    fi\n")
                script_file.write("  done\n")
                script_file.write("fi\n\n")
                
                script_file.write("if [ -z \"$MIRROR_DIR\" ]; then\n")
                script_file.write("  echo \"ERROR: No git cache mirror found in $GIT_CACHE_DIR\"\n")
                script_file.write("  echo \"Please run the benchmark with -l -g first to create the git cache.\"\n")
                script_file.write("  exit 1\n")
                script_file.write("fi\n\n")
                
                # Ensure patch_image exists
                script_file.write("# Ensure patch_image Docker image exists\n")
                script_file.write("if ! docker image inspect patch_image &>/dev/null; then\n")
                script_file.write("  echo \"Building patch_image Docker image...\"\n")
                script_file.write("  TEMP_DOCKERFILE=$(mktemp)\n")
                script_file.write("  cat > \"$TEMP_DOCKERFILE\" << 'EOF'\n")
                script_file.write("FROM ubuntu:22.04\n")
                script_file.write("RUN apt update && apt install -y git\n")
                script_file.write("EOF\n")
                script_file.write("  docker build -t patch_image -f \"$TEMP_DOCKERFILE\" .\n")
                script_file.write("  rm \"$TEMP_DOCKERFILE\"\n")
                script_file.write("fi\n\n")
                
                # Create temporary directory for patches
                script_file.write("# Create temporary directory for patches\n")
                script_file.write("PATCH_DIR=$(mktemp -d)\n")
                script_file.write("cleanup() {\n")
                script_file.write("  rm -rf \"$PATCH_DIR\"\n")
                script_file.write("}\n")
                script_file.write("trap cleanup EXIT\n\n")
                
                # Create patch files using the same format as GitRepositoryManager
                script_file.write("# Create patch.diff file (same format as GitRepositoryManager)\n")
                script_file.write("cat > \"$PATCH_DIR/patch.diff\" << 'PATCH_EOF'\n")
                if patches:
                    for file_path, patch_body in patches.items():
                        # Use same format as GitRepositoryManager: add proper headers
                        filename = f"{root_dir}/{file_path}" if root_dir else file_path
                        script_file.write(f"--- a/{filename}\n")
                        script_file.write(f"+++ b/{filename}\n")
                        script_file.write(patch_body)
                        script_file.write("\n\n")
                script_file.write("PATCH_EOF\n\n")
                
                # Create safe patch application script (same as GitRepositoryManager)
                script_file.write("# Create safe patch application script\n")
                script_file.write("cat > \"$PATCH_DIR/safe-apply.sh\" << 'SCRIPT_EOF'\n")
                script_file.write("#!/bin/bash\n")
                script_file.write("PATCH=\"$1\"\n")
                script_file.write("if [ -s \"$PATCH\" ]; then\n")
                script_file.write("	git apply -v \"$PATCH\"\n")
                script_file.write("else\n")
                script_file.write("	echo \"Patch is empty, skipping apply.\"\n")
                script_file.write("fi\n")
                script_file.write("SCRIPT_EOF\n")
                script_file.write("chmod +x \"$PATCH_DIR/safe-apply.sh\"\n\n")
                
                # Setup git repository in volume using mirror
                script_file.write("# Setup git repository in volume using existing mirror\n")
                script_file.write("echo \"Setting up workspace from git mirror...\"\n")
                script_file.write("USER_ID=$(id -u)\n")
                script_file.write("GROUP_ID=$(id -g)\n")
                script_file.write("docker run --rm \\\n")
                script_file.write("  -v \"$MIRROR_DIR:/repo:ro\" \\\n")
                script_file.write("  -v \"$PATCH_DIR:/patch:ro\" \\\n")
                script_file.write(f"  -v {volume_name}:/workspace \\\n")
                script_file.write("  patch_image bash -c '\n")
                script_file.write("    set -ex\n")
                script_file.write("    mkdir /rundir && cd /rundir\n")
                script_file.write("    git config --global --add safe.directory /repo\n")
                script_file.write("    git init\n")
                script_file.write("    git remote add origin /repo\n")
                script_file.write(f"    git fetch --depth 1 origin {commit_hash}\n")
                script_file.write(f"    git checkout {commit_hash}\n")
                script_file.write("    bash /patch/safe-apply.sh /patch/patch.diff\n")
                
                # Copy to workspace (same approach as GitRepositoryManager)
                root = f"{root_dir}/." if root_dir else "."
                script_file.write(f"    cp -a ./{root} /workspace/\n")
                script_file.write("    echo \"Workspace setup complete\"\n")
                script_file.write("'\n\n")
                
                # Fix volume ownership to current user
                script_file.write("# Fix volume ownership to current user\n")
                script_file.write("echo \"Fixing volume ownership...\"\n")
                script_file.write("docker run --rm \\\n")
                script_file.write(f"  -v {volume_name}:/workspace \\\n")
                script_file.write("  ubuntu:22.04 \\\n")
                script_file.write("  chown -R \"$USER_ID:$GROUP_ID\" /workspace\n\n")
            
            script_file.write("echo \"Workspace volume creation complete!\"\n")
            script_file.write(f"echo \"Volume {volume_name} is ready for use.\"\n")
            script_file.write("echo \"You can now run the agent and harness scripts.\"\n")
        
        # Make create script executable
        os.chmod(create_script_path, 0o755)
        
        # Create the destroy_workspace_volume.sh script
        destroy_script_path = os.path.join(docker_dir, 'destroy_workspace_volume.sh')
        with open(destroy_script_path, 'w') as script_file:
            script_file.write("#!/bin/bash\n")
            script_file.write("# Auto-generated script to destroy workspace volume for context heavy datapoint\n")
            script_file.write(f"# Datapoint ID: {self.id}\n")
            script_file.write(f"# Volume name: {volume_name}\n\n")
            
            script_file.write("set -e\n\n")
            
            script_file.write("echo \"Destroying workspace volume for context heavy datapoint...\"\n")
            script_file.write(f"echo \"Volume name: {volume_name}\"\n\n")
            
            # Check if volume exists
            script_file.write("# Check if volume exists\n")
            script_file.write(f"if ! docker volume inspect {volume_name} &>/dev/null; then\n")
            script_file.write(f"  echo \"Volume {volume_name} does not exist. Nothing to destroy.\"\n")
            script_file.write("  exit 0\n")
            script_file.write("fi\n\n")
            
            # Remove the volume
            script_file.write("# Remove Docker volume\n")
            script_file.write(f"echo \"Removing Docker volume: {volume_name}\"\n")
            script_file.write(f"docker volume rm -f {volume_name}\n\n")
            
            script_file.write("echo \"Workspace volume destruction complete!\"\n")
            script_file.write(f"echo \"Volume {volume_name} has been removed.\"\n")
        
        # Make destroy script executable
        os.chmod(destroy_script_path, 0o755)
        
        print(f"[INFO] Created workspace volume scripts:")
        print(f"[INFO]   Create: {create_script_path}")
        print(f"[INFO]   Destroy: {destroy_script_path}")

    def create_agent_script(self, docker_compose_path, agent_image=None):
        """
        Creates a run_docker_agent.sh script to run the agent in a Docker container.
        
        Args:
            docker_compose_path (str): Path to the docker-compose-agent.yml file
            agent_image (str, optional): Docker image to use for the agent
        """
        # Ensure docker_compose_path is absolute
        docker_compose_path = os.path.abspath(docker_compose_path)
        docker_dir = os.path.dirname(docker_compose_path)
        
        # Generate a unique project name using repo name and ID
        harness_dir = os.path.dirname(docker_dir)
        repo_name = os.path.basename(os.path.dirname(harness_dir))
        
        # Format repo_name to comply with Docker naming requirements
        formatted_repo = ''.join(c.lower() if c.isalnum() or c == '-' or c == '_' else '_' for c in repo_name)
        if not formatted_repo[0].isalnum():
            formatted_repo = 'p' + formatted_repo
        
        # Create project name with formatted repo name - using bash command for timestamp
        project_name = f"agent_{formatted_repo}_{self.id}_$(date +%s)"
        
        # Extract project name prefix for filtering (remove timestamp part)
        project_prefix = f"agent_{formatted_repo}_{self.id}"
        
        # Modify the docker-compose file to use our shared network if it doesn't have networks defined
        if self.network_name:
            try:
                network_util.add_network_to_docker_compose(docker_compose_path, self.network_name)
            except Exception as e:
                print(f"Warning: Failed to add network configuration to {docker_compose_path}: {str(e)}")
        
        # Create the script path
        script_path = os.path.join(docker_dir, 'run_docker_agent.sh')
        
        # Write the script with better error handling
        with open(script_path, 'w') as script_file:
            script_file.write("#!/bin/bash\n\n")
            script_file.write(f"# Auto-generated script to run agent Docker container\n")
            script_file.write(f"# Usage: {os.path.basename(script_path)} [-d] (where -d enables debug mode with bash entrypoint)\n")
            
            # If agent_image is provided, add a comment about which image will be used
            if agent_image:
                script_file.write(f"# Using agent image: {agent_image}\n")
                
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
            if self.network_name:
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
            
            # Only clean up network if we created it
            if self.network_name:
                script_file.write(f"  if [ $NETWORK_CREATED -eq 1 ]; then\n")
                script_file.write(f"    echo \"Removing Docker network {self.network_name}...\"\n")
                script_file.write(f"    docker network rm {self.network_name} 2>/dev/null || true\n")
                script_file.write(f"  fi\n")
            else:
                # Use more robust filtering approach
                script_file.write(f"  docker network ls --filter name={project_prefix} -q | xargs -r docker network rm 2>/dev/null || true\n")
                
            script_file.write(f"}}\n\n")
            script_file.write(f"# Set up cleanup trap\n")
            script_file.write(f"trap cleanup EXIT\n\n")
            
            # Run the agent container with or without debug entrypoint
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

    # ----------------------------------------
    # - Encapsulate Harness Execution Handler
    # ----------------------------------------
        
    def sbj(self, response, reference, category, problem_prompt=""):
        """
        Run subjective scoring on a response.
        
        Args:
            response: The response to evaluate
            reference: The reference answer
            category: The category number
            problem_prompt: Optional problem prompt for context
            
        Returns:
            Tuple of (results, error_count)
        """
        start_time = time.time()
        result = []
        err = 0
        logfile = ""

        try:
            # LLM-based subjective scoring if model is provided and it is LLM subjective categories
            if self.sbj_llm_model and category not in BLEU_SCORING_CATEGORIES:
                # Run LLM-based subjective scoring
                llm_score = self.subjective_score(response, reference, problem_prompt)
                llm_time = time.time()
                llm_pass = llm_score >= self.llm_score_th

                err = (not llm_pass)
                if logfile != "":
                    # LLM Score Logfile
                    with open(f"{logfile}_llm_score.txt", 'w+') as out:
                        out.write(res + f"\n\nLLM Score (0-1) : {llm_score}\n")

                result.append({"result": err, "log": f"{logfile}_llm_score.txt", "error_msg": None, "execution": llm_time - start_time, "llm_score": llm_score})
            else:
                # Traditional metrics - ROUGE and BLEU
                rouge = subjective.calculate_ROUGE(response, reference, self.n_gram)
                rouge_time = time.time()

                bleu = subjective.calculate_BLEU(response, reference, self.n_gram)
                bleu_time = time.time()

                rouge_pass = rouge > self.rouge_th
                bleu_pass = bleu > self.bleu_th

                err = (not rouge_pass) + (not bleu_pass)

                if logfile != "":
                    # Rouge Logfile
                    with open(f"{logfile}_rouge.txt", 'w+') as out:
                        out.write(res + f"\n\nScore : {rouge}\n")

                    # Bleu Logfile
                    with open(f"{logfile}_bleu.txt", 'w+') as out:
                        out.write(res + f"\n\nScore : {bleu}\n")
                    
                # Store BLEU score in result for BLEU scoring categories
                if category in BLEU_SCORING_CATEGORIES:
                    result.append({"result": err, "log": f"{logfile}_rouge.txt", "error_msg": None, "execution": rouge_time - start_time})
                    result.append({"result": err, "log": f"{logfile}_bleu.txt", "error_msg": None, "execution": bleu_time - rouge_time, "bleu_score": bleu})
                else:
                    # For LLM scoring categories, store the LLM score
                    result.append({"result": err, "log": f"{logfile}_rouge.txt", "error_msg": None, "execution": rouge_time - start_time})
                    result.append({"result": err, "log": f"{logfile}_bleu.txt", "error_msg": None, "execution": bleu_time - rouge_time})

        except Exception as e:
            print(f"Datapoint failed to execute subjective tests: {str(e)}")
            result = [{"result": 1, "log": None, "error_msg": f"Failed to execute subjective tests: {str(e)}", "execution": time.time() - start_time}]
            err = 3

        return (result, err)
        
    def subjective_score(self, response, reference, problem_prompt: str = ""):
        """
        Use an LLM to perform subjective scoring on a scale of 0.0-1.0.
        
        Args:
            response (str): The generated response to evaluate
            reference (str): The reference response to compare against
            problem_prompt (str): The original problem prompt for additional context
            
        Returns:
            float: A normalized score from 0.0-1.0 where 1.0 is perfect match and 0.0 is no match
        """
        try:
            # Use the model provided during initialization
            # if self.sbj_llm_model is not None:
            # Call the subjective_score method on the model
            score = self.sbj_llm_model.subjective_score(response, reference, problem_prompt)
            
            if self.debug:
                print(f"LLM-based subjective score: {score}/1.0 (threshold: {self.llm_score_th})")
            return score
            # else:
            #     # No model available - use fallback scoring method
            #     print("Warning: No LLM model provided for subjective scoring. Using fallback method.")
            #     # Simple fallback - compare word count similarity as a basic metric
            #     ref_words = set(reference.lower().split())
            #     res_words = set(response.lower().split())
                
            #     if len(ref_words) == 0:
            #         return 0.0  # Return lowest score for empty reference
                    
            #     # Calculate Jaccard similarity
            #     intersection = len(ref_words.intersection(res_words))
            #     union = len(ref_words.union(res_words))
            #     similarity = intersection / union if union > 0 else 0
                
            #     # Similarity is already in 0-1 range
            #     score = similarity
            #     print(f"Fallback similarity score: {score}/1.0")
            #     return score
                
        except Exception as e:
            print(f"Error in LLM subjective scoring: {str(e)}")
            # Return middle score on error to avoid failing the test completely
            return 0.0

    def obj_harness(self, issue_path : str = "", logfile : str = "", uut : str = None):

        docker = os.path.join(f"{issue_path}", "docker-compose.yml")

        if os.path.exists(docker):

            with open(docker, 'r') as f:
                data = yaml.safe_load(f)

            results = []

            # Identify services
            services = data['services'].keys()
            error    = 0

            for i, service in enumerate(services):
                print(f"Running service: {service}:\n")
                service_log = f'{logfile}{f"_{service}" if len(services) > 1 else ""}.txt'
                opts   = self.docker_cmd(issue_path)
                result = self.log_docker(docker, opts, service, service_log)
                error += result['result']
                results.append(result)

        else:

            # Objective LLM Evaluation
            sys.path.append("./src/llm_lib")
            from src.llm_lib.evaluator import Evaluator

            criteria_files = []
            print(issue_path)

            for file in os.listdir(os.path.join(issue_path, "src")):

                if file.endswith(".json"):
                    criteria_files.append(os.path.join(issue_path, "src", file))

            # Define the criterias

            llm = Evaluator(criteria_files)

            llm.model["type"] = "Response"
            llm.model["text"] = uut

            start_time = time.time()
            result     = llm.evaluate(self.id)
            results    = [{"result" : result, "log" : f"reports_{self.id}.json", "error_msg": None, "execution" : time.time() - start_time}]
            error      = not result

        return (results, error)

    def obj (self, uut : str = None):

        try:
            result = self.obj_harness(self.issue_path, self.logfile, uut)
        except:
            result = ([{"result" : 1, "log" : None, "error_msg": "Failed to execute objective harness", "execution": 0}], 1)
            print(f"[ERROR] Failed to execute {uut} harness...")

        return result

    def prepare(self):

        self.issue_path = os.path.join(f"{self.name}", "harness", f"{self.id}")
        self.report_path = os.path.join(f"{self.name}", "reports")

        # ----------------------------------------
        # - Create Context
        # ----------------------------------------

        self.create_issue_dirs(self.issue_path)
        self.restore_files(self.context)

        # ----------------------------------------
        # - Create Harness Context
        # ----------------------------------------

        if self.harness:
            self.restore_files(self.harness)

        # ----------------------------------------
        # - Update Docker compose with network settings
        # ----------------------------------------
        
        # NOTE: Network configuration for docker-compose files is now handled 
        # directly in log_docker() and create_agent_script() when the shell scripts are generated

        self.logfile = os.path.abspath(os.path.join(self.report_path, f"{self.id}"))

    def run(self):

        # ----------------------------------------
        # - Execute Harness
        # ----------------------------------------

        print(f"Running all tests within: {self.issue_path}")

        try:

            # Execute if exists a objective test
            if self.harness:
                (results, error) = self.obj(self.issue_path, self.logfile)
            else:
                raise ValueError("No harness found")

        except Exception as e:
            error   = 1
            results = []
            raise Exception(e)

        return (results, error)

    def clean_up(self):
        pass

class AgenticRepository(Repository):
    """
    A subclass of Repository that handles agentic evaluation.
    """

    def __init__(self,
            repo         = str,
            id           = int,
            context      = list[list[dict()]],
            harness      = list[list[dict()]],
            patches      = list[str],
            debug        = False,
            host         = False,
            network_name = None,
            manage_network = True,
            requires_eda_license = False
        ):

        super().__init__(repo, id, context, harness, patches, debug, host, network_name=network_name, manage_network=manage_network, requires_eda_license=requires_eda_license)
        self.volume_name = None

    def prepare(self):

        self.issue_path = os.path.join(f"{self.name}", "harness", f"{self.id}")
        self.report_path = os.path.join(f"{self.name}", "reports")

        self.create_issue_dirs(self.issue_path)
        
        # For agentic heavy datapoints, restore minimal context (e.g., prompt.json)
        if self.context:
            self.restore_files(self.context)

        if self.harness:
            self.restore_files(self.harness)

        self.logfile = os.path.abspath(os.path.join(self.report_path, f"{self.id}"))
    
    # ----------------------------------------
    # - Docker Command Line
    # ----------------------------------------

    def docker_cmd(self, issue_path):

        cwd  = os.getcwd()
        key  = os.getenv('OPENAI_USER_KEY')
        path = os.path.abspath(issue_path)

        if self.volume_name:

            # Mount only external/ into /code
            volumes = [f'-v {self.volume_name}:/code']
            volumes.extend([f'-v "{path}/src:/src"'])
            volumes.extend([f'-v "{path}/rundir:/rundir"'])
            volumes.extend([f'-v "{cwd}/llm_lib:/pysubj"'])

        else:
            raise ValueError("Unable to identify volume name for datapoint execution.")

        cmd     = " ".join(volumes)
        cmd    += f" --rm -w /rundir"

        # Adding OpenAI Key to the command line
        if key:
            cmd += f" --env OPENAI_USER_KEY={key}"
        return cmd

    def create_folders(self, path):

        self.safely_create_dir(f"{path}")
        self.safely_create_dir(f"{path}/external")
        self.safely_create_dir(f"{path}/rundir")
    
    def clean_up(self):
        """
        Clean up resources used by this repository, including Docker volumes.
        """
        if self.volume_name:
            try:
                print(f"[INFO] Cleaning up workspace volume: {self.volume_name}")
                import subprocess
                subprocess.run(
                    ["docker", "volume", "rm", "-f", self.volume_name],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except Exception as e:
                print(f"[WARNING] Failed to cleanup workspace volume {self.volume_name}: {e}")
        
        # Call parent cleanup if it exists
        if hasattr(super(), 'clean_up'):
            super().clean_up()