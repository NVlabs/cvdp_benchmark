# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import hashlib
import time
import yaml
import logging

def generate_network_name(dataset_path, shared=False):
    """
    Generate a consistent bridge network name based on dataset path and timestamp.
    
    Args:
        dataset_path (str): Path to the dataset file
        shared (bool): If True, creates a shared network name without timestamp
        
    Returns:
        str: The network name to use with Docker
    """
    # Extract just the filename without extension for clarity
    dataset_basename = os.path.basename(dataset_path)
    dataset_name = os.path.splitext(dataset_basename)[0]
    
    # Create a hash of the dataset path to ensure unique networks per dataset
    # but use only the first 8 chars to keep names reasonably short
    dataset_hash = hashlib.md5(dataset_path.encode()).hexdigest()[:8]
    
    # Create a network name with timestamp if not shared
    if shared:
        network_name = f"cvdp-bridge-{dataset_name}-{dataset_hash}"
    else:
        timestamp = int(time.time())
        network_name = f"cvdp-bridge-{dataset_name}-{dataset_hash}-{timestamp}"
    
    # Ensure network name is valid for Docker (only alphanumeric, dash, underscore)
    network_name = ''.join(c if c.isalnum() or c == '-' or c == '_' else '-' for c in network_name)
    
    # Ensure it's not too long (Docker has a 64 character limit)
    if len(network_name) > 64:
        network_name = network_name[:64]
    
    return network_name

def create_docker_network(network_name):
    """
    Create a Docker bridge network if it doesn't exist.
    
    Args:
        network_name (str): Name of the network to create
        
    Returns:
        bool: True if network was created or already exists, False if creation failed
    """
    try:
        # Check if network already exists
        result = subprocess.run(
            f"docker network ls --filter name=^{network_name}$ --format '{{{{.Name}}}}'",
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout.strip() == network_name:
            print(f"Docker network '{network_name}' already exists")
            return True
            
        # Create the network if it doesn't exist
        print(f"Creating Docker network '{network_name}'")
        result = subprocess.run(
            f"docker network create {network_name} --driver bridge",
            shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print(f"Successfully created Docker network '{network_name}'")
            return True
        else:
            print(f"Failed to create Docker network '{network_name}': {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error creating Docker network '{network_name}': {str(e)}")
        return False

def remove_docker_network(network_name, retries=5, retry_delay=1):
    """
    Remove a Docker bridge network if it exists.
    
    Args:
        network_name (str): Name of the network to remove
        
    Returns:
        bool: True if network was removed or didn't exist, False if removal failed
    """
    try:
        # Check if network exists
        result = subprocess.run(
            f"docker network ls --filter name=^{network_name}$ --format '{{{{.Name}}}}'",
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout.strip() != network_name:
            # Don't print a message when the network doesn't exist to reduce verbosity
            return True
            
        for attempt in range(1, retries + 1):
            print(f"Removing Docker network '{network_name}'")
            result = subprocess.run(
                f"docker network rm {network_name}",
                shell=True, capture_output=True, text=True
            )

            if result.returncode == 0:
                print(f"Successfully removed Docker network '{network_name}'")
                return True

            stderr = result.stderr.strip()
            if "active endpoints" not in stderr or attempt == retries:
                print(f"Failed to remove Docker network '{network_name}': {result.stderr}")
                return False

            print(
                f"Docker network '{network_name}' still has active endpoints; "
                f"retrying removal ({attempt}/{retries})"
            )
            time.sleep(retry_delay)

    except Exception as e:
        print(f"Error removing Docker network '{network_name}': {str(e)}")
        return False

def add_network_to_docker_compose(docker_compose_path, network_name):
    """
    Add a network section to a docker-compose file if it doesn't already have one.
    
    Args:
        docker_compose_path (str): Path to the docker-compose.yml file
        network_name (str): Name of the network to add
        
    Returns:
        bool: True if network was added or already existed, False on error
    """
    try:
        # Read the docker-compose file
        with open(docker_compose_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # If the file is empty or invalid, create a basic structure
        if not data:
            data = {'services': {}}
        
        # Check if networks section already exists and if it has a default network
        if 'networks' not in data or 'default' not in data.get('networks', {}):
            # Add/update networks section with our default network
            if 'networks' not in data:
                data['networks'] = {}
                
            data['networks']['default'] = {
                'name': network_name,
                'external': True
            }
            
            # Ensure all services use this network
            if 'services' in data:
                for service_name, service_config in data['services'].items():
                    # Add networks to the service if it doesn't already have it
                    if 'networks' not in service_config:
                        service_config['networks'] = ['default']
        
            # Write the updated docker-compose file
            with open(docker_compose_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                
            return True
        else:
            # Default network already exists, don't modify
            print(f"Default network already exists in {docker_compose_path}, not modifying")
            return True
            
    except Exception as e:
        print(f"Error adding network to docker-compose file {docker_compose_path}: {str(e)}")
        return False
