# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Git repository utilities for context-heavy agentic datapoints.

This module provides functionality for:
- Shared repository mirroring (to avoid redundant clones)
- Docker volume management for git-based workspaces
- Thread-safe repository operations with locking
"""

import os
import subprocess
import hashlib
import time
import threading
import tempfile
import shutil
from collections import defaultdict
from typing import Dict, Optional, Tuple
from .config_manager import config

# Global locks for thread-safe repository operations
_repo_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)


def get_repo_hash(repo_url: str) -> str:
    """
    Generate a consistent hash for a repository URL or local path key.

    Args:
        repo_url: Git repository URL or normalized local path

    Returns:
        8-character hex hash of the repository key
    """
    return hashlib.md5(repo_url.encode()).hexdigest()[:8]


def normalize_repo_url(repo_url: str) -> str:
    """
    Normalize repository URL for consistent handling.

    Args:
        repo_url: Original repository URL

    Returns:
        Normalized repository URL (SSH vs HTTPS based on CLONE_HTTP env var)
    """
    try:
        # Convert GitHub HTTPS URLs to SSH unless CLONE_HTTP is set
        if not os.getenv("CLONE_HTTP") and "github.com/" in repo_url:
            if repo_url.startswith("https://github.com/"):
                repo_path = repo_url.replace("https://github.com/", "")
                if not repo_path.endswith(".git"):
                    repo_path += ".git"
                return f"git@github.com:{repo_path}"
            elif "github.com/" in repo_url and not repo_url.startswith("git@"):
                repo_path = repo_url.split("github.com/")[-1]
                if not repo_path.endswith(".git"):
                    repo_path += ".git"
                return f"git@github.com:{repo_path}"
        return repo_url
    except Exception:
        return repo_url


def is_remote_repo_reference(repo_url: str) -> bool:
    """Return True when a repo reference should be treated as a remote URL."""
    return "://" in repo_url or "@" in repo_url


def resolve_heavy_repo_reference(repo_ref: Optional[str], repos_path: Optional[str]) -> Optional[str]:
    """
    Resolve a context-heavy datapoint repo reference under CVDP_HEAVY_REPOS_PATH.

    If a datapoint names ``my_repo`` and the configured directory contains either
    ``my_repo/`` or ``my_repo.bundle``, return the matching local path. Directory
    matches are preferred to preserve the previous unpacked-repo behavior.
    """
    if not repo_ref or not repos_path:
        return repo_ref

    if os.path.isabs(repo_ref) or is_remote_repo_reference(repo_ref):
        return repo_ref

    repo_path = os.path.join(repos_path, repo_ref)
    if os.path.exists(repo_path):
        return repo_path

    if not repo_path.endswith(".bundle"):
        bundle_path = f"{repo_path}.bundle"
        if os.path.isfile(bundle_path):
            return bundle_path

    return repo_path


def get_repo_mirror_filename(repo_url: str) -> str:
    """Return the mirror directory name used for a repository URL or local path."""
    if os.path.isdir(repo_url) or os.path.isfile(repo_url):
        mirror_key = os.path.abspath(repo_url)
    else:
        mirror_key = normalize_repo_url(repo_url)
    return f"{get_repo_hash(mirror_key)}.git"


class GitRepositoryManager:
    """
    Manages shared git repositories and Docker volumes for context-heavy agentic datapoints.
    
    Key features:
    - Shared repository mirrors to avoid redundant disk usage
    - Docker volume creation and management
    - Thread-safe operations with per-repository locking
    """
    
    def __init__(self, cache_dir: str):
        """
        Initialize the git repository manager.
        
        Args:
            cache_dir: Base directory for storing git mirrors and logs
        """
        self.cache_dir = cache_dir
        self.mirrors_dir = os.path.join(cache_dir, "mirrors")
        self.logs_dir = os.path.join(cache_dir, "logs")
        
        # Ensure directories exist
        os.makedirs(self.mirrors_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Ensure patch_image Docker image exists
        self._ensure_patch_image()
    
    def _ensure_patch_image(self):
        """
        Ensure the patch_image Docker image exists, building it if necessary.
        This image contains git and is used for repository operations.
        """
        try:
            result = subprocess.run(
                ["docker", "images", "-q", "patch_image"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if not result.stdout.strip():
                print("[INFO] Docker image 'patch_image' not found, building it...")
                
                # Create temporary dockerfile
                dockerfile_content = "FROM ubuntu:22.04\nRUN apt update && apt install -y git"
                dockerfile_path = os.path.join(self.cache_dir, "Dockerfile.patch_image")
                
                with open(dockerfile_path, "w") as f:
                    f.write(dockerfile_content)
                
                # Build image
                subprocess.run(
                    ["docker", "build", "-t", "patch_image", "-f", dockerfile_path, self.cache_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print("[INFO] Successfully built patch_image")
            else:
                print("[INFO] Docker image 'patch_image' already exists")
                
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Failed to ensure patch_image exists: {e}")
            # Continue anyway - the error will be caught when actually using the image
    
    def _get_repo_hash(self, repo_url: str) -> str:
        """
        Generate a consistent hash for a repository URL to enable sharing.
        
        Args:
            repo_url: Git repository URL
            
        Returns:
            8-character hex hash of the repository URL
        """
        return get_repo_hash(repo_url)
    
    def _normalize_repo_url(self, repo_url: str) -> str:
        """
        Normalize repository URL for consistent handling.
        
        Args:
            repo_url: Original repository URL
            
        Returns:
            Normalized repository URL (SSH vs HTTPS based on CLONE_HTTP env var)
        """
        return normalize_repo_url(repo_url)
    
    def get_or_create_mirror(self, repo_url: str) -> str:
        """
        Get or create a shared mirror for the given repository URL.

        Args:
            repo_url: Git repository URL, local directory path, or local bundle path

        Returns:
            Path to the shared mirror directory
        """
        # Handle local directory paths and bundle files by creating a bare mirror
        # from them. Bundles are immutable snapshots, so an existing mirror is
        # reused just like local unpacked repositories.
        if os.path.isdir(repo_url) or os.path.isfile(repo_url):
            abs_path = os.path.abspath(repo_url)
            mirror_filename = get_repo_mirror_filename(repo_url)
            repo_hash = mirror_filename.removesuffix(".git")
            mirror_path = os.path.join(self.mirrors_dir, mirror_filename)
            log_file = os.path.join(self.logs_dir, f"{repo_hash}_clone.log")
            source_kind = "local repo" if os.path.isdir(repo_url) else "local bundle"
            clone_cmd = ["git", "clone", "--mirror", abs_path, mirror_path]
            if os.path.isdir(repo_url):
                clone_cmd.insert(3, "--local")

            with _repo_locks[mirror_path]:
                if not os.path.exists(mirror_path):
                    print(f"[INFO] Creating mirror from {source_kind} {abs_path}")
                    with open(log_file, 'w') as logfile:
                        try:
                            subprocess.run(
                                clone_cmd,
                                check=True,
                                stdout=logfile,
                                stderr=subprocess.STDOUT,
                                text=True
                            )
                            # Allow the patch container to fetch by SHA (orphan commits are
                            # branch tips, so they're reachable from advertised refs)
                            subprocess.run(
                                ["git", "config", "uploadpack.allowReachableSHA1InWant", "true"],
                                cwd=mirror_path,
                                check=True,
                                stdout=logfile,
                                stderr=subprocess.STDOUT,
                                text=True
                            )
                            print(f"[INFO] Successfully created mirror from {source_kind}: {mirror_path}")
                        except subprocess.CalledProcessError as e:
                            print(f"[ERROR] Failed to mirror {source_kind} {abs_path}: {e}")
                            if os.path.exists(mirror_path):
                                shutil.rmtree(mirror_path, ignore_errors=True)
                            raise
                else:
                    print(f"[INFO] Using existing mirror for {source_kind}: {mirror_path}")

            return os.path.abspath(mirror_path)

        normalized_url = self._normalize_repo_url(repo_url)
        mirror_filename = get_repo_mirror_filename(repo_url)
        repo_hash = mirror_filename.removesuffix(".git")
        mirror_path = os.path.join(self.mirrors_dir, mirror_filename)
        log_file = os.path.join(self.logs_dir, f"{repo_hash}_clone.log")

        # Use per-repository locking to prevent concurrent operations
        with _repo_locks[mirror_path]:
            if not os.path.exists(mirror_path):
                print(f"[INFO] Cloning {normalized_url} into shared mirror at {mirror_path}")

                with open(log_file, 'w') as logfile:
                    try:
                        subprocess.run(
                            ["git", "clone", "--mirror", normalized_url, mirror_path],
                            check=True,
                            stdout=logfile,
                            stderr=subprocess.STDOUT,
                            text=True
                        )
                        print(f"[INFO] Successfully created mirror: {mirror_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] Failed to clone {normalized_url}: {e}")
                        # Clean up partial clone
                        if os.path.exists(mirror_path):
                            subprocess.run(["rm", "-rf", mirror_path], check=False)
                        raise
            else:
                # Update existing mirror
                print(f"[INFO] Updating existing mirror: {mirror_path}")
                with open(log_file, 'a') as logfile:
                    try:
                        logfile.write(f"\n--- Update at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                        subprocess.run(
                            ["git", "fetch", "--all"],
                            cwd=mirror_path,
                            check=True,
                            stdout=logfile,
                            stderr=subprocess.STDOUT,
                            text=True
                        )
                    except subprocess.CalledProcessError as e:
                        print(f"[WARNING] Failed to update mirror {mirror_path}: {e}")
                        # Don't fail on update errors - use existing mirror

        return os.path.abspath(mirror_path)
    
    def create_volume_with_checkout(self, 
                                  repo_url: str, 
                                  commit_hash: str, 
                                  volume_name: str,
                                  patches: Optional[Dict[str, str]] = None,
                                  root_dir: Optional[str] = None) -> bool:
        """
        Create a Docker volume with git checkout and optional patches applied.
        Uses the original file-based approach for better reliability.
        
        Args:
            repo_url: Git repository URL
            commit_hash: Commit hash to checkout
            volume_name: Name for the Docker volume
            patches: Optional dict of file->patch_content mappings
            root_dir: Optional subdirectory to extract (e.g., "external")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get shared mirror
            mirror_path = self.get_or_create_mirror(repo_url)
            
            # Create Docker volume
            print(f"[INFO] Creating Docker volume: {volume_name}")
            subprocess.run(
                ["docker", "volume", "create", volume_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Create temporary patch directory
            patch_dir = os.path.join(self.cache_dir, f"patches_{volume_name}")
            os.makedirs(patch_dir, exist_ok=True)
            
            try:
                # Prepare patch files using the original approach
                self._prepare_patch_files(patch_dir, patches, root_dir)
                
                # Run the patch container using the original approach
                success = self._run_patch_container(
                    commit_hash, patch_dir, mirror_path, volume_name, root_dir
                )
                
                if success:
                    print(f"[INFO] Successfully prepared volume: {volume_name}")
                    
                    # Fix ownership of files in the volume to match current user
                    # This is needed because the agent runs with --user $USER_ID:$GROUP_ID
                    self._fix_volume_ownership(volume_name)
                    
                    return True
                else:
                    return False
                    
            finally:
                # Clean up temporary patch directory
                try:
                    import shutil
                    shutil.rmtree(patch_dir, ignore_errors=True)
                except Exception:
                    pass
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to create volume {volume_name}: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                print(f"[ERROR] stdout: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"[ERROR] stderr: {e.stderr}")
            
            # Clean up failed volume
            try:
                subprocess.run(
                    ["docker", "volume", "rm", "-f", volume_name],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except Exception:
                pass
            
            return False
    
    def _fix_volume_ownership(self, volume_name: str):
        """
        Fix ownership of files in the Docker volume to match the current user.
        This is needed because the agent runs with --user $USER_ID:$GROUP_ID.
        """
        try:
            import os
            user_id = os.getuid()
            group_id = os.getgid()
            
            print(f"[INFO] Fixing ownership in volume {volume_name} to {user_id}:{group_id}")
            
            # Use a container to change ownership of all files in the volume
            chown_cmd = [
                "docker", "run", "--rm",
                "-v", f"{volume_name}:/workspace",
                "ubuntu:22.04",
                "chown", "-R", f"{user_id}:{group_id}", "/workspace"
            ]
            
            result = subprocess.run(
                chown_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            print(f"[INFO] Successfully fixed ownership in volume {volume_name}")
            
        except Exception as e:
            print(f"[WARNING] Failed to fix ownership in volume {volume_name}: {e}")
            # Don't fail the entire operation if ownership fix fails
    
    def _prepare_patch_files(self, patch_dir: str, patches: Optional[Dict[str, str]], root_dir: Optional[str]):
        """
        Prepare patch files on disk using the original approach.
        
        Args:
            patch_dir: Directory to store patch files
            patches: Optional dict of file->patch_content mappings
            root_dir: Optional subdirectory prefix for file paths
        """
        patch_path = os.path.join(patch_dir, "patch.diff")
        
        # Create patch file
        if patches:
            with open(patch_path, "w", encoding="utf-8", newline='\n') as f:
                for file_path, patch_body in patches.items():
                    if not patch_body or not patch_body.strip():
                        continue
                    filename = f"{root_dir}/{file_path}" if root_dir else file_path
                    f.write(f"--- a/{filename}\n+++ b/{filename}\n{patch_body}\n\n")
        else:
            # Create empty patch file
            with open(patch_path, 'w') as f:
                f.write('')
        
        # Create safe patch application script
        patch_script = os.path.join(patch_dir, "safe-apply.sh")
        script = '#!/bin/bash\nPATCH="$1"\nif [ -s "$PATCH" ]; then\n\tgit apply -v "$PATCH"\nelse\n\techo "Patch is empty, skipping apply."\nfi'
        
        with open(patch_script, 'w', encoding="utf-8", newline='\n') as f:
            f.write(script)
        
        # Make script executable
        os.chmod(patch_script, 0o755)
    
    def _run_patch_container(self, commit_hash: str, patch_dir: str, mirror_path: str, 
                           volume_name: str, root_dir: Optional[str]) -> bool:
        """
        Run the patch container using the original approach.
        
        Args:
            commit_hash: Commit hash to checkout
            patch_dir: Directory containing patch files
            mirror_path: Path to git mirror
            volume_name: Docker volume name
            root_dir: Optional subdirectory to copy
            
        Returns:
            True if successful, False otherwise
        """
        try:
            root = f"{root_dir}/." if root_dir else "."
            log_file = os.path.join(patch_dir, "patch_log.txt")
            
            # Build command using original approach
            patch_cmd = [
                "docker", "run", "--rm",
                "-v", f"{os.path.abspath(mirror_path)}:/repo:ro",
                "-v", f"{os.path.abspath(patch_dir)}:/patch:ro",
                "-v", f"{volume_name}:/workspace",
                "patch_image",
                "bash", "-c",
                "set -ex && "
                "mkdir /rundir && cd /rundir && "
                "git config --global --add safe.directory /repo && "
                "git init && "
                "git remote add origin /repo && "
                f"git fetch --depth 1 origin {commit_hash} && "
                f"git checkout {commit_hash} && "
                "bash /patch/safe-apply.sh /patch/patch.diff && "
                f"cp -a ./{root} /workspace/"
            ]
            
            print(f"[INFO] Running patch container: {' '.join(patch_cmd)}")
            
            # Execute with logging
            with open(log_file, 'w') as logfile:
                subprocess.run(
                    patch_cmd, 
                    check=True, 
                    stdout=logfile, 
                    stderr=subprocess.STDOUT, 
                    text=True
                )
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Patch container failed: {e}")
            return False
    

    
    def cleanup_volume(self, volume_name: str) -> bool:
        """
        Clean up a Docker volume.
        
        Args:
            volume_name: Name of the volume to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"[INFO] Removing Docker volume: {volume_name}")
            subprocess.run(
                ["docker", "volume", "rm", "-f", volume_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Failed to remove volume {volume_name}: {e}")
            return False
    
    def volume_exists(self, volume_name: str) -> bool:
        """
        Check if a Docker volume exists.
        
        Args:
            volume_name: Name of the volume to check
            
        Returns:
            True if volume exists, False otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "volume", "ls", "--format", "{{.Name}}"],
                capture_output=True,
                text=True,
                check=True
            )
            return volume_name in result.stdout.splitlines()
        except subprocess.CalledProcessError:
            return False
    



def get_git_manager(prefix: str) -> GitRepositoryManager:
    """
    Get a GitRepositoryManager instance for the given prefix.
    
    Args:
        prefix: Base directory prefix for the benchmark run
        
    Returns:
        GitRepositoryManager instance
    """
    cache_dir = os.path.join(prefix, "git_cache")
    return GitRepositoryManager(cache_dir)
