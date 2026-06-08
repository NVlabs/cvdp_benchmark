# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import tempfile
import unittest

from src.git_utils import GitRepositoryManager, resolve_heavy_repo_reference


class HeavyRepoReferenceTest(unittest.TestCase):

    def test_resolves_existing_directory_before_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = os.path.join(tmp, "my_repo")
            bundle_path = os.path.join(tmp, "my_repo.bundle")
            os.mkdir(repo_dir)
            with open(bundle_path, "w", encoding="utf-8"):
                pass

            resolved = resolve_heavy_repo_reference("my_repo", tmp)

            self.assertEqual(resolved, repo_dir)

    def test_resolves_bundle_when_directory_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = os.path.join(tmp, "my_repo.bundle")
            with open(bundle_path, "w", encoding="utf-8"):
                pass

            resolved = resolve_heavy_repo_reference("my_repo", tmp)

            self.assertEqual(resolved, bundle_path)

    def test_preserves_remote_references(self):
        remote = "git@github.com:example/my_repo.git"

        resolved = resolve_heavy_repo_reference(remote, "/repos")

        self.assertEqual(resolved, remote)


class BundleMirrorTest(unittest.TestCase):

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git is required for bundle mirror tests")

    def test_creates_mirror_from_bundle_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_repo = os.path.join(tmp, "src")
            bundle_path = os.path.join(tmp, "src.bundle")
            mirrors_dir = os.path.join(tmp, "cache", "mirrors")
            logs_dir = os.path.join(tmp, "cache", "logs")
            os.makedirs(mirrors_dir)
            os.makedirs(logs_dir)

            subprocess.run(
                ["git", "init", src_repo],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            subprocess.run(["git", "-C", src_repo, "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", src_repo, "config", "user.name", "Test User"], check=True)
            with open(os.path.join(src_repo, "file.txt"), "w", encoding="utf-8") as f:
                f.write("hello\n")
            subprocess.run(["git", "-C", src_repo, "add", "file.txt"], check=True)
            subprocess.run(
                ["git", "-C", src_repo, "commit", "-m", "init"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            commit = subprocess.check_output(["git", "-C", src_repo, "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", "-C", src_repo, "bundle", "create", bundle_path, "--all"], check=True)

            manager = GitRepositoryManager.__new__(GitRepositoryManager)
            manager.mirrors_dir = mirrors_dir
            manager.logs_dir = logs_dir

            mirror_path = manager.get_or_create_mirror(bundle_path)

            is_bare = subprocess.check_output(
                ["git", "-C", mirror_path, "rev-parse", "--is-bare-repository"],
                text=True
            ).strip()
            subprocess.run(["git", "-C", mirror_path, "cat-file", "-e", f"{commit}^{{commit}}"], check=True)
            checkout_repo = os.path.join(tmp, "checkout")
            subprocess.run(
                ["git", "init", checkout_repo],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            subprocess.run(["git", "-C", checkout_repo, "remote", "add", "origin", mirror_path], check=True)
            subprocess.run(
                ["git", "-C", checkout_repo, "fetch", "--depth", "1", "origin", commit],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            subprocess.run(
                ["git", "-C", checkout_repo, "checkout", commit],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.assertEqual(is_bare, "true")


class PatchPreparationTest(unittest.TestCase):

    def test_skips_blank_patch_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = GitRepositoryManager.__new__(GitRepositoryManager)

            manager._prepare_patch_files(
                tmp,
                {
                    "rtl/empty.sv": "",
                    "rtl/whitespace.sv": " \n\t",
                },
                root_dir="external",
            )

            with open(os.path.join(tmp, "patch.diff"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "")

    def test_writes_nonblank_patch_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = GitRepositoryManager.__new__(GitRepositoryManager)

            manager._prepare_patch_files(
                tmp,
                {
                    "rtl/empty.sv": "",
                    "rtl/real.sv": "@@ -1 +1 @@\n-old\n+new\n",
                },
                root_dir="external",
            )

            with open(os.path.join(tmp, "patch.diff"), encoding="utf-8") as f:
                patch = f.read()

            self.assertNotIn("empty.sv", patch)
            self.assertIn("--- a/external/rtl/real.sv", patch)
            self.assertIn("@@ -1 +1 @@", patch)


if __name__ == "__main__":
    unittest.main()
