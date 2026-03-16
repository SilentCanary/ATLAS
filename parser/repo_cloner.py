import os
import shutil
import stat
from git import Repo, GitCommandError

class RepoCloner:
    def __init__(self, base_dir="repos"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _remove_readonly(self, func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def clone_repo(self, repo_url):
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        clone_path = os.path.join(self.base_dir, repo_name)

        if os.path.exists(clone_path):
            try:
                # Try to open existing repo
                repo = Repo(clone_path)
                print(f"Repo already exists. Pulling latest changes from {repo_url}...")
                repo.remotes.origin.pull()
            except GitCommandError:
                # If repo is corrupted or not a git repo, delete and reclone
                print("Existing folder is not a valid git repo. Deleting and recloning...")
                shutil.rmtree(clone_path, onerror=self._remove_readonly)
                print(f"Cloning {repo_url}...")
                Repo.clone_from(repo_url, clone_path)
                print("Clone completed.")
        else:
            print(f"Cloning {repo_url}...")
            Repo.clone_from(repo_url, clone_path)
            print("Clone completed.")

        return clone_path