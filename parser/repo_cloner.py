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

    def parse_github_url(self, repo_url):
       

        parts = repo_url.strip("/").split("/")

        if "tree" in parts:
            tree_index = parts.index("tree")
            branch = parts[tree_index + 1]

            repo_url_clean = "/".join(parts[:tree_index])
            repo_url_clean = "https://" + repo_url_clean.replace("https://", "")

        else:
            branch = None
            repo_url_clean = repo_url

        if not repo_url_clean.endswith(".git"):
            repo_url_clean += ".git"

        repo_name = repo_url_clean.split("/")[-1].replace(".git", "")

        return repo_url_clean, repo_name, branch
    
    def clone_repo(self, repo_url):

        repo_url, repo_name, branch = self.parse_github_url(repo_url)

        clone_path = os.path.join(self.base_dir, repo_name)

        if os.path.exists(clone_path):
            try:
                repo = Repo(clone_path)

                print(f"Repo already exists. Pulling latest changes...")
                repo.remotes.origin.pull()

            except GitCommandError:
                print("Invalid repo folder. Re-cloning...")

                shutil.rmtree(clone_path, onerror=self._remove_readonly)

                print(f"Cloning {repo_url}...")
                repo = Repo.clone_from(repo_url, clone_path)

        else:
            print(f"Cloning {repo_url}...")
            repo = Repo.clone_from(repo_url, clone_path)

        if branch:
            print(f"Checking out branch: {branch}")
            repo.git.checkout(branch)

        print("Clone completed.")

        return clone_path
