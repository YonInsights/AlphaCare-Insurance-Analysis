import os
import subprocess
from typing import Optional

def create_branch(branch_name: str) -> bool:
    """Create a new git branch"""
    try:
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def commit_changes(message: str) -> bool:
    """Commit changes with a message"""
    try:
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', message], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def merge_branch(
    source_branch: str,
    target_branch: str = 'main',
    message: Optional[str] = None
) -> bool:
    """Merge source branch into target branch"""
    try:
        # Checkout target branch
        subprocess.run(['git', 'checkout', target_branch], check=True)
        
        # Merge source branch
        if message:
            subprocess.run(
                ['git', 'merge', source_branch, '-m', message],
                check=True
            )
        else:
            subprocess.run(['git', 'merge', source_branch], check=True)
        
        return True
    except subprocess.CalledProcessError:
        return False

def create_pull_request(
    title: str,
    body: str,
    source_branch: str,
    target_branch: str = 'main'
) -> bool:
    """Create a pull request (requires gh CLI)"""
    try:
        subprocess.run([
            'gh', 'pr', 'create',
            '--title', title,
            '--body', body,
            '--base', target_branch,
            '--head', source_branch
        ], check=True)
        return True
    except subprocess.CalledProcessError:
        return False