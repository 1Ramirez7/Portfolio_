# BYUI DS Portfolio Template
## Building a Quarto Portfolo



# To install the required dependencies:
pip install -r requirements.txt

- quarto assumes libraries are pre-installed on the system
- update requirements.txt and remove any libraries that are no longer needed so they are not unnecessarily installed with new clones. 



# Git Commands:

## MISTAKES WERE MADE - this terminal code will either delete all changes since last pull or do a force pull to overwrite local changes.

git restore --source=HEAD --staged --worktree .
git clean -fd # Remove any new, untracked files (be cautious with this step)

Use the following code to check the status of the repository to confirm up to date with last pull.
git status


Use the following commands to reset everything to the remote branch: 

git fetch origin
git reset --hard origin/<branch-name>  # Replace <branch-name> with the branch you’re working on (e.g., main or master).
# git reset doesn't delete created files!!!!


# Useful Git Commands

A reference guide for commonly used and helpful Git commands.

---

## 1. Viewing Repository Status and History

- `git status`  Displays the current state of your repository, including untracked, modified, or staged files.

- Shows the commit history. Use flags for a concise view:
  git log --oneline --graph --decorate


- `git diff`  
  Shows differences between files:
  - Compare working directory to the last commit:
    ```bash
    git diff
    ```
  - Compare staged changes to the last commit:
    ```bash
    git diff --staged
    ```

- `git blame <file>`  
  Displays who made changes to each line in a file.

- `git show <commit-hash>`  
  Displays details of a specific commit, including changes made.

---

## 2. Working with Branches

- `git branch`  
  Lists all branches. Use flags for more details:
  ```bash
  git branch -a  # List all branches (local and remote)
  ```

- `git switch <branch>`  
  Switches to an existing branch. Use `-c` to create and switch to a new branch:
  ```bash
  git switch -c new-branch
  ```

- `git merge <branch>`  
  Merges the specified branch into the current branch.

---

## 3. Undoing Changes

- `git restore <file>`  
  Restores a file in the working directory to the last committed state.

- `git reset HEAD <file>`  
  Unstages a file but keeps its changes in the working directory.

- `git reset --soft HEAD~1`  
  Undo the last commit but keep changes staged.

- `git reset --hard HEAD~1`  
  Completely undo the last commit and remove all changes.
  the use to forcefully push. this basically removes the last push to github.
  git push --force 

---

## 4. Fetching and Updating

- `git fetch`  
  Downloads the latest changes from the remote but does not merge them into your branch.

- `git pull`  
  Fetches and merges changes from the remote into your branch.

- `git push`  
  Pushes your commits to the remote repository.

- `git push --force`  
  Forcefully pushes changes, overwriting the remote branch (use with caution!).

---

## 5. Managing Files

- `git mv <old-file> <new-file>`  # Moves or renames a file.

---

## 6. Advanced Commands (research)

- `git stash` # Saves your uncommitted changes to a temporary stash, allowing you to work on something else:
- `git cherry-pick <commit-hash>`  
- `git rebase <branch>`  
- `git tag <tag-name>`  


---

## 7. Useful Info and Stats

- `git remote -v`  
  Lists the remote URLs for your repository.

- `git shortlog -s -n`  
  Displays commit contributions grouped by author.

- `git count-objects -v`  
  Provides statistics about the repository size and objects.

- `git branch --merged`  
  Lists branches already merged into the current branch.

- `git log --stat`  
  Shows commit history with a summary of changes made in each commit.

---

## 8. Configurations and Aliases

- `git config --global user.name "Your Name"`  
  Sets your username globally.

- `git config --global user.email "your_email@example.com"`  
  Sets your email globally.

- **Create Aliases for Common Commands**:

  git config --global alias.co checkout
  git config --global alias.br branch
  git config --global alias.st status
  git config --global alias.lg "log --oneline --graph --decorate"

---

### Recommendations:
1. **Learn `git log` with flags** to better understand history.
2. Use `git stash` when experimenting.
3. Be cautious with `git push --force` and always confirm you're on the correct branch.
