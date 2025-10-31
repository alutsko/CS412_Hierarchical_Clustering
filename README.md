# CS412 Course Workspace

This repository contains student code and materials for CS412 (Intro to Data Mining).

Structure (high level):
- `Week_2_Apriori_mining/` — Apriori implementation and tests
- `Week_4_Cont_seq_mining/` — Contiguous sequential pattern miner
- `Week_7_k-means_clustering/` — K-means examples
- `Week_8_Hierarchical_clustering/` — Hierarchical clustering template

How to use this repo locally

1. Initialize git (if not already):
   git init
2. Stage and commit changes:
   git add -A
   git commit -m "Initial commit"

Create a GitHub remote and push

Option A — using GitHub CLI (recommended):
   gh repo create <owner>/<repo-name> --public --source=. --remote=origin --push

Option B — using the web UI:
   - Create a new repo on github.com
   - Add the shown remote URL and push:
       git remote add origin <remote-url>
       git branch -M main
       git push -u origin main

Notes
- Large data files should be kept out of the repo; put them in `data/` and add them to `.gitignore`.
- If you want, I can create the GitHub repo for you; your machine doesn't have the GitHub CLI installed. If you want me to, please provide a GitHub username/org and repo name, and whether it should be public or private.
