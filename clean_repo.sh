#!/bin/bash

REPO_URL="https://github.com/LeDuong2408/HR-Scan-CV.git"
SECRET="AIzaSyD9DrFLD55EZfEIfXbFy963Tx0IXumfZlY"

echo "$SECRET==>REMOVED" > replacements.txt

git filter-repo --replace-text replacements.txt --force

git remote add origin "$REPO_URL"

git push origin --force --all

del replacements.txt

echo "Done! Secret removed from git history."