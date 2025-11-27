@echo off
echo ========================================
echo   GitHub Push Script
echo ========================================
echo.

echo Step 1: Checking Git status...
git status
echo.

echo Step 2: Adding all files...
git add .
echo.

echo Step 3: Committing changes...
git commit -m "Clean up project - Remove redundant docs, add comprehensive README and documentation"
echo.

echo Step 4: Checking remote...
git remote -v
echo.

echo Step 5: Pushing to GitHub...
echo.
echo If remote doesn't exist, run this first:
echo git remote add origin https://github.com/RameshreddyDwarakacherla/IBMPROJECT.git
echo.

pause
echo.
echo Pushing now...
git push -u origin main
echo.

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Push failed. Trying 'master' branch...
    git push -u origin master
)

echo.
echo ========================================
echo   Push Complete!
echo ========================================
echo.
echo Visit: https://github.com/RameshreddyDwarakacherla/IBMPROJECT
echo.
pause
