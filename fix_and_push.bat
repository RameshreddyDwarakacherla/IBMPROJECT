@echo off
echo ========================================
echo   Fix Push Error and Retry
echo ========================================
echo.

echo Step 1: Increase Git buffer size...
git config --global http.postBuffer 524288000
echo Done!
echo.

echo Step 2: Increase timeout to 10 minutes...
git config --global http.timeout 600
echo Done!
echo.

echo Step 3: Retry push...
echo This may take a few minutes for large files...
echo.
git push -u origin main
echo.

if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo   SUCCESS! Push completed!
    echo ========================================
    echo.
    echo Visit: https://github.com/RameshreddyDwarakacherla/IBMPROJECT
) else (
    echo ========================================
    echo   Push failed again
    echo ========================================
    echo.
    echo The model files are too large for GitHub.
    echo.
    echo RECOMMENDED SOLUTION:
    echo 1. Install Git LFS: https://git-lfs.github.com/
    echo 2. Run: git lfs install
    echo 3. Run: git lfs track "*.sav"
    echo 4. Run: git add .gitattributes
    echo 5. Run: git commit -m "Track models with Git LFS"
    echo 6. Run: git push
    echo.
    echo OR upload models to cloud storage and link in README
)

echo.
pause
