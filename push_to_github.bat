@echo off
echo ========================================================
echo  SafeFall AI - Pushing to Personal GitHub Repository
echo ========================================================
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo If prompted by Git Credential Manager, click 'Sign in with your browser'.
    echo.
) else (
    echo.
    echo ========================================================
    echo  SUCCESS! Repository pushed to:
    echo  https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA
    echo ========================================================
)
pause
