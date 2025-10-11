:: Installation script :::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::
::                           ALPR (YOLOv8)
::
:: This script is only called from ..\..\CodeProject.AI-Server\src\setup.bat in
:: Dev setup, or ..\..\src\setup.bat in production

@if "%1" NEQ "install" (
    echo This script is only called from ..\..\CodeProject.AI-Server\src\setup.bat
    @pause
    @goto:eof
)

:: Create directories if they don't exist
if not exist "models" mkdir "models"
if not exist "test" mkdir "test"

REM YOLO11 ONNX models - Download from Hugging Face
REM Note: Hugging Face raw file URL format is different from tree view
set baseUrl=https://huggingface.co/MikeLud/ALPR-YOLO11-ONNX/resolve/main

REM List of files to download
set "files=char_classifier.json char_classifier.onnx char_detector.json char_detector.onnx plate_detector.json plate_detector.onnx state_classifier.json state_classifier.onnx vehicle_segment.onnx"

for %%f in (%files%) do (
    set fileToGet=%%f
    
    if not exist "!moduleDirPath!\models\!fileToGet!" (
        set destination=!downloadDirPath!\!modulesDir!\!moduleDirName!\!fileToGet!

        if not exist "!downloadDirPath!\!modulesDir!\!moduleDirName!" mkdir "!downloadDirPath!\!modulesDir!\!moduleDirName!"
        if not exist "!moduleDirPath!\models" mkdir "!moduleDirPath!\models"

        call "!utilsScript!" WriteLine "Downloading !fileToGet!..." "!color_info!"

        powershell -command "Start-BitsTransfer -Source '!baseUrl!/!fileToGet!' -Destination '!destination!'"
        if errorlevel 1 (
            powershell -Command "Get-BitsTransfer | Remove-BitsTransfer"
            powershell -command "Start-BitsTransfer -Source '!baseUrl!/!fileToGet!' -Destination '!destination!'"
        )
        if errorlevel 1 (
            powershell -Command "Invoke-WebRequest '!baseUrl!/!fileToGet!' -OutFile '!destination!'"
            if errorlevel 1 (
                call "!utilsScript!" WriteLine "Download failed for !fileToGet!. Sorry." "!color_error!"
                set moduleInstallErrors=Unable to download !fileToGet!
            )
        )

        if exist "!destination!" (
            call "!utilsScript!" WriteLine "Moving !fileToGet! into the models folder." "!color_info!"
            move "!destination!" "!moduleDirPath!\models\" > nul
            call "!utilsScript!" WriteLine "!fileToGet! downloaded successfully." "!color_success!"
        ) else (
            call "!utilsScript!" WriteLine "Download failed for !fileToGet!. Sad face." "!color_warn!"
        )
    ) else (
        call "!utilsScript!" WriteLine "!fileToGet! already downloaded." "!color_success!"
    )
)