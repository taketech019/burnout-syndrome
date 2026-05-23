@echo off
setlocal

set MODEL=E:\coding\host-koalpaca\ai-model\models\koalpaca-q4km.gguf
set PORT=8080
set NGL=%1
if "%NGL%"=="" set NGL=30
if "%KOALPACA_API_KEY%"=="" set KOALPACA_API_KEY=changeme-set-this

echo llama-server start (ngl=%NGL%, port=%PORT%)...
E:\llama.cpp-bin\llama-server.exe ^
    -m "%MODEL%" ^
    -c 2048 ^
    -ngl %NGL% ^
    --host 0.0.0.0 ^
    --port %PORT% ^
    --api-key %KOALPACA_API_KEY% ^
    -t 8 ^
    --batch-size 256 ^
    --log-disable

endlocal
