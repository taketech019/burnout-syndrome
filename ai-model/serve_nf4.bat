@echo off
REM KoAlpaca NF4 + LoRA attach 서버 기동. workers=1 필수 (12.8B 메모리 한계).
REM 전제: .venv-nf4 활성화된 셸에서 실행, 또는 아래 venv 활성화 라인 사용.

cd /d %~dp0

REM venv 자동 활성화 (이미 활성화돼 있으면 무해)
if exist ".venv-nf4\Scripts\activate.bat" call ".venv-nf4\Scripts\activate.bat"

set KOALPACA_BASE=EleutherAI/polyglot-ko-12.8b
set KOALPACA_ADAPTER=%~dp0koalpaca_save
set KOALPACA_COMPUTE_DTYPE=bfloat16
set KOALPACA_GPU_MEM=7GiB
set KOALPACA_CPU_MEM=28GiB

uvicorn nf4_server:app --host 127.0.0.1 --port 8765 --workers 1
