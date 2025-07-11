# -*- coding: utf-8 -*-
import sys
import io
import warnings
import builtins
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 전역 print 함수 오버라이드 (가장 먼저 적용)
_orig_print = builtins.print
def print(*args, **kwargs):
    # 기본적으로 flush=True 적용
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    _orig_print(*args, **kwargs)
builtins.print = print


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# 특정 경고 메시지 필터링
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qdrant_client.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets.*")

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["PYTHONIOENCODING"] = "utf-8"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import add_routes_to_app
from src.parallel.polling_manager import start_todolist_polling, initialize_connections

# 백그라운드 태스크 관리를 위한 lifespan 함수
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    initialize_connections()
    # 통합 polling 태스크 시작
    asyncio.create_task(start_todolist_polling(interval=7))
    yield
    # 종료 시 실행 (필요한 경우 cleanup 로직 추가)

app = FastAPI(
    title="AgentMonitoring Server",
    version="1.0",
    description="Agent Monitoring API Server",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 라우터 추가
add_routes_to_app(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8000)),
        ws="none"
    )