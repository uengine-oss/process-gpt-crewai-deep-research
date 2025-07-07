import os
import logging
import traceback
from typing import Optional, List, Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from dotenv import load_dotenv
from mem0 import Memory

# ============================================================================
# 설정 및 초기화
# ============================================================================

load_dotenv()

# 로거 설정
logger = logging.getLogger("knowledge_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 데이터베이스 연결 정보
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("❌ DB 연결 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def _handle_error(operation: str, error: Exception) -> str:
    """통합 에러 처리"""
    error_msg = f"❌ [{operation}] 오류 발생: {str(error)}"
    logger.error(error_msg)
    logger.error(f"상세 정보: {traceback.format_exc()}")
    return f"{operation} 실패: {error}"

# ============================================================================
# 스키마 정의
# ============================================================================

class KnowledgeQuerySchema(BaseModel):
    user_id: str = Field(..., description="에이전트 식별자(UUID)")
    query: str = Field(..., description="검색할 지식 쿼리")

# ============================================================================
# 지식 검색 도구
# ============================================================================

class Mem0Tool(BaseTool):
    """Supabase 기반 mem0 지식 검색 도구"""
    name: str = "mem0"
    description: str = "Supabase에 저장된 지식을 검색하여 전체 결과를 반환합니다."
    args_schema: Type[KnowledgeQuerySchema] = KnowledgeQuerySchema
    
    _memory: Memory = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._memory = self._initialize_memory()
        logger.info("✅ Mem0Tool 초기화 완료")

    def _initialize_memory(self) -> Memory:
        """Memory 인스턴스 초기화"""
        config = {
            "vector_store": {
                "provider": "supabase",
                "config": {
                    "connection_string": CONNECTION_STRING,
                    "collection_name": "memories",
                    "index_method": "hnsw",
                    "index_measure": "cosine_distance"
                }
            }
        }
        return Memory.from_config(config_dict=config)

    def _run(self, user_id: str, query: str) -> str:
        """지식 검색 및 결과 반환"""
        if not query:
            return "검색할 쿼리를 입력해주세요."
        
        try:
            logger.info(f"🔍 지식 검색 시작: user_id={user_id}, query='{query}'")
            
            # 검색 실행
            results = self._memory.search(query, user_id=user_id)
            hits = results.get("results", [])
            
            logger.info(f"📋 검색 결과: {len(hits)}개 항목 발견")
            
            # 결과 처리
            if not hits:
                return f"'{query}'에 대한 지식이 없습니다."
            
            return self._format_results(hits)
            
        except Exception as e:
            return _handle_error("지식검색", e)

    def _format_results(self, hits: List[dict]) -> str:
        """검색 결과 포맷팅"""
        items = []
        for idx, hit in enumerate(hits, start=1):
            memory_text = hit.get("memory", "")
            score = hit.get("score", 0)
            items.append(f"지식 {idx} (관련도: {score:.2f})\n{memory_text}")
        
        return "\n\n".join(items)
