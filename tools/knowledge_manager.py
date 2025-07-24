import os
import logging
import traceback
from typing import Optional, List, Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from dotenv import load_dotenv
from mem0 import Memory
import requests

# ============================================================================
# 설정 및 초기화
# ============================================================================

load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)

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
            
            # hybrid 필터링 적용: threshold=0.6, 최소 5개 보장
            THRESHOLD = 0.6
            MIN_RESULTS = 5
            # 1) 유사도 내림차순 정렬
            hits_sorted = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
            # 2) Threshold 이상 항목 필터
            filtered_hits = [h for h in hits_sorted if h.get("score", 0) >= THRESHOLD]
            # 3) 최소 개수 보장
            if len(filtered_hits) < MIN_RESULTS:
                filtered_hits = hits_sorted[:MIN_RESULTS]
            hits = filtered_hits

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

# ============================================================================
# 사내 문서 검색 (memento) 도구
# ============================================================================

class MementoQuerySchema(BaseModel):
    query: str = Field(..., description="검색 키워드 또는 질문")
    tenant_id: str = Field("localhost", description="테넌트 식별자 (기본값 localhost)")

class MementoTool(BaseTool):
    """사내 문서 검색을 수행하는 도구"""
    name: str = "memento"
    description: str = "사내 문서 검색을 위한 도구"
    args_schema: Type[MementoQuerySchema] = MementoQuerySchema

    def _run(self, query: str, tenant_id: str = "localhost") -> str:
        try:
            logger.info(f"🔍 Memento 문서 검색 시작: tenant_id='{tenant_id}', query='{query}'")

            response = requests.post(
                # "http://memento.process-gpt.io/retrieve",
                "http://localhost:8005/retrieve",
                json={"query": query, "options": {"tenant_id": tenant_id}}
            )
            if response.status_code != 200:
                return f"API 오류: {response.status_code}"
            data = response.json()
            if not data.get("response"):
                return f"테넌트 '{tenant_id}'에서 '{query}' 검색 결과가 없습니다."
            results = []
            # 검색 결과 로그 출력
            docs = data.get("response", [])
            logger.info(f"🔍 Memento 검색 결과 개수: {len(docs)}")
            for doc in docs:
                fname = doc.get('metadata', {}).get('file_name', 'unknown')
                idx = doc.get('metadata', {}).get('chunk_index', 'unknown')
                content = doc.get('page_content', '')
                logger.info(f"📄 문서: {fname}, 청크: {idx}, 내용: {content[:100]}...")
                results.append(f"📄 파일: {fname} (청크 #{idx})\n내용: {content}\n---")
            return f"테넌트 '{tenant_id}'에서 '{query}' 검색 결과:\n\n" + "\n".join(results)
        except Exception as e:
            return f"검색 중 오류 발생: {e}"
