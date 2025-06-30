import os
from typing import Optional, List, Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from dotenv import load_dotenv
from mem0 import Memory

# .env 파일 로드
load_dotenv()

# Supabase 연결 정보 environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("DB 연결 환경 변수 중 일부가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# PostgreSQL connection string
connection_string = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

class KnowledgeQuerySchema(BaseModel):
    user_id: str = Field(..., description="에이전트 식별자(UUID)")
    query: str = Field(..., description="검색할 지식 쿼리")

class Mem0Tool(BaseTool):
    """Supabase 기반 mem0 지식 검색 도구 (읽기 전용, 모든 결과 반환)"""
    name: str = "mem0"
    description: str = "Supabase에 저장된 지식을 검색하여 전체 결과를 반환합니다."
    args_schema: Type[KnowledgeQuerySchema] = KnowledgeQuerySchema

    # PrivateAttr: Pydantic 필드가 아닌 런타임 전용 속성
    _memory: Memory = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = {
            "vector_store": {
                "provider": "supabase",
                "config": {
                    "connection_string": connection_string,
                    "collection_name": "memories",
                    "index_method": "hnsw",
                    "index_measure": "cosine_distance"
                }
            }
        }
        # Supabase 벡터 스토어로부터 Memory 인스턴스 생성
        self._memory = Memory.from_config(config_dict=config)
        print("🚀 Mem0Tool initialized with Supabase backend")

    def _run(self, user_id: str, query: str) -> str:
        """지식을 검색하고 사용자에게 모든 결과를 반환하며, 검색 현황을 출력합니다."""
        print(f"▶▶ [Debug] Mem0Tool._run called with agent_id={user_id!r}, query={query!r}")

        if not query:
            print("⚠️ Empty query received in Mem0Tool._run")
            return "검색할 쿼리를 입력해주세요."
        try:
            results = self._memory.search(query, user_id=user_id)
            hits = results.get("results", [])

            print(f"🔍 [Mem0Tool] agent_id={user_id}, query='{query}', returning all {len(hits)} items")
            for idx, hit in enumerate(hits, start=1):
                score = hit.get("score", 0)
                snippet = hit.get("memory", "")[0:50].replace("\n", " ")
                print(f"   ▶ Hit {idx}: score={score:.4f}, snippet='{snippet}...'")

            if not hits:
                return f"'{query}'에 대한 지식이 없습니다."

            # 결과 포맷팅
            items: List[str] = []
            for idx, hit in enumerate(hits, start=1):
                mem_text = hit.get("memory", "")
                score = hit.get("score", 0)
                items.append(f"지식 {idx} (관련도: {score:.2f})\n{mem_text}")

            return "\n\n".join(items)
        except Exception as e:
            print(f"❌ Mem0Tool 검색 중 오류: {e}")
            return f"지식 검색 오류: {e}"
