import os
from typing import Optional, List, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from dotenv import load_dotenv
from mem0 import MemoryClient  # Mem0 필수 의존성

# .env 파일 로드
load_dotenv()

class KnowledgeQuerySchema(BaseModel):
    agent_name: str = Field(..., description="에이전트 이름 (네임스페이스)")
    mode: str       = Field(..., description="'add' 또는 'retrieve'")
    content: Optional[str] = Field(None, description="추가할 지식 (mode='add')")
    query:   Optional[str] = Field(None, description="검색 쿼리 (mode='retrieve')")

class Mem0Tool(BaseTool):
    """Mem0 기반 지식 관리 도구"""
    name: str = "mem0"
    description: str = "Mem0 클라우드 지식 저장 및 검색 기능"
    args_schema: type = KnowledgeQuerySchema
    mem0_client: Optional[MemoryClient] = Field(None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv('MEM_ZERO_API_KEY')
        self.mem0_client = MemoryClient(api_key=api_key) if api_key else None

    def _run(self, agent_name: str, mode: str, content: Optional[str] = None, query: Optional[str] = None) -> str:
        if self.mem0_client is None:
            return "Mem0 클라이언트 미연결"
        if mode == 'add':
            return self._add_knowledge(agent_name, content)
        if mode == 'retrieve':
            return self._retrieve_knowledge(agent_name, query)
        return "mode는 'add' 또는 'retrieve'만 지원합니다."

    def _add_knowledge(self, agent_name: str, content: Optional[str]) -> str:
        if not content:
            return "추가할 지식 내용이 필요합니다."
        try:
            self.mem0_client.add(
                [{"role": "user", "content": content}],
                agent_id=agent_name
            )
            return "지식 저장 성공"
        except Exception as e:
            return f"지식 저장 실패: {e}"

    def _retrieve_knowledge(self, agent_name: str, query: Optional[str]) -> str:
        if not query:
            query = agent_name
        try:
            results = self.mem0_client.search(query, agent_id=agent_name)
            if not results:
                return f"'{query}'에 대한 지식이 없습니다."
            items: List[str] = []
            for idx, res in enumerate(results[:3], start=1):
                mem = res.get('memory', '')
                score = res.get('score', 0)
                if mem:
                    items.append(f"지식 {idx} (관련도: {score:.2f})\n{mem}")
            return "\n\n".join(items)
        except Exception as e:
            return f"지식 검색 오류: {e}"
