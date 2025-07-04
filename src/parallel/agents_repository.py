import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
load_dotenv()

class AgentsRepository:
    """Supabase agents 테이블에서 데이터 조회만"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        print("[AgentsRepo] Supabase 연결 완료")
    
    def _get_fallback_agents(self) -> List[Dict[str, Any]]:
        """기본 6개 에이전트 반환"""
        return [
            {
                "id": "fallback_1",
                "name": "리서처",
                "role": "researcher", 
                "goal": "정보를 조사하고 분석합니다",
                "persona": "꼼꼼하고 분석적인 연구원",
                "description": "다양한 소스에서 정보를 수집하고 분석하는 전문가",
                "tools": "mem0",
                "profile": "정보수집 및 분석 전문가"
            },
            {
                "id": "fallback_2", 
                "name": "분석가",
                "role": "analyst",
                "goal": "데이터를 분석하고 인사이트를 제공합니다",
                "persona": "논리적이고 체계적인 분석 전문가",
                "description": "복잡한 정보를 분석하여 명확한 결론을 도출하는 전문가",
                "tools": "mem0",
                "profile": "데이터 분석 및 인사이트 전문가"
            },
            {
                "id": "fallback_3",
                "name": "작성자", 
                "role": "writer",
                "goal": "명확하고 이해하기 쉬운 글을 작성합니다",
                "persona": "창의적이고 소통에 능한 작가",
                "description": "복잡한 내용을 쉽고 명확하게 전달하는 글쓰기 전문가",
                "tools": "mem0",
                "profile": "콘텐츠 작성 및 편집 전문가"
            },
            {
                "id": "fallback_4",
                "name": "검토자",
                "role": "reviewer", 
                "goal": "내용을 검토하고 품질을 개선합니다",
                "persona": "세심하고 비판적 사고를 하는 검토자",
                "description": "작성된 내용의 정확성과 품질을 검증하는 전문가",
                "tools": "mem0",
                "profile": "품질 검토 및 개선 전문가"
            },
            {
                "id": "fallback_5",
                "name": "기획자",
                "role": "planner",
                "goal": "전략을 수립하고 계획을 세웁니다", 
                "persona": "체계적이고 전략적 사고를 하는 기획자",
                "description": "목표 달성을 위한 체계적인 계획을 수립하는 전문가",
                "tools": "mem0",
                "profile": "전략 수립 및 기획 전문가"
            },
            {
                "id": "fallback_6",
                "name": "전문가",
                "role": "expert",
                "goal": "전문 지식을 제공하고 자문합니다",
                "persona": "경험이 풍부하고 지식이 해박한 전문가", 
                "description": "해당 분야의 깊은 전문 지식을 바탕으로 조언하는 전문가",
                "tools": "mem0",
                "profile": "분야별 전문 지식 자문가"
            }
        ]

    async def get_all_agents(self, tenant_id: str = "default") -> List[Dict[str, Any]]:
        """agents 테이블에서 5개 필드(name, role, goal, persona, description)가 모두 비어있지 않은 데이터만 조회"""
        try:
            # 에이전트 조회
            response = (self.client.table("users").select("*" )
                       .eq("is_agent", True)
                       .not_.is_("name", "null").not_.is_("role", "null")
                       .not_.is_("goal", "null").not_.is_("persona", "null")
                       .neq("name", "").neq("role", "").neq("goal", "").neq("persona", "")
                       .execute())
            agents = response.data or []
            # fallback 처리
            if not agents:
                print("[AgentsRepo] 기본 에이전트 사용")
                return self._get_fallback_agents()
            # tools 기본값 설정
            for ag in agents:
                if not ag.get('tools'):
                    ag['tools'] = 'mem0'
            print(f"[AgentsRepo] {len(agents)} 에이전트 조회 완료")
            return agents
        except Exception as e:
            print(f"[AgentsRepo] 조회 오류: {e} - 기본 에이전트 사용")
            fallback = self._get_fallback_agents()
            return fallback
