import asyncio
import logging
import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from dataclasses import dataclass

from ..agents_repository import AgentsRepository
# diff_util은 polling_manager에서 처리하므로 여기서는 사용하지 않음
from ..tools.knowledge_manager import Mem0Tool
from ..settings.crew_event_logger import CrewAIEventLogger

# 로거 설정
logger = logging.getLogger("agent_feedback_analyzer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class AgentFeedback:
    """에이전트 피드백 데이터 구조"""
    agent: str
    feedback: str

class AgentFeedbackAnalyzer:
    """
    DIFF 분석을 통해 에이전트별 개선점을 식별하고 피드백을 생성하는 클래스
    """
    
    def __init__(self):
        load_dotenv()
        self.agents_repository = AgentsRepository()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.knowledge_manager = Mem0Tool()
        self.event_logger = CrewAIEventLogger()
        
    async def generate_feedback_from_diff_result(
        self,
        diff_result: Dict[str, Any],
        original_content: str = None,
        todo_id: str = None,
        proc_inst_id: str = None,
        tenant_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        이미 분석된 DIFF 결과로부터 에이전트별 피드백 생성
        
        Args:
            diff_result: 이미 분석된 diff 결과
            todo_id: TODO 리스트 레코드 ID
            proc_inst_id: 프로세스 인스턴스 ID
            tenant_id: 테넌트 ID
            
        Returns:
            에이전트별 피드백 리스트
        """
        try:
            if not diff_result.get('unified_diff'):
                print("📝 변화없음 - 피드백 건너뜀")
                return []
            
            # 1. 에이전트 목록 조회
            agents = await self.agents_repository.get_all_agents(tenant_id)
            
            # 2. 이미 분석된 변경사항을 집계
            all_insertions = []
            all_deletions = []
            
            for comparison in diff_result.get('comparisons', []):
                changes = comparison.get('changes', {})
                all_insertions.extend(changes.get('insertions', []))
                all_deletions.extend(changes.get('deletions', []))
            
            aggregated_changes = {
                'insertions': all_insertions,
                'deletions': all_deletions,
                'has_changes': bool(all_insertions or all_deletions)
            }
            
            # 3. LLM을 통한 에이전트별 피드백 생성
            feedback_list = await self._generate_agent_feedback_with_llm(
                agents, aggregated_changes, diff_result, original_content
            )
            
            print(f"✅ 에이전트 피드백: {len(feedback_list)}개 생성")
            
            # 4. 피드백이 있으면 Mem0에 지식 적재
            if feedback_list:
                await self._store_feedback_to_memory(feedback_list)
                # 피드백 내용 출력
                for fb in feedback_list:
                    print(f"  • {fb.get('agent')}: {fb.get('feedback')}")
            
            return feedback_list
            
        except Exception as e:
            print(f"❌ 피드백 분석 오류: {e}")
            return []
    
    async def _generate_agent_feedback_with_llm(
        self, 
        agents: List[Dict[str, Any]], 
        changes: Dict[str, str], 
        diff_result: Dict[str, Any],
        original_content: str = None
    ) -> List[Dict[str, Any]]:
        """
        LLM을 사용하여 에이전트별 맞춤 피드백 생성
        """
        
        # 에이전트 정보
        agents_summary = agents
        
        # 변화 내용 (새로운 구조에 맞게 수정)
        deleted_content = '\n'.join(changes.get('deletions', []))
        added_content = '\n'.join(changes.get('insertions', []))
        
        # LLM 프롬프트 생성
        prompt = self._create_feedback_prompt(agents_summary, deleted_content, added_content, diff_result, original_content)
        
        # LLM 호출 (OpenAI 사용)
        feedback_result = await self._call_openai_for_feedback(prompt)
        
        return feedback_result
    
    def _create_feedback_prompt(
        self, 
        agents: List[Dict[str, Any]], 
        deleted_content: str, 
        added_content: str,
        diff_result: Dict[str, Any],
        original_content: str = None
    ) -> str:
        """
        의미적 분석을 통한 에이전트별 피드백 생성 프롬포트
        """
        
        # 에이전트 정보를 간단하게 정리
        agent_info = []
        for agent in agents:
            agent_info.append({
                "name": agent.get("name"),
                "role": agent.get("role"), 
                "goal": agent.get("goal"),
                "persona": agent.get("persona")
            })
        
        prompt = f"""당신은 문서 변경사항을 분석하여 에이전트별 맞춤 피드백을 생성하는 전문가입니다.

## 에이전트 목록
{json.dumps(agent_info, indent=2, ensure_ascii=False)}

## 원본 내용 (전체 맥락)
{original_content if original_content and original_content.strip() else "없음"}

## 원본에서 삭제된 내용
{deleted_content if deleted_content.strip() else "없음"}

## 새로 추가된 내용  
{added_content if added_content.strip() else "없음"}

## 분석 과정
1. **변경의 의도**: 삭제된 내용과 추가된 내용을 비교하여 어떤 개선이 이루어졌는지 파악
2. **에이전트 매칭**: 변경사항과 각 에이전트의 역할(role)과 목표(goal)를 비교하여 관련성 판단
3. **피드백 생성**: 관련성이 높은 에이전트에게만 구체적이고 실행 가능한 피드백 제공

## 매칭 기준
- **리서처/분석가**: 정보 정확성, 데이터 분석 관련 변경
- **작성자/writer**: 문체, 구조, 가독성 관련 변경  
- **검토자/reviewer**: 품질 개선, 오류 수정 관련 변경
- **기획자/planner**: 구성, 흐름, 전략 관련 변경
- **전문가/expert**: 전문 지식, 기술적 내용 관련 변경

## 출력 형식
관련성이 있는 에이전트에게만 피드백을 제공하세요. 관련성이 낮으면 피드백하지 마세요.

```json
[
  {{"agent": "에이전트명", "feedback": "구체적 개선점 (1-2줄)"}}
]
```

**중요**: 형식 변경(공백, 마크다운)은 무시하고 실제 내용 변화에만 집중하세요."""
        
        return prompt
    
    async def _call_openai_for_feedback(self, prompt: str) -> List[Dict[str, Any]]:
        """
        OpenAI API를 호출하여 피드백 생성
        """
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=self.openai_api_key)
            
            response = await client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 AI 에이전트 성과 분석 전문가입니다. 문서 변화를 분석하여 각 에이전트에게 구체적이고 건설적인 피드백을 제공합니다."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # JSON 추출 (```json 블록이 있는 경우)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            # JSON 파싱
            feedback_list = json.loads(content)
            
            return feedback_list
            
        except Exception as e:
            print(f"❌ OpenAI API 오류: {e}")
            return []
    
    async def _store_feedback_to_memory(self, feedback_list: List[Dict[str, Any]]):
        """
        생성된 피드백을 Mem0에 지식으로 적재
        """
        try:            
            for feedback in feedback_list:
                agent_name = feedback.get('agent')
                feedback_content = feedback.get('feedback')
                
                if agent_name and feedback_content:
                    # 피드백을 지식 형태로 포맷팅
                    knowledge_content = f"[피드백] {feedback_content}"
                    
                    # Mem0에 저장
                    self.knowledge_manager._run(
                        agent_name=agent_name,
                        mode="add",
                        content=knowledge_content
                    )
            
            print(f"🧠 Mem0 저장완료: {len(feedback_list)}개")
            
        except Exception as e:
            print(f"❌ Mem0 저장오류: {e}")
    