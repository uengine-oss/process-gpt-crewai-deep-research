import asyncio
import logging
import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from dataclasses import dataclass

from .agents_repository import AgentsRepository
from .diff_util import compare_report_changes, extract_changes
from .knowledge_manager import Mem0Tool
from .event_logging.crew_event_logger import CrewAIEventLogger

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
        
    async def analyze_diff_and_generate_feedback(
        self, 
        draft_content: str, 
        output_content: str,
        todo_id: str = None,
        proc_inst_id: str = None,
        tenant_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        DIFF 분석 후 에이전트별 피드백 생성
        
        Args:
            draft_content: Draft 내용
            output_content: Output 내용
            todo_id: TODO 리스트 레코드 ID
            proc_inst_id: 프로세스 인스턴스 ID
            tenant_id: 테넌트 ID
            
        Returns:
            에이전트별 피드백 리스트
        """
        try:
            # 1. DIFF 분석
            diff_result = compare_report_changes(draft_content, output_content)
            
            if not diff_result.get('unified_diff'):
                print("변화가 없어 피드백 분석을 건너뜁니다.")
                return []
            
            # 2. 에이전트 목록 조회
            agents = await self.agents_repository.get_all_agents(tenant_id)
            
            # 3. 변화 분석
            changes = extract_changes(
                diff_result.get('draft_content', ''), 
                diff_result.get('output_content', '')
            )
            
            # 4. 피드백 생성 전 이벤트 기록 (한 번만, 빈 데이터)
            self.event_logger.emit_feedback_started_event(
                feedback_json={},
                todo_id=todo_id,
                proc_inst_id=proc_inst_id
            )
            
            # 5. LLM을 통한 에이전트별 피드백 생성
            feedback_list = await self._generate_agent_feedback_with_llm(
                agents, changes, diff_result
            )
            
            logger.info(f"✅ {len(feedback_list)}개의 에이전트 피드백 생성 완료")
            
            # 6. 피드백 생성 후 이벤트 기록 (한 번만, 전체 피드백 리스트 전달)
            self.event_logger.emit_feedback_completed_event(
                feedback_json={"feedbacks": feedback_list},
                todo_id=todo_id,
                proc_inst_id=proc_inst_id
            )
            
            # 7. 피드백이 있으면 Mem0에 지식 적재
            if feedback_list:
                await self._store_feedback_to_memory(feedback_list)
            
            return feedback_list
            
        except Exception as e:
            logger.error(f"피드백 분석 중 오류 발생: {e}")
            return []
    
    async def _generate_agent_feedback_with_llm(
        self, 
        agents: List[Dict[str, Any]], 
        changes: Dict[str, str], 
        diff_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        LLM을 사용하여 에이전트별 맞춤 피드백 생성
        """
        
        # 에이전트 정보
        agents_summary = agents
        
        # 변화 내용
        deleted_content = changes['original_changes']
        added_content = changes['modified_changes']
        
        # LLM 프롬프트 생성
        prompt = self._create_feedback_prompt(agents_summary, deleted_content, added_content, diff_result)
        
        # LLM 호출 (OpenAI 사용)
        feedback_result = await self._call_openai_for_feedback(prompt)
        
        return feedback_result
    
    def _create_feedback_prompt(
        self, 
        agents: List[Dict[str, Any]], 
        deleted_content: str, 
        added_content: str,
        diff_result: Dict[str, Any]
    ) -> str:
        """
        에이전트 피드백 생성을 위한 간단한 LLM 프롬프트 작성
        """
        
        prompt = f"""
# DIFF 분석을 통한 에이전트 피드백 생성

## 에이전트 목록
{json.dumps(agents, indent=2, ensure_ascii=False)}

## 변화 내용 분석
### 삭제된 내용:
{deleted_content if deleted_content.strip() else "없음"}

### 추가된 내용:  
{added_content if added_content.strip() else "없음"}

## 분석 목표
**추가된 내용을 보고 다음을 파악하세요:**
1. **어떤 내용이 새로 추가되었는가?**
2. **어떤 부분을 강조하려고 하는가?**

## 피드백 생성 원칙
- 추가된 내용의 의도와 강조점을 구체적으로 파악
- 해당 내용과 관련있는 에이전트에게만 피드백 제공
- 간단하고 명확하게 2-3줄로 작성

예시: "마이그레이션 과정을 단계별로 더 디테일하게 작성하는 방향으로 개선되었네요. 앞으로도 복잡한 프로세스는 2단계로 나눠서 구체적으로 설명해주세요."

## 출력 형식
```json
[
  {{
    "agent": "에이전트_이름", 
    "feedback": "구체적인 피드백 (2-3줄)"
  }}
]
```

**중요**: 단순한 형식 변경(마크다운 문법, 공백 등)은 무시하고, 실제 내용 추가/강조에만 집중하세요.
"""
        
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
            logger.error(f"OpenAI API 호출 중 오류: {e}")
            return []
    
    async def _store_feedback_to_memory(self, feedback_list: List[Dict[str, Any]]):
        """
        생성된 피드백을 Mem0에 지식으로 적재
        """
        try:
            logger.info(f"🧠 {len(feedback_list)}개의 피드백을 Mem0에 저장 중...")
            
            for feedback in feedback_list:
                agent_name = feedback.get('agent')
                feedback_content = feedback.get('feedback')
                
                if agent_name and feedback_content:
                    # 피드백을 지식 형태로 포맷팅
                    knowledge_content = f"[피드백] {feedback_content}"
                    
                    # Mem0에 저장
                    result = self.knowledge_manager._run(
                        agent_name=agent_name,
                        mode="add",
                        content=knowledge_content
                    )
                    
                    logger.info(f"💾 {agent_name}에게 피드백 저장: {result}")
            
            logger.info("✅ 모든 피드백이 Mem0에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"Mem0 지식 적재 중 오류: {e}")
    