from crewai import Agent, Crew, Process, Task
from typing import Dict, Any, Optional
from ...tools.safe_tool_loader import SafeToolLoader
from ...context_manager import set_crew_context, reset_crew_context

# Agent에 profile 필드를 허용하는 서브클래스 정의
class AgentWithProfile(Agent):
    profile: Optional[str] = None
    user_id: Optional[str] = None

class DynamicReportCrew:
    """
    AgentMatchingCrew 결과물에서 섹션별 {toc, agent, task} 정보를 받아서
    동적으로 Agent와 Task를 생성해서 Crew를 만드는 클래스 (도구 연결 버전)
    """
    
    def __init__(self, section_data: Dict[str, Any], topic: str, previous_context: Optional[Dict[str, Any]] = None):
        """
        인자:
            section_data: 섹션별 {toc, agent, task} 데이터
            topic: 주제
            previous_context: 이전 완료 작업 컨텍스트
        """
        # 이전 컨텍스트 저장
        self.previous_context = previous_context
        # 기본 설정
        self.topic = topic
        self.toc_info = section_data.get("toc", {})
        self.agent_config = section_data.get("agent", {})
        self.task_config = section_data.get("task", {})
        
        # SafeToolLoader 다시 생성 (실제 도구 로딩용)
        self.safe_tool_loader = SafeToolLoader()
        
        self.section_title = self.toc_info.get("title", "Unknown Section")
        
        print(f"   └─ 매칭된 에이전트: {self.agent_config.get('name', 'Unknown')} ({self.agent_config.get('role', 'Unknown')})")
        
        # tool_names에서 실제 도구 객체 생성
        self.tool_names = self.agent_config.get('tool_names', [])
        self.actual_tools = self.safe_tool_loader.create_tools_from_names(self.tool_names)
        
        print(f"   └─ 실제 생성된 도구: {len(self.actual_tools)}개")
    
    def create_dynamic_agent(self) -> Agent:
        """동적으로 Agent 생성 (실제 도구 포함)"""
        
        # 기본 Agent 정보
        agent_role = self.agent_config.get("role", "Unknown Role")
        agent_goal = self.agent_config.get("goal", "Unknown Goal")
        agent_backstory = self.agent_config.get("persona", "Unknown Background")
        llm_model = self.agent_config.get("model", "gpt-4.1")

        print(f"   └─ 실제 할당된 도구: {len(self.actual_tools)}개")
        
        # Agent 생성 (실제 도구 할당)
        agent = AgentWithProfile(
            role=agent_role,
            goal=agent_goal,
            backstory=agent_backstory,
            llm=llm_model,
            tools=self.actual_tools,  # 실제 Tool 객체들 할당
            verbose=True,
            cache=True
        )
        
        # 에이전트 프로필 설정 (section_data에서 전달된 agent_profile 사용)
        agent.profile = self.agent_config.get('agent_profile', '')
        agent.user_id = self.agent_config.get('agent_id', '')
        return agent
    
    def create_section_task(self, agent: Agent) -> Task:
        """동적으로 섹션 작성 Task 생성 (안전 지침 포함)"""
        
        base_description = self.task_config.get("description", "")
        expected_output = self.task_config.get("expected_output", "")

        # 🔄 이전 작업 컨텍스트를 description에 추가 (제한 없음)
        context_info = ""
        if self.previous_context:
            context_str = str(self.previous_context)
            context_info = f"\n\n[이전 작업 컨텍스트]\n{context_str}"

        # 자연스러운 작업 지침 추가
        safe_description = base_description + context_info + f"""
        
        user_id = "{agent.user_id}"

        📋 작업 진행 방식:
        작업을 시작하기 전에 반드시 mem0 도구를 사용하여 관련 배경지식을 조회하세요. 
        mem0(user_id="{agent.user_id}", query="섹션 제목: {self.section_title}에 대한 배경지식")와 같이 
        구체적인 검색어로 조회한 후, 필요시 perplexity 도구도 활용하세요.

        중요한 것은 도구에서 얻은 정보에만 의존하지 말고, 조회한 배경지식을 바탕으로 
        전문가적 관점에서 창의적이고 포괄적인 보고서를 작성하는 것입니다. 
        mem0에 관련 지식이 없거나 검색 결과가 부족해도 포기하지 말고, 
        업계 표준과 모범 사례를 활용하여 반드시 완성된 보고서를 제공하세요.

        🔍 도구 사용 시 주의사항:
        - mem0, perplexity 도구 사용은 필수입니다
        - query 파라미터에는 반드시 구체적이고 명확한 검색어를 입력하세요
        - null이나 빈 값으로 query를 전달하는 것은 절대 금지입니다
        - 웹사이트 URL 직접 접속이나 임의 주소 생성은 하지 마세요
        """
        
        # 보고서 품질 기준 및 출력 형식
        enhanced_expected_output = expected_output + """
        
        📊 보고서 품질 기준:
        이 보고서는 최소 3,000-4,000단어 이상의 상세하고 전문적인 내용으로 작성되어야 합니다.
        표면적인 설명보다는 심층적인 분석을 제공하고, 실무에서 바로 활용할 수 있는 
        구체적인 사례와 예시를 다수 포함해야 합니다.

        내용 구성 시에는 관련 법규, 절차, 모범 사례, 주의사항을 포괄적으로 다루어 
        독자에게 실질적인 가치를 제공하세요. 각 주요 포인트는 상세한 하위 섹션으로 
        체계적으로 구분하여 작성하고, 해당 분야 전문가 수준의 통찰력을 발휘하세요.

        출력 형식은 순수한 마크다운 텍스트로 제공하되, 코드 블록(```markdown, ```html, ``` 등)으로 
        감싸지 마세요. 마크다운 형식의 제목, 목록, 강조 등을 직접 사용하여 
        체계적이고 읽기 쉬운 보고서를 완성하세요.

        기억하세요: 도구에서 충분한 정보를 얻지 못해도 포기하지 말고, 
        전문적 지식과 업계 표준을 바탕으로 반드시 완성된 보고서를 제공해야 합니다.
        """
        
        return Task(
            description=safe_description,
            expected_output=enhanced_expected_output,
            agent=agent
        )
    
    def create_crew(self) -> Crew:
        """동적으로 Crew 생성 - CrewAI 0.117.1 호환"""
        # 1) 동적 Agent, Task 생성
        agent        = self.create_dynamic_agent()
        section_task = self.create_section_task(agent)

        # 2) 클로저를 위한 로컬 변수 복사
        section_title    = self.section_title
        previous_context = self.previous_context

        # 3) WrappedCrew 서브클래스 정의 (kickoff_async 오버라이드)
        class WrappedCrew(Crew):
            async def kickoff_async(self, inputs=None):
                # ContextVar 설정
                token_ct, token_td, token_pid = set_crew_context(
                    crew_type="report",
                    todo_id=inputs.get('todo_id') if inputs else None,
                    proc_inst_id=inputs.get('proc_inst_id') if inputs else None
                )
                # 시작 로그 (클로저 변수 사용)
                print(f"[DynamicReportCrew] 시작합니다 - section={section_title}", flush=True)
                if previous_context:
                    snippet = str(previous_context)[:100]
                    print(f"[DynamicReportCrew] 이전 컨텍스트: {snippet}", flush=True)
                else:
                    print("[DynamicReportCrew] 이전 컨텍스트: 없음", flush=True)
                try:
                    # 실제 부모 클래스 kickoff_async 실행
                    return await super(WrappedCrew, self).kickoff_async(inputs=inputs)
                finally:
                    # ContextVar 복원
                    reset_crew_context(token_ct, token_td, token_pid)

        # 4) WrappedCrew 인스턴스 반환
        return WrappedCrew(
            agents=[agent],
            tasks=[section_task],
            process=Process.sequential,
            verbose=True,
            cache=True,
        )

