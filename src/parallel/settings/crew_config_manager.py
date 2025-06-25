from crewai import Crew
from crewai.utilities.events import CrewAIEventsBus
from crewai.utilities.events.task_events import TaskStartedEvent, TaskCompletedEvent
from crewai.utilities.events.agent_events import AgentExecutionStartedEvent, AgentExecutionCompletedEvent
from crewai.utilities.events import ToolUsageStartedEvent, ToolUsageFinishedEvent
from crewai.utilities.events import LLMCallStartedEvent, LLMCallCompletedEvent
from . import CrewAIEventLogger
from ..crews.planning_crew.ExecutionPlanningCrew import ExecutionPlanningCrew
from ..crews.planning_crew.AgentMatchingCrew import AgentMatchingCrew
from ..crews.form_crew.FormCrew import FormCrew
from ..crews.slide_crew.SlideCrew import SlideCrew

# 🔒 글로벌 상태 관리 (Singleton 패턴)
_global_event_logger = None
_global_listeners_registered = False

# ==============================================
# CrewConfigManager: Initialization & Listener Registration
# ==============================================

class CrewConfigManager:
    """
    크루 구성 및 글로벌 이벤트 시스템 연동을 담당하는 매니저 클래스
    """
    
    # === Initialization ===
    def __init__(self) -> None:
        """
        크루 구성 매니저 초기화
        항상 Supabase에 기록하며 파일 로깅은 사용하지 않습니다.
        """
        # --- Global Logger (Singleton) ---
        global _global_event_logger, _global_listeners_registered
        
        # 🔒 글로벌 이벤트 로거 Singleton 패턴
        if _global_event_logger is None:
            _global_event_logger = CrewAIEventLogger()
            print("🆕 새로운 글로벌 이벤트 로거 생성")
        else:
            print("♻️ 기존 글로벌 이벤트 로거 재사용")
        
        self.event_logger = _global_event_logger
        
        # --- Register Global Event Listeners (Singleton) ---
        # 🔒 글로벌 이벤트 버스 리스너 중복 등록 방지
        if not _global_listeners_registered:
            self.event_bus = CrewAIEventsBus()
            self._setup_global_listeners()
            _global_listeners_registered = True
            print("✅ 글로벌 이벤트 리스너 등록 완료")
        else:
            print("♻️ 글로벌 이벤트 리스너 이미 등록됨 (재사용)")
        
        print("🎯 글로벌 이벤트 버스 연결 완료")
        print(f"   - Task/Agent 이벤트만 기록 (Crew 이벤트 제외)")
        print(f"   - 모든 크루가 자동으로 이벤트 로깅에 연결됨")
    
    # === Listener Setup ===
    def _setup_global_listeners(self) -> None:
        """에이전트, 태스크, 툴 이벤트 시작/종료만 등록"""
        events = [
            AgentExecutionStartedEvent, AgentExecutionCompletedEvent,
            TaskStartedEvent, TaskCompletedEvent,
            ToolUsageStartedEvent, ToolUsageFinishedEvent,
            LLMCallStartedEvent, LLMCallCompletedEvent
        ]
        for evt in events:
            @self.event_bus.on(evt)
            def _handler(source, event, evt=evt):
                # 진행상황 및 이벤트 로깅
                self._display_manus_style_progress(event)
                self.event_logger.on_event(event, source)
    
    def _display_manus_style_progress(self, event):
        """에이전트, 태스크, 툴 시작/종료 상태를 간단히 출력"""
        et = event.type
        # 에이전트 시작/종료
        if et == "agent_execution_started":
            role = getattr(event.agent, 'role', 'Unknown')
            print(f"🤖 에이전트 시작: {role}")
        elif et == "agent_execution_completed":
            print("✅ 에이전트 종료")
        # 태스크 시작/종료
        elif et == "task_started":
            tid = getattr(event.task, 'id', 'unknown')
            print(f"📝 태스크 시작: {tid}")
        elif et == "task_completed":
            print("✅ 태스크 완료")
        elif et == "llm_call_started":
            print("🔍 LLM 호출 시작")
        elif et == "llm_call_completed":
            print("✅ LLM 호출 완료")
        # 툴 사용 시작/종료
        elif et == "tool_usage_started":
            tool = getattr(event, 'tool_name', 'tool')
            print(f"🔧 도구 시작: {tool}")
        elif et == "tool_usage_finished":
            tool = getattr(event, 'tool_name', 'tool')
            print(f"✅ 도구 종료: {tool}")
    
    # === Crew Factory Methods ===
    def create_execution_planning_crew(self, **kwargs) -> Crew:
        """Execution Planning Crew 생성 (글로벌 이벤트 자동 연결)"""
        execution_planning_crew_instance = ExecutionPlanningCrew()
        crew = execution_planning_crew_instance.crew()
        print("🤖 Execution Planning Crew가 생성되었습니다. (글로벌 이벤트 로깅)")
        return crew
    
    def create_agent_matching_crew(self, **kwargs) -> Crew:
        """Agent Matching Crew 생성 (글로벌 이벤트 자동 연결)"""
        agent_matching_crew_instance = AgentMatchingCrew()
        crew = agent_matching_crew_instance.crew()
        print("🎯 Agent Matching Crew가 생성되었습니다. (글로벌 이벤트 로깅)")
        return crew
    
    def create_form_crew(self, **kwargs) -> Crew:
        """Form Crew 생성 (글로벌 이벤트 자동 연결)"""
        form_crew_instance = FormCrew()
        crew = form_crew_instance.crew()
        print("📋 Form Crew가 생성되었습니다. (글로벌 이벤트 로깅)")
        return crew
    
    def create_slide_crew(self, **kwargs) -> Crew:
        """Slide Crew 생성 (글로벌 이벤트 자동 연결)"""
        slide_crew_instance = SlideCrew()
        crew = slide_crew_instance.crew()
        print("🎨 Slide Crew가 생성되었습니다. (글로벌 이벤트 로깅)")
        return crew
    
 