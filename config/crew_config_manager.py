from crewai import Crew
from .crew_event_logger import CrewAIEventLogger
from crews.ExecutionPlanningCrew import ExecutionPlanningCrew
from crews.AgentMatchingCrew import AgentMatchingCrew
from crews.FormCrew import FormCrew
from crews.SlideCrew import SlideCrew


try:
    # 최신 버전 (>=0.186.x) 경로
    from crewai.events import CrewAIEventsBus
    from crewai.events import (
        TaskStartedEvent,      # ← 다만 최신 문서 이벤트 이름 확인 필요
        TaskCompletedEvent,    # 예시이므로 실제 이름과 매핑되는지 확인
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
    )
except ImportError:
    # 구버전 (예: 0.175 이하) 경로
    from crewai.utilities.events import CrewAIEventsBus
    from crewai.utilities.events import TaskStartedEvent, TaskCompletedEvent
    from crewai.utilities.events import ToolUsageStartedEvent, ToolUsageFinishedEvent


# ============================================
# 글로벌 상태 관리 (Singleton 패턴)
# ============================================
_global_event_logger = None
_global_listeners_registered = False

class CrewConfigManager:
    """크루 구성 및 글로벌 이벤트 시스템 연동 매니저"""
    
    # ============================================
    # 초기화 및 설정
    # ============================================
    
    def __init__(self) -> None:
        """크루 구성 매니저 초기화"""
        try:
            self._setup_global_logger()
            self._setup_event_system()
            print("🎯 크루 구성 매니저 초기화 완료")
        except Exception as e:
            print(f"❌ 크루 구성 매니저 초기화 실패: {e}")
            raise
    
    def _setup_global_logger(self) -> None:
        """글로벌 이벤트 로거 설정 (Singleton)"""
        global _global_event_logger
        
        if _global_event_logger is None:
            _global_event_logger = CrewAIEventLogger()
            print("🆕 글로벌 이벤트 로거 생성")
        else:
            print("♻️ 기존 글로벌 이벤트 로거 재사용")
        
        self.event_logger = _global_event_logger
    
    def _setup_event_system(self) -> None:
        """이벤트 버스 및 리스너 설정"""
        global _global_listeners_registered
        
        if not _global_listeners_registered:
            self.event_bus = CrewAIEventsBus()
            self._register_event_listeners()
            _global_listeners_registered = True
            print("✅ 이벤트 리스너 등록 완료")
        else:
            print("♻️ 이벤트 리스너 이미 등록됨")
    
    # ============================================
    # 이벤트 시스템 관리
    # ============================================
    
    def _register_event_listeners(self) -> None:
        """이벤트 리스너 등록"""
        events = [
            TaskStartedEvent, TaskCompletedEvent,
            ToolUsageStartedEvent, ToolUsageFinishedEvent
        ]
        
        for evt in events:
            @self.event_bus.on(evt)
            def _handler(source, event, evt=evt):
                self._display_progress(event)
                self.event_logger.on_event(event, source)
    
    def _display_progress(self, event):
        """이벤트 진행 상황 출력"""
        event_type = event.type
        
        if event_type == "agent_execution_started":
            role = getattr(event.agent, 'role', 'Unknown')
            print(f"🤖 에이전트 시작: {role}")
        elif event_type == "agent_execution_completed":
            print("✅ 에이전트 종료")
        elif event_type == "task_started":
            task_id = getattr(event.task, 'id', 'unknown')
            print(f"📝 태스크 시작: {task_id}")
        elif event_type == "task_completed":
            print("✅ 태스크 완료")
        elif event_type == "llm_call_started":
            print("🔍 LLM 호출 시작")
        elif event_type == "llm_call_completed":
            print("✅ LLM 호출 완료")
        elif event_type == "tool_usage_started":
            tool_name = getattr(event, 'tool_name', 'tool')
            print(f"🔧 도구 시작: {tool_name}")
        elif event_type == "tool_usage_finished":
            tool_name = getattr(event, 'tool_name', 'tool')
            print(f"✅ 도구 종료: {tool_name}")
    
    # ============================================
    # 크루 생성 팩토리
    # ============================================
    
    def _create_crew(self, crew_class, crew_name: str, icon: str) -> Crew:
        """공통 크루 생성 로직"""
        try:
            crew_instance = crew_class()
            crew = crew_instance.crew()
            print(f"{icon} {crew_name} 생성 완료")
            return crew
        except Exception as e:
            print(f"❌ {crew_name} 생성 실패: {e}")
            raise
    
    def create_execution_planning_crew(self, **kwargs) -> Crew:
        """Execution Planning Crew 생성"""
        return self._create_crew(ExecutionPlanningCrew, "Execution Planning Crew", "🤖")
    
    def create_agent_matching_crew(self, **kwargs) -> Crew:
        """Agent Matching Crew 생성"""
        return self._create_crew(AgentMatchingCrew, "Agent Matching Crew", "🎯")
    
    def create_form_crew(self, **kwargs) -> Crew:
        """Form Crew 생성"""
        return self._create_crew(FormCrew, "Form Crew", "📋")
    
    def create_slide_crew(self, **kwargs) -> Crew:
        """Slide Crew 생성"""
        return self._create_crew(SlideCrew, "Slide Crew", "🎨")
    
 