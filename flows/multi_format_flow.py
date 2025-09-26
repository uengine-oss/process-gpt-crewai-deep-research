import json
import re
import traceback
import asyncio
from typing import Dict, List, Any, Optional
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field

from config.crew_config_manager import CrewConfigManager
from crews.DynamicReportCrew import DynamicReportCrew
from config.crew_event_logger import CrewAIEventLogger
from core.database import save_task_result, fetch_all_agents

# ============================================================================
# 데이터 모델 정의
# ============================================================================

class Phase(BaseModel):
    forms: List[Dict[str, Any]] = Field(default_factory=list)

class ExecutionPlan(BaseModel):
    report_phase: Phase = Field(default_factory=Phase)
    slide_phase: Phase = Field(default_factory=Phase)
    text_phase: Phase = Field(default_factory=Phase)

class MultiFormatState(BaseModel):
    topic: str = ""
    user_info: List[Dict[str, Any]] = Field(default_factory=list)
    form_types: List[Dict[str, Any]] = Field(default_factory=list)
    execution_plan: Optional[ExecutionPlan] = None
    report_sections: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    section_contents: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    report_contents: Dict[str, str] = Field(default_factory=dict)
    slide_contents: Dict[str, str] = Field(default_factory=dict)
    text_contents: Dict[str, Any] = Field(default_factory=dict)
    todo_id: Optional[str] = None
    proc_inst_id: Optional[str] = None
    agent_info: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    query: str = ""  # query 필드값 (output 관련 지침 포함)
    feedback: str = ""  # feedback 컬럼값
    proc_form_id: Optional[str] = None
    form_html: Optional[str] = None

# ============================================================================
# 유틸리티 함수
# ============================================================================

def clean_json_response(raw_text: Any) -> str:
    """JSON 응답에서 코드 블록 제거"""
    text = str(raw_text or "")
    # ```json ... ``` 패턴 제거
    match = re.search(r"```(?:json)?[\r\n]+(.*?)[\r\n]+```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    # 전체 코드 블록 제거
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.split("\n")
        return "\n".join(lines[1:-1])
    return text

# ============================================================================
# 메인 플로우 클래스
# ============================================================================

class MultiFormatFlow(Flow[MultiFormatState]):
    def __init__(self):
        super().__init__()
        self.config_manager = CrewConfigManager()
        self.event_logger = CrewAIEventLogger()

    def _handle_error(self, stage: str, error: Exception) -> None:
        """통합 에러 처리"""
        error_msg = f"❌ [{stage}] 오류 발생: {str(error)}"
        print(error_msg)
        print(f"상세 정보: {traceback.format_exc()}")
        raise Exception(f"{stage} 실패: {error}")

    # ============================================================================
    # 1. 실행 계획 생성
    # ============================================================================

    @start()
    async def create_execution_plan(self) -> ExecutionPlan:
        """AI를 이용한 실행 계획 생성"""
        try:
            crew = self.config_manager.create_execution_planning_crew()
            result = await crew.kickoff_async(inputs={
                "topic": self.state.topic,
                "form_types": self.state.form_types,
                "todo_id": self.state.todo_id,
                "proc_inst_id": self.state.proc_inst_id
            })
            
            # JSON 파싱 및 계획 저장
            raw_text = getattr(result, 'raw', result)
            cleaned_text = clean_json_response(raw_text)
            parsed_data = json.loads(cleaned_text)
            plan_data = parsed_data.get('execution_plan', {})
            self.state.execution_plan = ExecutionPlan.model_validate(plan_data)
            
            return self.state.execution_plan
            
        except Exception as e:
            self._handle_error("실행계획생성", e)

    # ============================================================================
    # 2. 리포트 생성 및 병합
    # ============================================================================

    @listen("create_execution_plan")
    async def generate_reports(self) -> Dict[str, Dict[str, str]]:
        """리포트 섹션 생성 및 병합"""
        try:
            for report_form in self.state.execution_plan.report_phase.forms:
                report_key = report_form.get('key')
                
                # 섹션 목록 생성
                sections = await self._create_report_sections()
                self.state.report_sections[report_key] = sections
                self.state.section_contents[report_key] = {}
                
                # 섹션별 내용 생성
                await self._generate_section_contents(report_key, sections)
                
                # 섹션 병합
                await self._merge_report_sections(report_key, sections)
                
            return self.state.section_contents
            
        except Exception as e:
            self._handle_error("리포트생성", e)

    async def _create_report_sections(self) -> List[Dict[str, Any]]:
        """리포트 섹션 목록 생성"""
        # available_agents 규칙
        # - 우선선정 에이전트가 있으면 그것만 전달
        # - 없으면 전체 에이전트 전달
        prioritized_agents = self.state.agent_info or []
        if prioritized_agents:
            available_agents = prioritized_agents
        else:
            available_agents = await fetch_all_agents()
        # 에이전트 선택 모드 출력 (우선선정/전체조회)
        print(f"👥 에이전트 선택 모드: {'우선선정' if prioritized_agents else '전체조회'} (선택 {len(available_agents)}명)")
        # 이후 매핑에도 동일 목록 사용
        agents = available_agents

        crew = self.config_manager.create_agent_matching_crew()
        
        result = await crew.kickoff_async(inputs={
            "topic": self.state.topic,
            "user_info": self.state.user_info,
            "query": self.state.query,  # query 필드값
            "feedback": self.state.feedback,  # feedback 컬럼값
            "available_agents": available_agents,
            "todo_id": self.state.todo_id,
            "proc_inst_id": self.state.proc_inst_id
        })
        
        raw_text = getattr(result, 'raw', result)
        cleaned_text = clean_json_response(raw_text)
        parsed_data = json.loads(cleaned_text)
        sections = parsed_data.get('sections', parsed_data)  # 하위 호환성: sections 키가 없으면 전체를 배열로 간주

        for sec in sections:
            agent_ref = sec.get('agent', {}) or {}
            agent_id = agent_ref.get('agent_id')
            full_agent = next((a for a in agents if a['id'] == agent_id), None)
            if full_agent:
                sec['agent'] = {
                    'agent_id': full_agent['id'],
                    'name': full_agent['name'],
                    'role': full_agent['role'],
                    'goal': full_agent['goal'],
                    'persona': full_agent['persona'],
                    'tool_names': full_agent['tools'],
                    'agent_profile': full_agent['profile'],
                    'model': full_agent['model'],
                    'tenant_id': full_agent['tenant_id']
                }
        return sections

    async def _generate_section_contents(self, report_key: str, sections: List[Dict[str, Any]]) -> None:
        """섹션별 내용 비동기 생성"""
        # 비동기 작업 생성
        tasks = [
            asyncio.create_task(self._create_single_section(section, report_key))
            for section in sections
        ]
        
        # 완료 순서대로 처리
        section_map = {task: section for task, section in zip(tasks, sections)}
        pending_tasks = set(tasks)
        
        while pending_tasks:
            done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done_tasks:
                section = section_map[task]
                title = section.get('toc', {}).get('title', 'unknown')
                
                try:
                    content = task.result()
                    self.state.section_contents[report_key][title] = content
                except Exception as e:
                    self.state.section_contents[report_key][title] = f"섹션 생성 실패: {str(e)}"
                
                # 중간 결과 저장
                await self._save_intermediate_result(report_key, sections)

    async def _create_single_section(self, section: Dict[str, Any], report_key: str) -> str:
        """단일 섹션 내용 생성"""
        crew = DynamicReportCrew(
            section, 
            self.state.topic, 
            query=self.state.query,
            feedback=self.state.feedback
        )
        result = await crew.create_crew().kickoff_async(inputs={
            "todo_id": self.state.todo_id,
            "proc_inst_id": self.state.proc_inst_id,
            "report_form_id": report_key,
            "user_info": self.state.user_info,
            "query": self.state.query,  # query 필드값
            "feedback": self.state.feedback  # feedback 컬럼값
        })
        return getattr(result, 'raw', result)

    async def _merge_report_sections(self, report_key: str, sections: List[Dict[str, Any]]) -> None:
        """리포트 섹션 병합"""
        # 병합 시작 이벤트
        self.event_logger.emit_event(
            event_type="task_started",
            data={"role": "리포트 통합 전문가", "goal": "섹션을 하나의 완전한 문서로 병합", "agent_profile": "/images/chat-icon.png", "name": "리포트 통합 전문가"},
            job_id=f"final_report_merge_{report_key}",
            crew_type="report",
            todo_id=self.state.todo_id,
            proc_inst_id=self.state.proc_inst_id,
        )
        
        # 순서대로 병합
        ordered_titles = [sec.get('toc', {}).get('title', 'unknown') for sec in sections]
        merged_content = "\n\n---\n\n".join([
            self.state.section_contents[report_key][title]
            for title in ordered_titles
            if title in self.state.section_contents[report_key]
        ])
        
        self.state.report_contents[report_key] = merged_content
        
        # 병합 완료 이벤트
        self.event_logger.emit_event(
            event_type="task_completed",
            data={report_key: merged_content},
            job_id=f"final_report_merge_{report_key}",
            crew_type="report",
            todo_id=self.state.todo_id,
            proc_inst_id=self.state.proc_inst_id
        )

    async def _save_intermediate_result(self, report_key: str, sections: List[Dict[str, Any]]) -> None:
        """중간 결과 DB 저장"""
        ordered_titles = [s.get('toc', {}).get('title', 'unknown') for s in sections]
        merged_content = "\n\n---\n\n".join([
            self.state.section_contents[report_key][title]
            for title in ordered_titles
            if title in self.state.section_contents[report_key]
        ])
        
        self.state.report_contents[report_key] = merged_content
        
        if self.state.todo_id and self.state.proc_form_id and self.state.report_contents:
            result = {self.state.proc_form_id: self.state.report_contents}
            await save_task_result(self.state.todo_id, result)

    # ============================================================================
    # 3. 슬라이드 생성
    # ============================================================================

    @listen("generate_reports")
    async def generate_slides(self) -> Dict[str, str]:
        """슬라이드 생성 - 리포트 내용 또는 이전 결과물 기반"""
        try:
            # 리포트 기반 슬라이드 생성
            if self.state.report_contents:
                for report_key, content in self.state.report_contents.items():
                    await self._create_slides(content, report_key)
            
            # 이전 결과물 기반 슬라이드 생성
            else:
                await self._create_slides(self.state.query)
                
            return self.state.slide_contents
            
        except Exception as e:
            self._handle_error("슬라이드생성", e)

    async def _create_slides(self, content: str, report_key: str = None) -> None:
        """통합 슬라이드 생성 함수"""
        for slide_form in self.state.execution_plan.slide_phase.forms:
            # 리포트 기반인 경우 dependency 체크
            if report_key and report_key not in slide_form.get('dependencies', []):
                continue
                
            slide_key = slide_form['key']
            crew = self.config_manager.create_slide_crew()
            
            result = await crew.kickoff_async(inputs={
                'report_content': content,  # 리포트 내용 또는 이전 결과물
                'feedback': self.state.feedback,  # feedback 컬럼값
                'user_info': self.state.user_info,
                'todo_id': self.state.todo_id,
                'proc_inst_id': self.state.proc_inst_id,
                "slide_form_id": slide_key
            })
            
            self.state.slide_contents[slide_key] = getattr(result, 'raw', result)

    # ============================================================================
    # 4. 텍스트 생성
    # ============================================================================

    @listen("generate_slides")
    async def generate_texts(self) -> Dict[str, Any]:
        """텍스트 폼 생성 - 리포트 내용 또는 이전 결과물 기반"""
        try:
            # content 결정: 리포트가 있으면 리포트, 없으면 이전 결과물
            if self.state.report_contents:
                content = self.state.report_contents  # 리포트 내용
            else:
                content = self.state.query or ""  # query 필드값
            
            # 실행계획의 모든 text_phase form들에 매칭되는 form_type들을 한번에 수집
            all_target_form_types = []
            for text_form in self.state.execution_plan.text_phase.forms:
                text_key = text_form.get('key')
                if not text_key:
                    continue
                matching_form_types = [ft for ft in self.state.form_types if ft.get('key') == text_key]
                all_target_form_types.extend(matching_form_types)

            if all_target_form_types:
                await self._generate_text_content(content, all_target_form_types)
                
            return self.state.text_contents
            
        except Exception as e:
            self._handle_error("텍스트생성", e)


    async def _generate_text_content(self, content: Any, target_form_types: List[Dict]) -> None:
        """텍스트 내용 생성 - 한번에 모든 form_type 처리"""
        crew = self.config_manager.create_form_crew()

        result = await crew.kickoff_async(inputs={
            'report_content': content,  # 리포트 내용 또는 이전 결과물
            'feedback': self.state.feedback,  # feedback 컬럼값
            'topic': self.state.topic,
            'user_info': self.state.user_info,
            'todo_id': self.state.todo_id,
            'proc_inst_id': self.state.proc_inst_id,
            'form_type': target_form_types,  # 매칭된 모든 form_type들을 한번에 전달
            'form_html': self.state.form_html
        })
        
        raw_result = getattr(result, 'raw', result)
        await self._parse_text_results(raw_result)

    async def _parse_text_results(self, raw_result: str) -> None:
        """텍스트 결과 파싱 및 저장"""
        try:
            cleaned_result = clean_json_response(raw_result)
            parsed_results = json.loads(cleaned_result)
            # FormCrew에서 반환된 결과를 그대로 저장 (이미 {key: value} 형태)
            if isinstance(parsed_results, dict):
                self.state.text_contents.update(parsed_results)
            else:
                # 파싱 실패 시 기본 형태로 저장
                self.state.text_contents["text_result"] = {"text": cleaned_result}
                    
        except json.JSONDecodeError:
            self.state.text_contents["text_result"] = {"text": str(raw_result)}

    # ============================================================================
    # 5. 최종 결과 저장
    # ============================================================================

    @listen("generate_texts")
    async def save_final_results(self) -> None:
        """최종 결과 저장 및 출력"""
        try:
            print("\n" + "="*60)
            print("🎉 다중 포맷 생성 완료!")
            print("="*60)
            
            # 최종 결과 DB 저장
            if self.state.todo_id and self.state.proc_inst_id:
                all_results = {
                    **self.state.report_contents,
                    **self.state.slide_contents,
                    **self.state.text_contents
                }
                
                if all_results:
                    final_result = {self.state.proc_form_id: all_results}
                    await save_task_result(self.state.todo_id, final_result, final=True)
                    
                    # 처리 결과 출력
                    report_count = len(self.state.report_contents)
                    slide_count = len(self.state.slide_contents)
                    text_count = len(self.state.text_contents)
                    print(f"📊 처리 결과: 리포트 {report_count}개, 슬라이드 {slide_count}개, 텍스트 {text_count}개")

        except Exception as e:
            self._handle_error("최종결과저장", e)

