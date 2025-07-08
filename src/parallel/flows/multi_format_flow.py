import json
import re
import traceback
import asyncio
from typing import Dict, List, Any, Optional
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field

from ..settings.crew_config_manager import CrewConfigManager
from ..crews.report_crew.DynamicReportCrew import DynamicReportCrew
from ..settings.crew_event_logger import CrewAIEventLogger
from ..database import save_task_result, fetch_all_agents

# ============================================================================
# 데이터 모델 정의
# ============================================================================

class Phase(BaseModel):
    forms: List[Dict[str, Any]] = Field(default_factory=list)
    strategy: Optional[str] = None

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
    todo_id: Optional[int] = None
    proc_inst_id: Optional[str] = None
    agent_info: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    previous_context: str = ""

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
            plan_data = json.loads(cleaned_text).get('execution_plan', {})
            self.state.execution_plan = ExecutionPlan.parse_obj(plan_data)
            
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
        agents = await fetch_all_agents()
        crew = self.config_manager.create_agent_matching_crew()
        
        result = await crew.kickoff_async(inputs={
            "topic": self.state.topic,
            "user_info": self.state.user_info,
            "agent_info": self.state.agent_info,
            "previous_context": self.state.previous_context,
            "available_agents": agents,
            "todo_id": self.state.todo_id,
            "proc_inst_id": self.state.proc_inst_id
        })
        
        raw_text = getattr(result, 'raw', result)
        cleaned_text = clean_json_response(raw_text)
        return json.loads(cleaned_text)

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
        crew = DynamicReportCrew(section, self.state.topic, self.state.previous_context)
        result = await crew.create_crew().kickoff_async(inputs={
            "todo_id": self.state.todo_id,
            "proc_inst_id": self.state.proc_inst_id,
            "report_form_id": report_key
        })
        return getattr(result, 'raw', result)

    async def _merge_report_sections(self, report_key: str, sections: List[Dict[str, Any]]) -> None:
        """리포트 섹션 병합"""
        # 병합 시작 이벤트
        self.event_logger.emit_event(
            event_type="task_started",
            data={"role": "리포트 통합 전문가", "goal": "섹션을 하나의 완전한 문서로 병합"},
            job_id=f"merge-{report_key}",
            crew_type="report",
            todo_id=self.state.todo_id,
            proc_inst_id=self.state.proc_inst_id
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
            data={"final_result": merged_content},
            job_id=f"merge-{report_key}",
            crew_type="report",
            todo_id=self.state.todo_id,
            proc_inst_id=self.state.proc_inst_id,
            form_id=report_key
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
        
        if self.state.todo_id and self.state.proc_inst_id and self.state.report_contents:
            result = {self.state.proc_inst_id: self.state.report_contents}
            await save_task_result(self.state.todo_id, result)

    # ============================================================================
    # 3. 슬라이드 생성
    # ============================================================================

    @listen("generate_reports")
    async def generate_slides(self) -> Dict[str, str]:
        """리포트 기반 슬라이드 생성"""
        try:
            # 리포트 기반 슬라이드 생성
            if self.state.report_contents:
                for report_key, content in self.state.report_contents.items():
                    await self._create_slides_from_report(report_key, content)
            
            # 이전 컨텍스트 기반 슬라이드 생성
            else:
                await self._create_slides_from_context()
                
            return self.state.slide_contents
            
        except Exception as e:
            self._handle_error("슬라이드생성", e)

    async def _create_slides_from_report(self, report_key: str, content: str) -> None:
        """리포트 기반 슬라이드 생성"""
        for slide_form in self.state.execution_plan.slide_phase.forms:
            if report_key in slide_form.get('dependencies', []):
                slide_key = slide_form['key']
                crew = self.config_manager.create_slide_crew()
                
                result = await crew.kickoff_async(inputs={
                    'report_content': content,
                    'user_info': self.state.user_info,
                    'todo_id': self.state.todo_id,
                    'proc_inst_id': self.state.proc_inst_id,
                    "slide_form_id": slide_key
                })
                
                self.state.slide_contents[slide_key] = getattr(result, 'raw', result)

    async def _create_slides_from_context(self) -> None:
        """이전 컨텍스트 기반 슬라이드 생성"""
        for slide_form in self.state.execution_plan.slide_phase.forms:
            slide_key = slide_form['key']
            crew = self.config_manager.create_slide_crew()
            
            result = await crew.kickoff_async(inputs={
                'report_content': self.state.previous_context,
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
        """리포트 기반 텍스트 생성"""
        try:
            # 리포트 기반 텍스트 생성
            if self.state.report_contents:
                for report_key, content in self.state.report_contents.items():
                    await self._create_texts_from_report(report_key, content)
            
            # 이전 컨텍스트 기반 텍스트 생성
            else:
                await self._create_texts_from_context()
                
            return self.state.text_contents
            
        except Exception as e:
            self._handle_error("텍스트생성", e)

    async def _create_texts_from_report(self, report_key: str, content: str) -> None:
        """리포트 기반 텍스트 생성"""
        dependent_forms = [
            form for form in self.state.execution_plan.text_phase.forms
            if report_key in form.get('dependencies', [])
        ]
        
        if dependent_forms:
            form_keys = [form['key'] for form in dependent_forms]
            await self._generate_text_content(content, form_keys)

    async def _create_texts_from_context(self) -> None:
        """이전 컨텍스트 기반 텍스트 생성"""
        if self.state.execution_plan.text_phase.forms:
            form_keys = [form['key'] for form in self.state.execution_plan.text_phase.forms]
            # report_content 없이 이전 컨텍스트만 전달
            await self._generate_text_content("", form_keys)

    async def _generate_text_content(self, content: str, form_keys: List[str]) -> None:
        """텍스트 내용 생성"""
        crew = self.config_manager.create_form_crew()
        result = await crew.kickoff_async(inputs={
            'report_content': content,
            'topic': self.state.topic,
            'user_info': self.state.user_info,
            'todo_id': self.state.todo_id,
            'proc_inst_id': self.state.proc_inst_id,
            'text_form_keys': form_keys
        })
        
        raw_result = getattr(result, 'raw', result)
        await self._parse_text_results(raw_result, form_keys)

    async def _parse_text_results(self, raw_result: str, form_keys: List[str]) -> None:
        """텍스트 결과 파싱 및 저장"""
        try:
            cleaned_result = clean_json_response(raw_result)
            parsed_results = json.loads(cleaned_result)
            
            for form_key in form_keys:
                if form_key in parsed_results:
                    self.state.text_contents[form_key] = parsed_results[form_key]
                else:
                    self.state.text_contents[form_key] = raw_result
                    
        except json.JSONDecodeError:
            for form_key in form_keys:
                self.state.text_contents[form_key] = raw_result

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
                    final_result = {self.state.proc_inst_id: all_results}
                    await save_task_result(self.state.todo_id, final_result, final=True)
                    
                    # 처리 결과 출력
                    report_count = len(self.state.report_contents)
                    slide_count = len(self.state.slide_contents)
                    text_count = len(self.state.text_contents)
                    print(f"📊 처리 결과: 리포트 {report_count}개, 슬라이드 {slide_count}개, 텍스트 {text_count}개")

            # 완료 이벤트 발행
            self.event_logger.emit_event(
                event_type="crew_completed",
                data={},
                job_id="CREW_FINISHED",
                crew_type="crew",
                todo_id=self.state.todo_id,
                proc_inst_id=self.state.proc_inst_id
            )
            
        except Exception as e:
            self._handle_error("최종결과저장", e)