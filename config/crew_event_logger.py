import os
import uuid
import json
import traceback
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging

from core.database import initialize_db, get_db_client
from utils.context_manager import crew_type_var, todo_id_var, proc_id_var, form_id_var

# ============================================================================
# 초기화 및 설정
# ============================================================================

logger = logging.getLogger(__name__)

class CrewAIEventLogger:
    """CrewAI 이벤트 로깅 시스템 - Supabase 전용"""
    def __init__(self):
        """이벤트 로거 초기화"""
        # DB 싱글턴 초기화 및 클라이언트 가져오기
        initialize_db()
        self.supabase_client = get_db_client()
        logger.info("🎯 CrewAI Event Logger 초기화 완료")
        print("   - Supabase: ✅")

    # ============================================================================
    # 유틸리티 함수
    # ============================================================================

    def _handle_error(self, operation: str, error: Exception) -> None:
        """통합 에러 처리"""
        error_msg = f"❌ [{operation}] 오류 발생: {str(error)}"
        logger.error(error_msg)
        logger.error(f"상세 정보: {traceback.format_exc()}")

    def _generate_job_id(self, event_obj: Any, source: Any = None) -> str:
        """이벤트 객체에서 Job ID 생성"""
        if hasattr(event_obj, 'task') and hasattr(event_obj.task, 'id'):
            return str(event_obj.task.id)
        if source and hasattr(source, 'task') and hasattr(source.task, 'id'):
            return str(source.task.id)
        return 'unknown'

    def _create_event_record(self, event_type: str, data: Dict[str, Any], job_id: str, 
                           crew_type: str, todo_id: str, proc_inst_id: str) -> Dict[str, Any]:
        """이벤트 레코드 생성"""
        return {
            "id": str(uuid.uuid4()),
            "job_id": job_id,
            "todo_id": todo_id,
            "proc_inst_id": proc_inst_id,
            "event_type": event_type,
            "crew_type": crew_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ============================================================================
    # 이벤트 데이터 추출
    # ============================================================================

    def _extract_event_data(self, event_obj: Any, source: Any = None) -> Dict[str, Any]:
        """이벤트 데이터 추출"""
        event_type = event_obj.type
        
        try:
            if event_type == "task_started":
                return self._extract_task_started_data(event_obj)
            elif event_type == "task_completed":
                return self._extract_task_completed_data(event_obj)
            elif event_type.startswith('tool_'):
                return self._extract_tool_data(event_obj)
            else:
                return {"info": f"Event type: {event_type}"}
                
        except Exception as e:
            self._handle_error("데이터추출", e)
            return {"error": f"데이터 추출 실패: {str(e)}"}

    def _extract_task_started_data(self, event_obj: Any) -> Dict[str, Any]:
        """Task 시작 이벤트 데이터 추출"""
        agent = event_obj.task.agent
        return {
            "role": getattr(agent, 'role', 'Unknown'),
            "goal": getattr(agent, 'goal', 'Unknown'),
            "agent_profile": getattr(agent, 'profile', None) or "/images/chat-icon.png",
            "name": getattr(agent, 'name', 'Unknown')
        }

    def _extract_task_completed_data(self, event_obj: Any) -> Dict[str, Any]:
        """Task 완료 이벤트 데이터 추출"""
        final_result = getattr(event_obj, 'output', 'Completed')
        return {"final_result": str(final_result)}

    def _extract_tool_data(self, event_obj: Any) -> Dict[str, Any]:
        """Tool 사용 이벤트 데이터 추출"""
        tool_name = getattr(event_obj, 'tool_name', None)
        tool_args = getattr(event_obj, 'tool_args', None)
        query = None
        
        if tool_args:
            try:
                args_dict = json.loads(tool_args)
                query = args_dict.get('query')
            except Exception:
                query = None
                
        return {"tool_name": tool_name, "query": query}

    # ============================================================================
    # 데이터 직렬화 및 래핑
    # ============================================================================

    def _safe_serialize_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """이벤트 데이터 안전 직렬화"""
        safe_data = {}
        
        for key, value in event_data.items():
            try:
                if hasattr(value, 'raw'):
                    safe_data[key] = str(value.raw)
                elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, type(None))):
                    safe_data[key] = str(value)
                else:
                    safe_data[key] = value
            except Exception as e:
                logger.warning(f"데이터 직렬화 실패 ({key}): {e}")
                safe_data[key] = f"[직렬화 실패: {type(value).__name__}]"
                
        return safe_data

    def _apply_form_id_wrapper(self, event_data: Dict[str, Any], form_id: str, event_type: str) -> Dict[str, Any]:
        """form_id로 final_result 래핑"""
        if form_id and event_type == "task_completed" and "final_result" in event_data:
            original_result = event_data["final_result"]
            event_data["final_result"] = {form_id: original_result}
        return event_data

    # ============================================================================
    # 데이터베이스 저장
    # ============================================================================
    def _save_to_supabase(self, event_record: Dict[str, Any]) -> None:
        """Supabase에 이벤트 레코드 저장"""
        if not self.supabase_client:
            return
            
        try:
            def safe_serialize(obj):
                if hasattr(obj, 'raw'):
                    return str(obj.raw)
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return str(obj)
            
            serializable_record = json.loads(json.dumps(event_record, default=safe_serialize))
            self.supabase_client.table("events").insert(serializable_record).execute()
            
        except Exception as e:
            self._handle_error("Supabase저장", e)
            print(f"🔍 문제 데이터: {type(event_record.get('data', {}))}")

    # ============================================================================
    # 메인 이벤트 처리
    # ============================================================================

    def on_event(self, event_obj: Any, source: Any = None) -> None:
        """CrewAI 이벤트 자동 처리"""
        try:
            # Task, Tool 이벤트만 필터링
            if event_obj.type not in ["task_started", "task_completed", "tool_usage_started", "tool_usage_finished"]:
                return
            
            # 기본 데이터 추출
            job_id = self._generate_job_id(event_obj, source)
            event_data = self._extract_event_data(event_obj, source)
            
            # 컨텍스트 정보 가져오기
            crew_type = crew_type_var.get()
            todo_id = todo_id_var.get()
            proc_inst_id = proc_id_var.get()
            form_id = form_id_var.get()
            
            # form_id 래핑 적용
            event_data = self._apply_form_id_wrapper(event_data, form_id, event_obj.type)
            
            # 데이터 직렬화
            safe_data = self._safe_serialize_data(event_data)
            
            # 이벤트 레코드 생성 및 저장
            event_record = self._create_event_record(
                event_obj.type, safe_data, job_id, crew_type, todo_id, proc_inst_id
            )
            self._save_to_supabase(event_record)
            
            # 콘솔 출력
            tool_info = f" ({safe_data.get('tool_name', '')})" if event_obj.type.startswith('tool_') else ""
            print(f"📝 [{event_obj.type}]{tool_info} [{crew_type}] {job_id[:8]} → Supabase: {'✅' if self.supabase_client else '❌'}")
            
        except Exception as e:
            self._handle_error("이벤트처리", e)

    def emit_event(self, event_type: str, data: Dict[str, Any], job_id: str = None, 
                   crew_type: str = None, todo_id: str = None, proc_inst_id: str = None, 
                   form_id: str = None) -> None:
        """수동 커스텀 이벤트 발행"""
        try:
            # 컨텍스트 정보 설정
            crew_type = crew_type or crew_type_var.get()
            todo_id = todo_id or todo_id_var.get()
            proc_inst_id = proc_inst_id or proc_id_var.get()
            form_id = form_id or form_id_var.get()
            job_id = job_id or event_type
            
            # form_id 래핑 적용
            if form_id and "final_result" in data:
                original_result = data["final_result"]
                data = data.copy()
                data["final_result"] = {form_id: original_result}
            
            # 이벤트 레코드 생성 및 저장
            record = self._create_event_record(
                event_type, data, job_id, crew_type, todo_id, proc_inst_id
            )
            self._save_to_supabase(record)
            
            print(f"📝 [{event_type}] [{crew_type}] {job_id[:8]} → Supabase: {'✅' if self.supabase_client else '❌'}")
            
        except Exception as e:
            self._handle_error("커스텀이벤트발행", e) 