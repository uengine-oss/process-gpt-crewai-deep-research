import asyncio
import logging
import json
import os
import sys
from typing import Optional, List, Dict, Any
from contextvars import ContextVar
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import RealDictCursor
from supabase import create_client, Client

# 필요한 모듈 임포트
from .flows.multi_format_flow import MultiFormatFlow
from .feedback.diff_util import compare_report_changes, extract_changes
from .feedback.agent_feedback_analyzer import AgentFeedbackAnalyzer
from .context_manager import context_manager


# ============================================================================
# 공통 설정 및 초기화
# ============================================================================

logger = logging.getLogger("polling_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 공통 컨텍스트 변수
db_config_var = ContextVar('db_config', default={})
supabase_client_var = ContextVar('supabase', default=None)


def initialize_connections():
    """데이터베이스 및 Supabase 연결 초기화"""
    try:
        if os.getenv("ENV") != "production":
            load_dotenv()

        # Supabase 클라이언트 설정
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        supabase_client_var.set(supabase)

        # PostgreSQL 접속 정보 설정
        db_config = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT")
        }
        db_config_var.set(db_config)
        logger.info("✅ 데이터베이스 연결 초기화 완료")

    except Exception as e:
        logger.error(f"❌ 데이터베이스 설정 오류: {e}")


# ============================================================================
# 새 작업 처리 (TodoList Polling)
# ============================================================================

async def fetch_pending_tasks(limit: int = 1) -> Optional[List[Dict]]:
    """새로 처리할 작업 조회 (FOR UPDATE SKIP LOCKED)"""
    db_config = db_config_var.get()
    connection = psycopg2.connect(**db_config)
    connection.autocommit = False
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT * FROM todolist
        WHERE agent_mode = 'DRAFT' 
        AND (draft IS NULL OR draft::text = '' OR draft::text = 'EMPTY') 
        AND status = 'IN_PROGRESS'
        ORDER BY start_date ASC LIMIT %s FOR UPDATE SKIP LOCKED
    """, (limit,))

    row = cursor.fetchone()
    if row:
        return [{'row': row, 'connection': connection, 'cursor': cursor}]
    else:
        cursor.close()
        connection.close()
        return None


async def _load_previous_outputs_to_context(proc_inst_id: Optional[str], activity_name: Optional[str]):
    """같은 proc_inst_id의 완료된 작업 output을 context에 저장"""
    if not proc_inst_id:
        return
    conn = None
    cursor = None
    try:
        db_config = db_config_var.get()
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT output FROM todolist
               WHERE proc_inst_id = %s AND status = 'DONE' AND output IS NOT NULL
               ORDER BY start_date ASC""",
            (proc_inst_id,)
        )
        rows = cursor.fetchall()
        outputs = [row['output'] if isinstance(row, dict) else row[0] for row in rows]
        if outputs:
            context_manager.save_context(proc_inst_id, activity_name or '', {'outputs': outputs})
            logger.info(f"💾 이전 완료 작업 outputs 저장: proc_inst_id={proc_inst_id}, activity_name={activity_name}, count={len(outputs)}")
    except Exception as e:
        logger.error(f"❌ 이전 완료 작업 outputs 로드 오류 {proc_inst_id}: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


async def process_new_task(bundle: Dict):
    """새 작업 처리 - 컨텐츠 생성"""
    row, conn, cur = bundle['row'], bundle['connection'], bundle['cursor']

    try:
        logger.info(f"🆕 새 작업 처리: {row['id']}")
        
        # 이전 완료 작업 outputs 컨텍스트 로드
        await _load_previous_outputs_to_context(
            row.get('proc_inst_id'),
            row.get('activity_name', '')
        )

        # 사용자 정보 및 폼 정보 조회
        user_info = await _get_user_info(row.get('user_id'))
        form_types = await _get_form_types(row.get('tool', ''))
        
        # 컨텐츠 생성
        result = await _generate_content(row, form_types, user_info)
        
        # 결과 저장
        cur.execute("UPDATE todolist SET draft = %s WHERE id = %s", 
                   (json.dumps(result), row['id']))
        conn.commit()
        logger.info(f"✅ 작업 완료: {row['id']}")

    except Exception as e:
        logger.error(f"❌ 작업 처리 오류 {row['id']}: {e}")
        cur.execute("UPDATE todolist SET draft = %s WHERE id = %s", (json.dumps({}), row['id']))
        conn.commit()
    finally:
        cur.close()
        conn.close()


async def _get_user_info(user_email: str) -> Dict:
    """사용자 정보 조회"""
    supabase = supabase_client_var.get()
    resp = supabase.table('users').select('username').eq('email', user_email).execute()
    username = resp.data[0]['username'] if resp.data else None
    
    return {
        'email': user_email,
        'name': username,
        'department': '인사팀',
        'position': '사원'
    }


async def _get_form_types(tool_val: str) -> List[Dict]:
    """폼 타입 정보 조회 및 정규화"""
    form_id = tool_val[12:] if tool_val.startswith('formHandler:') else tool_val
    
    supabase = supabase_client_var.get()
    resp = supabase.table('form_def').select('fields_json').eq('id', form_id).execute()
    fields_json = resp.data[0].get('fields_json') if resp.data else None
    
    if not fields_json:
        return [{'id': form_id, 'type': 'default'}]
    
    form_types = []
    for field in fields_json:
        field_type = field.get('type', '').lower()
        normalized_type = field_type if field_type in ['report', 'slide'] else 'text'
        form_types.append({
            'id': field.get('key'),
            'type': normalized_type,
            'key': field.get('key'),
            'text': field.get('text', '')
        })
    
    return form_types


async def _generate_content(row: Dict, form_types: List[Dict], user_info: Dict) -> Dict:
    """컨텐츠 생성 실행"""
    # Flow 실행
    flow = MultiFormatFlow()
    flow.state.topic = row.get('activity_name', '')
    flow.state.form_types = form_types
    flow.state.user_info = user_info
    flow.state.todo_id = row.get('id')
    flow.state.proc_inst_id = row.get('proc_inst_id')
    flow.state.form_id = form_types[0].get('id') if form_types else None
    
    result = await flow.kickoff_async()
    return result


# ============================================================================
# 완료 작업 피드백 처리 (Feedback Polling)
# ============================================================================

async def fetch_completed_tasks(limit: int = 1) -> Optional[List[Dict]]:
    """피드백 대상 완료 작업 조회"""
    db_config = db_config_var.get()
    connection = psycopg2.connect(**db_config)
    connection.autocommit = False
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    try:
        cursor.execute("""
            UPDATE todolist SET feedback = '{}'::jsonb
            WHERE id = (
                SELECT id FROM todolist 
                WHERE status = 'DONE' AND output IS NOT NULL 
                AND draft IS NOT NULL 
                AND (feedback IS NULL OR feedback::text = '' OR feedback::text = 'EMPTY')
                ORDER BY start_date ASC LIMIT 1 FOR UPDATE SKIP LOCKED
            ) RETURNING *
        """)
        
        row = cursor.fetchone()
        if row:
            connection.commit()
            return [{'row': row, 'connection': connection, 'cursor': cursor}]
        else:
            cursor.close()
            connection.close()
            return None
            
    except Exception as e:
        logger.error(f"❌ 완료 작업 조회 오류: {e}")
        cursor.close()
        connection.close()
        return None


async def process_completed_task(bundle: Dict):
    """완료 작업 피드백 분석"""
    row, conn, cur = bundle['row'], bundle['connection'], bundle['cursor']

    try:
        logger.info(f"🔍 피드백 분석: {row['id']}")
        
        draft_value = row.get('draft')
        output_value = row.get('output')
        
        if not (draft_value and output_value):
            logger.warning(f"⚠️ draft 또는 output 없음: {row['id']}")
            return
        
        # 변경사항 분석
        feedback_list = await _analyze_changes(draft_value, output_value, row)
        
        # 피드백 저장
        cur.execute("UPDATE todolist SET feedback = %s WHERE id = %s",
                   (json.dumps(feedback_list, ensure_ascii=False), row['id']))
        conn.commit()
        
        logger.info(f"✅ 피드백 저장 완료: {row['id']} ({len(feedback_list)}개)")

    except Exception as e:
        logger.error(f"❌ 피드백 처리 오류 {row['id']}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


async def _analyze_changes(draft_value: Any, output_value: Any, row: Dict) -> List[Dict]:
    """변경사항 분석 및 피드백 생성"""
    try:
        # Diff 분석
        diff_result = compare_report_changes(
            json.dumps(draft_value) if isinstance(draft_value, dict) else str(draft_value),
            json.dumps(output_value) if isinstance(output_value, dict) else str(output_value)
        )
        
        if not diff_result.get('unified_diff'):
            logger.info("📝 변경사항 없음")
            return []
        
        # 변경사항 로깅
        extract_changes(
            diff_result.get('draft_content', ''), 
            diff_result.get('output_content', '')
        )

        # 에이전트 피드백 생성
        analyzer = AgentFeedbackAnalyzer()
        
        feedback_list = await analyzer.analyze_diff_and_generate_feedback(
            json.dumps(draft_value) if isinstance(draft_value, dict) else str(draft_value),
            json.dumps(output_value) if isinstance(output_value, dict) else str(output_value),
            todo_id=row.get('id'),
            proc_inst_id=row.get('proc_inst_id')
        )
        
        return feedback_list or []
        
    except Exception as e:
        logger.error(f"❌ 변경사항 분석 오류: {e}")
        return []


# ============================================================================
# 통합 Polling 실행부
# ============================================================================

async def start_todolist_polling(interval: int = 7):
    """새 작업 처리 폴링 시작"""
    logger.info("🚀 TodoList 폴링 시작")
    while True:
        try:
            tasks = await fetch_pending_tasks()
            if tasks:
                for bundle in tasks:
                    await process_new_task(bundle)
        except Exception as e:
            logger.error(f"❌ TodoList 폴링 오류: {e}")
        await asyncio.sleep(interval)


async def start_feedback_polling(interval: int = 10):
    """완료 작업 피드백 폴링 시작"""
    logger.info("🚀 피드백 폴링 시작")
    while True:
        try:
            tasks = await fetch_completed_tasks()
            if tasks:
                for bundle in tasks:
                    await process_completed_task(bundle)
        except Exception as e:
            logger.error(f"❌ 피드백 폴링 오류: {e}")
        await asyncio.sleep(interval)