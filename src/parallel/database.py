import os
import json
import asyncio
import socket
import traceback
from contextvars import ContextVar
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from supabase import create_client, Client

# ============================================================================
# 설정 및 초기화
# ============================================================================

# DB 설정 보관용 ContextVar
db_config_var = ContextVar('db_config', default={})
supabase_client_var = ContextVar('supabase', default=None)

def initialize_db():
    """환경변수 로드 및 DB 설정 초기화"""
    try:
        if os.getenv("ENV") != "production":
            load_dotenv()
            
        # PostgreSQL 설정
        db_config = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
        }
        db_config_var.set(db_config)
        
        # Supabase 설정
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        supabase_client_var.set(supabase)
        
    except Exception as e:
        print(f"❌ DB 초기화 실패: {str(e)}")
        print(f"상세 정보: {traceback.format_exc()}")
        raise

def _get_connection():
    """DB 연결 생성"""
    try:
        config = db_config_var.get()
        conn = psycopg2.connect(**config)
        conn.autocommit = False
        return conn
    except Exception as e:
        print(f"❌ DB 연결 실패: {str(e)}")
        print(f"상세 정보: {traceback.format_exc()}")
        raise

def _handle_db_error(operation: str, error: Exception) -> None:
    """통합 DB 에러 처리"""
    error_msg = f"❌ [{operation}] DB 오류 발생: {str(error)}"
    print(error_msg)
    print(f"상세 정보: {traceback.format_exc()}")
    raise Exception(f"{operation} 실패: {error}")

# ============================================================================
# 작업 조회 및 상태 관리
# ============================================================================

async def fetch_pending_task(limit: int = 1) -> Optional[Dict[str, Any]]:
    """대기중인 작업 조회 및 상태 업데이트"""
    def _sync():
        conn = None
        cur = None
        try:
            conn = _get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            consumer_id = socket.gethostname()
            
            # 대기중인 작업 조회
            cur.execute("""
                SELECT * FROM todolist
                WHERE status = 'IN_PROGRESS'
                  AND (
                    (agent_mode = 'DRAFT'
                      AND (draft IS NULL OR draft::text = '' OR draft::text = 'EMPTY')
                      AND draft_status IS NULL
                    )
                    OR draft_status = 'FB_REQUESTED'
                  )
                ORDER BY start_date ASC
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            """, (limit,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            # 작업 상태 업데이트
            cur.execute(
                "UPDATE todolist SET draft_status = 'STARTED', consumer = %s WHERE id = %s",
                (consumer_id, row['id'])
            )
            conn.commit()
            
            return {'row': row, 'connection': conn, 'cursor': cur}
            
        except Exception as e:
            if conn:
                conn.rollback()
            if cur:
                cur.close()
            if conn:
                conn.close()
            _handle_db_error("작업조회", e)
            
    return await asyncio.to_thread(_sync)

async def fetch_task_status(todo_id: int) -> Optional[str]:
    """작업 상태 조회"""
    def _sync():
        conn = None
        cur = None
        try:
            conn = _get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute(
                "SELECT draft_status FROM todolist WHERE id = %s",
                (todo_id,)
            )
            row = cur.fetchone()
            
            return row.get('draft_status') if row else None
            
        except Exception as e:
            _handle_db_error("상태조회", e)
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
                
    return await asyncio.to_thread(_sync)

# ============================================================================
# 완료된 데이터 조회
# ============================================================================

async def fetch_done_data(proc_inst_id: Optional[str]) -> Tuple[List[Any], List[Any]]:
    """완료된 작업들의 output 및 feedback 조회"""
    def _sync():
        outputs = []
        feedbacks = []
        
        if not proc_inst_id:
            return outputs, feedbacks
            
        conn = None
        cur = None
        try:
            conn = _get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT status, output, feedback, draft FROM todolist
                WHERE proc_inst_id = %s
                  AND (
                    (status = 'DONE' AND output IS NOT NULL)
                    OR (status = 'IN_PROGRESS' AND feedback IS NOT NULL)
                  )
                ORDER BY start_date ASC
            """, (proc_inst_id,))
            
            rows = cur.fetchall()
            print(f"✅ 완료데이터 조회 완료: {len(rows)}개")
            
            for row in rows:
                output_data, feedback_data = _extract_row_data(row)
                outputs.append(output_data)
                feedbacks.append(feedback_data)
                
            return outputs, feedbacks
            
        except Exception as e:
            _handle_db_error("완료데이터조회", e)
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
                
    return await asyncio.to_thread(_sync)

def _extract_row_data(row: Dict) -> Tuple[Any, Any]:
    """행 데이터에서 output과 feedback 추출"""
    status = row.get('status')
    feedback = row.get('feedback')
    
    if status == 'IN_PROGRESS' and feedback is not None:
        draft = row.get('draft')
        return _parse_draft_data(draft), feedback
    else:
        return row.get('output'), feedback

def _parse_draft_data(draft: Any) -> Any:
    """draft 데이터 파싱"""
    if isinstance(draft, str):
        try:
            parsed = json.loads(draft)
            return parsed.get('reports')
        except Exception:
            return draft
    elif isinstance(draft, dict):
        return draft.get('reports')
    else:
        return draft

# ============================================================================
# 결과 저장
# ============================================================================

async def save_task_result(todo_id: int, result: Any, final: bool = False) -> None:
    """작업 결과 저장"""
    def _sync():
        conn = None
        cur = None
        try:
            conn = _get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            payload = json.dumps(result, ensure_ascii=False)
            
            # 기존 agent_mode 조회
            cur.execute("SELECT agent_mode FROM todolist WHERE id = %s", (todo_id,))
            row = cur.fetchone() or {}
            mode = row.get('agent_mode')
            
            if final:
                _save_final_result(cur, todo_id, payload, mode)
            else:
                _save_intermediate_result(cur, todo_id, payload)
                
            conn.commit()
            
        except Exception as e:
            if conn:
                conn.rollback()
            _handle_db_error("결과저장", e)
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
                
    await asyncio.to_thread(_sync)

def _save_final_result(cur, todo_id: int, payload: str, mode: str) -> None:
    """최종 결과 저장"""
    if mode == 'COMPLETE':
        cur.execute(
            "UPDATE todolist SET output=%s, draft=%s, status='SUBMITTED', draft_status='COMPLETED', consumer = %s WHERE id = %s",
            (payload, payload, None, todo_id)
        )
    else:
        cur.execute(
            "UPDATE todolist SET draft=%s, draft_status='COMPLETED', consumer = %s WHERE id = %s",
            (payload, None, todo_id)
        )

def _save_intermediate_result(cur, todo_id: int, payload: str) -> None:
    """중간 결과 저장"""
    cur.execute(
        "UPDATE todolist SET draft=%s WHERE id = %s",
        (payload, todo_id)
    )

# ============================================================================
# 사용자 및 에이전트 정보 조회 (Supabase)
# ============================================================================

async def fetch_participants_info(user_ids: str) -> Dict:
    """사용자 또는 에이전트 정보 조회"""
    def _sync():
        try:
            supabase = supabase_client_var.get()
            id_list = [id.strip() for id in user_ids.split(',') if id.strip()]
            
            user_info_list = []
            agent_info_list = []
            
            for user_id in id_list:
                # 이메일로 사용자 조회
                user_data = _get_user_by_email(supabase, user_id)
                if user_data:
                    user_info_list.append(user_data)
                    continue
                    
                # ID로 에이전트 조회
                agent_data = _get_agent_by_id(supabase, user_id)
                if agent_data:
                    agent_info_list.append(agent_data)
            
            result = {}
            if user_info_list:
                result['user_info'] = user_info_list
            if agent_info_list:
                result['agent_info'] = agent_info_list
            
            return result
            
        except Exception as e:
            _handle_db_error("참가자정보조회", e)
            
    return await asyncio.to_thread(_sync)

def _get_user_by_email(supabase: Client, user_id: str) -> Optional[Dict]:
    """이메일로 사용자 조회"""
    resp = supabase.table('users').select('id, email, username').eq('email', user_id).execute()
    if resp.data:
        user = resp.data[0]
        return {
            'email': user.get('email'),
            'name': user.get('username')
        }
    return None

def _get_agent_by_id(supabase: Client, user_id: str) -> Optional[Dict]:
    """ID로 에이전트 조회"""
    resp = supabase.table('users').select(
        'id, username, role, goal, persona, tools, profile, is_agent, model'
    ).eq('id', user_id).execute()
    
    if resp.data and resp.data[0].get('is_agent'):
        agent = resp.data[0]
        return {
            'id': agent.get('id'),
            'name': agent.get('username'),
            'role': agent.get('role'),
            'goal': agent.get('goal'),
            'persona': agent.get('persona'),
            'tools': agent.get('tools'),
            'profile': agent.get('profile'),
            'model': agent.get('model')
        }
    return None

# ============================================================================
# 폼 타입 조회 (Supabase)
# ============================================================================

async def fetch_form_types(tool_val: str) -> List[Dict]:
    """폼 타입 정보 조회 및 정규화"""
    def _sync():
        try:
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
            
        except Exception as e:
            _handle_db_error("폼타입조회", e)
            
    return await asyncio.to_thread(_sync)

# ============================================================================
# 에이전트 조회 (Supabase)
# ============================================================================

async def fetch_all_agents() -> List[Dict[str, Any]]:
    """모든 에이전트 조회 (is_agent=True만)"""
    def _sync():
        try:
            supabase = supabase_client_var.get()
            
            # is_agent=True인 에이전트만 조회
            resp = supabase.table('users').select('*').eq('is_agent', True).execute()
            agents = resp.data or []
            
            # tools 기본값 설정
            for agent in agents:
                if not agent.get('tools'):
                    agent['tools'] = 'mem0'
            
            print(f"✅ 에이전트 {len(agents)}개 조회 완료")
            return agents
            
        except Exception as e:
            print(f"❌ 에이전트 조회 실패: {str(e)}")
            print(f"상세 정보: {traceback.format_exc()}")
            return []
            
    return await asyncio.to_thread(_sync) 