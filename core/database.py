import os
import json
import asyncio
import socket
import traceback
from typing import Optional, List, Dict, Any, Tuple
import re
from dotenv import load_dotenv
from supabase import create_client, Client

# ============================================================================  
# 설정 및 초기화  
# ============================================================================  

_supabase_client: Optional[Client] = None

def initialize_db():
    """환경변수 로드 및 Supabase 클라이언트 초기화"""
    global _supabase_client
    if _supabase_client is not None:
        return
    try:
        if os.getenv("ENV") != "production":
            load_dotenv()

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise RuntimeError("SUPABASE_URL 및 SUPABASE_KEY를 .env에 설정하세요.")
        client: Client = create_client(supabase_url, supabase_key)
        _supabase_client = client

    except Exception as e:
        print(f"❌ DB 초기화 실패: {e}")
        print(f"상세 정보: {traceback.format_exc()}")
        raise

def _handle_db_error(operation: str, error: Exception) -> None:
    """통합 DB 에러 처리"""
    error_msg = f"❌ [{operation}] DB 오류 발생: {error}"
    print(error_msg)
    print(f"상세 정보: {traceback.format_exc()}")
    raise Exception(f"{operation} 실패: {error}")


def get_db_client() -> Client:
    """초기화된 Supabase 클라이언트를 반환"""
    if _supabase_client is None:
        raise RuntimeError("DB 클라이언트가 초기화되지 않았습니다. initialize_db()를 먼저 호출하세요.")
    return _supabase_client 

# ============================================================================  
# 작업 조회 및 상태 관리  
# ============================================================================  

async def fetch_pending_task(limit: int = 1) -> Optional[Dict[str, Any]]:
    """Supabase RPC로 대기중인 작업 조회 및 상태 업데이트"""
    try:
        supabase = get_db_client()
        consumer_id = socket.gethostname()
        resp = supabase.rpc(
            'crewai_deep_fetch_pending_task',
            {'p_limit': limit, 'p_consumer': consumer_id}
        ).execute()
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception as e:
        _handle_db_error("작업조회", e)

async def fetch_task_status(todo_id: str) -> Optional[str]:
    """Supabase 테이블 조회로 작업 상태 조회"""
    try:
        supabase = get_db_client()
        resp = (
            supabase
            .table('todolist')
            .select('draft_status')
            .eq('id', todo_id)
            .single()
            .execute()
        )
        return resp.data.get('draft_status') if resp.data else None
    except Exception as e:
        _handle_db_error("상태조회", e)

# ============================================================================  
# 완료된 데이터 조회  
# ============================================================================  

async def fetch_done_data(proc_inst_id: Optional[str]) -> List[Any]:
    """완료된 데이터 조회 (output만)"""
    if not proc_inst_id:
        return []
    try:
        supabase = get_db_client()
        resp = supabase.rpc(
            'fetch_done_data',
            {'p_proc_inst_id': proc_inst_id}
        ).execute()
        outputs = []
        for row in resp.data or []:
            outputs.append(row.get('output'))
        return outputs
    except Exception as e:
        _handle_db_error("완료데이터조회", e)
        return []

# ============================================================================  
# 결과 저장  
# ============================================================================  

async def save_task_result(todo_id: str, result: Any, final: bool = False) -> None:
    """Supabase RPC로 작업 결과 저장 호출"""
    def _sync():
        try:
            supabase = get_db_client()
            # 이미 dict/list면 그대로, 아니면 JSON 직렬화
            payload = result if isinstance(result, (dict, list)) else json.loads(json.dumps(result))
            supabase.rpc(
                'save_task_result',
                {
                    'p_todo_id': todo_id,
                    'p_payload': payload,
                    'p_final':   final
                }
            ).execute()
        except Exception as e:
            _handle_db_error("결과저장", e)

    await asyncio.to_thread(_sync)

# ============================================================================
# 사용자 및 에이전트 정보 조회 (Supabase)
# ============================================================================

async def fetch_participants_info(user_ids: str) -> Dict:
    """사용자 또는 에이전트 정보 조회"""
    def _sync():
        try:
            supabase = get_db_client()
            id_list = [id.strip() for id in user_ids.split(',') if id.strip()]
            
            user_info_list = []
            agent_info_list = []
            
            for user_id in id_list:
                # UUID가 아니면 사용자/에이전트 모두 스킵
                if not _is_valid_uuid(user_id):
                    continue

                # (기존 스타일 유지) 이메일로 사용자 조회 시도
                user_data = _get_user_by_email(supabase, user_id)
                if user_data:
                    user_info_list.append(user_data)
                    continue

                # UUID인 경우에만 ID로 에이전트 조회
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
            'name': user.get('username'),
            'tenant_id': user.get('tenant_id')
        }
    return None

def _get_agent_by_id(supabase: Client, user_id: str) -> Optional[Dict]:
    """UUID `id`로 에이전트 조회 (비-UUID는 호출하지 않음)"""
    resp = supabase.table('users').select(
        'id, username, role, goal, persona, tools, profile, is_agent, model, tenant_id'
    ).eq('id', user_id).execute()
    
    if resp.data:
        # is_agent=True 인 첫 행 선택
        rows = [r for r in resp.data if r.get('is_agent')]
        agent = rows[0] if rows else None
        if agent:
            return {
                'id': agent.get('id'),
                'name': agent.get('username'),
                'role': agent.get('role'),
                'goal': agent.get('goal'),
                'persona': agent.get('persona'),
                'tools': agent.get('tools'),
                'profile': agent.get('profile'),
                'model': agent.get('model'),
                'tenant_id': agent.get('tenant_id')
            }
    return None

def _is_valid_uuid(value: str) -> bool:
    """UUID v1~v5 형식인지 검증 (소문자/대문자 허용)"""
    uuid_regex = re.compile(
        r'^[0-9a-fA-F]{8}-'
        r'[0-9a-fA-F]{4}-'
        r'[1-5][0-9a-fA-F]{3}-'
        r'[89abAB][0-9a-fA-F]{3}-'
        r'[0-9a-fA-F]{12}$'
    )
    return bool(uuid_regex.match(value))

# ============================================================================
# 폼 타입 조회 (Supabase)
# ============================================================================

async def fetch_form_types(tool_val: str, tenant_id: str) -> Tuple[str, List[Dict]]:
    """폼 타입 정보 조회 및 정규화 - form_id와 form_types 함께 반환"""
    def _sync():
        try:
            supabase = get_db_client()
            form_id = tool_val[12:] if tool_val.startswith('formHandler:') else tool_val
            
            resp = (
                supabase
                .table('form_def')
                .select('fields_json')
                .eq('id', form_id)
                .eq('tenant_id', tenant_id)
                .execute()
            )
            print(f'✅ 폼 타입 조회 완료: {resp}')
            fields_json = resp.data[0].get('fields_json') if resp.data else None
            print(f'✅ 폼 필드 JSON: {fields_json}')
            if not fields_json:
                return form_id, [{'key': form_id, 'type': 'default', 'text': ''}]

            return form_id, fields_json
            
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
            supabase = get_db_client()
            
            resp = (
                supabase
                .table('users')
                .select('id, username, role, goal, persona, tools, profile, model, tenant_id')
                .eq('is_agent', True)
                .execute()
            )
            rows = resp.data or []
            normalized = []
            for row in rows:
                tools_val = row.get('tools') or 'mem0'
                normalized.append({
                    'id': row.get('id'),
                    'name': row.get('username'),
                    'role': row.get('role'),
                    'goal': row.get('goal'),
                    'persona': row.get('persona'),
                    'tools': tools_val,
                    'profile': row.get('profile'),
                    'model': row.get('model'),
                    'tenant_id': row.get('tenant_id')
                })
            print(f'✅ 에이전트 {len(normalized)}개 조회 완료')
            return normalized
            
        except Exception as e:
            print(f"❌ 에이전트 조회 실패: {str(e)}")
            print(f"상세 정보: {traceback.format_exc()}")
            return []
            
    return await asyncio.to_thread(_sync) 

# ============================================================================
# 테넌트 MCP 설정 조회 (Supabase)
# ============================================================================

def fetch_tenant_mcp_config(tenant_id: str) -> Optional[Dict[str, Any]]:
    """테넌트 MCP 설정 조회"""
    try:
        supabase = get_db_client()
        resp = (
            supabase
            .table('tenants')
            .select('mcp')
            .eq('id', tenant_id)
            .single()
            .execute()
        )
        return resp.data.get('mcp') if resp.data else None
    except Exception as e:
        _handle_db_error("테넌트MCP설정조회", e)
