import os
import subprocess
import time
import logging
import traceback
from typing import List
import anyio
from mcp.client.stdio import StdioServerParameters
from crewai_tools import MCPServerAdapter
from core.database import fetch_tenant_mcp_config
from .knowledge_manager import Mem0Tool, MementoTool

# ============================================================================
# 설정 및 초기화
# ============================================================================

# 로거 설정
logger = logging.getLogger(__name__)

def _handle_error(operation: str, error: Exception) -> List:
    """통합 에러 처리"""
    error_msg = f"❌ [{operation}] 오류 발생: {str(error)}"
    logger.error(error_msg)
    logger.error(f"상세 정보: {traceback.format_exc()}")
    return []

# ============================================================================
# 도구 로더 클래스
# ============================================================================

class SafeToolLoader:
    """도구 로더 클래스"""
    adapters = []  # MCPServerAdapter 인스턴스 등록
    
    def __init__(self, tenant_id: str = None, user_id: str = None):
        self.tenant_id = tenant_id
        self.user_id = user_id
        # 직접 선언한 도구들
        self.local_tools = ["mem0", "memento"]
        logger.info(f"SafeToolLoader 초기화 완료 (tenant_id: {tenant_id}, user_id: {user_id})")

    def create_tools_from_names(self, tool_names: List[str]) -> List:
        """tool_names 리스트에서 실제 Tool 객체들 생성"""
        if isinstance(tool_names, str):
            tool_names = [tool_names]
        logger.info(f"도구 생성 요청: {tool_names}")
        
        tools = []
        
        # mem0, memento는 항상 기본 로드
        tools.extend(self._load_mem0())
        tools.extend(self._load_memento())
        
        # 요청된 도구들 처리
        for name in tool_names:
            key = name.strip().lower()
            if key in self.local_tools:
                continue  # 이미 기본 로드됨
            else:
                # 나머지는 모두 MCP 도구로 처리
                tools.extend(self._load_mcp_tool(key))
        
        logger.info(f"총 {len(tools)}개 도구 생성 완료")
        return tools

    # ============================================================================
    # 개별 도구 로더들
    # ============================================================================

    def _load_mem0(self) -> List:
        """mem0 도구 로드 - 에이전트별 메모리"""
        try:
            return [Mem0Tool(tenant_id=self.tenant_id, user_id=self.user_id)]
        except Exception as e:
            return _handle_error("mem0로드", e)

    def _load_memento(self) -> List:
        """memento 도구 로드"""
        try:
            return [MementoTool(tenant_id=self.tenant_id)]
        except Exception as e:
            return _handle_error("memento로드", e)

    def _load_mcp_tool(self, tool_name: str) -> List:
        """MCP 도구 로드 (timeout & retry 지원)"""
        self._apply_anyio_patch()
        
        server_cfg = self._load_mcp_config_from_db(tool_name)
        if not server_cfg:
            return []
        
        env_vars = os.environ.copy()
        env_vars.update(server_cfg.get("env", {}))
        timeout = server_cfg.get("timeout", 40)

        max_retries = 2
        retry_delay = 5

        for attempt in range(1, max_retries + 1):
            try:
                params = StdioServerParameters(
                    command=server_cfg["command"],
                    args=server_cfg.get("args", []),
                    env=env_vars,
                    timeout=timeout
                )
                
                adapter = MCPServerAdapter(params)
                SafeToolLoader.adapters.append(adapter)
                logger.info(f"{tool_name} MCP 로드 성공 (툴 {len(adapter.tools)}개): {[tool.name for tool in adapter.tools]}")
                return adapter.tools

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    return _handle_error(f"{tool_name}MCP로드", e)

    # ============================================================================
    # 헬퍼 메서드들
    # ============================================================================

    def _apply_anyio_patch(self):
        """anyio stderr 패치 적용"""
        from anyio._core._subprocesses import open_process as _orig

        async def patched_open_process(*args, **kwargs):
            stderr = kwargs.get('stderr')
            if not (hasattr(stderr, 'fileno') and stderr.fileno()):
                kwargs['stderr'] = subprocess.PIPE
            return await _orig(*args, **kwargs)

        anyio.open_process = patched_open_process
        anyio._core._subprocesses.open_process = patched_open_process

    def _load_mcp_config_from_db(self, tool_name: str) -> dict:
        """DB의 tenants 테이블에서 MCP 설정 로드"""
        try:
            if not self.tenant_id:
                return {}
            
            mcp_config = fetch_tenant_mcp_config(self.tenant_id)
            
            if mcp_config:
                tool_config = mcp_config.get('mcpServers', {}).get(tool_name, {})
                if tool_config:
                    return tool_config
                else:
                    return {}
            else:
                return {}
                        
        except Exception as e:
            return _handle_error(f"{tool_name}DB설정로드", e)

    @classmethod
    def shutdown_all_adapters(cls):
        """모든 MCPServerAdapter 연결 종료"""
        for adapter in cls.adapters:
            try:
                adapter.stop()
            except Exception as e:
                logger.error(f"❌ MCPServerAdapter_stop 오류: {e}")
        cls.adapters.clear()
