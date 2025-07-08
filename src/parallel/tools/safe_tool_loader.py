import os
import json
import logging
import traceback
from typing import List
import anyio
from pathlib import Path

# ============================================================================
# 설정 및 초기화
# ============================================================================

# 로거 설정
logger = logging.getLogger("safe_tool_loader")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
    """도구 이름만 관리하는 간소화된 로더"""
    
    def __init__(self):
        self.allowed_tools = ["mem0", "perplexity(mcp)"]
        logger.info(f"✅ SafeToolLoader 초기화 완료 (허용 도구: {self.allowed_tools})")

    def create_tools_from_names(self, tool_names: List[str]) -> List:
        """tool_names 리스트에서 실제 Tool 객체들 생성"""
        if isinstance(tool_names, str):
            tool_names = [tool_names]
        logger.info(f"🔧 도구 생성 요청: {tool_names}")
        
        tools = []
        
        # mem0는 항상 기본 로드
        tools.extend(self._load_mem0())
        
        # 요청된 도구들 처리
        for name in tool_names:
            key = name.strip().lower()
            if key == "mem0":
                continue
            elif key == "perplexity":
                tools.extend(self._load_perplexity())
            else:
                logger.warning(f"⚠️ 미지원 도구 요청: {name}")
        
        logger.info(f"✅ 총 {len(tools)}개 도구 생성 완료")
        return tools

    # ============================================================================
    # 개별 도구 로더들
    # ============================================================================

    def _load_mem0(self) -> List:
        """mem0 도구 로드"""
        try:
            from .knowledge_manager import Mem0Tool
            logger.info("✅ mem0 도구 로드 성공")
            return [Mem0Tool()]
        except Exception as e:
            return _handle_error("mem0 로드", e)

    def _load_perplexity(self) -> List:
        """perplexity(mcp) 도구 로드"""
        try:
            from mcp import StdioServerParameters
            from crewai_tools import MCPServerAdapter
            
            # perplexity stderr 패치 적용
            self._apply_perplexity_patch()
            
            # MCP 설정 로드
            config_path = self._get_mcp_config_path()
            server_cfg = self._load_mcp_config(config_path)
            
            # MCP 서버 어댑터 생성
            params = StdioServerParameters(
                command=server_cfg.get('command'),
                args=server_cfg.get('args'),
                env=os.environ
            )
            adapter = MCPServerAdapter(params)
            
            logger.info("✅ perplexity(mcp) 도구 로드 성공")
            return adapter.tools
            
        except Exception as e:
            return _handle_error("perplexity(mcp) 로드", e)

    # ============================================================================
    # 헬퍼 메서드들
    # ============================================================================

    def _apply_perplexity_patch(self):
        """perplexity stderr 패치 적용"""
        from anyio._core._subprocesses import open_process as _orig
        import subprocess

        async def patched_open_process(*args, **kwargs):
            stderr = kwargs.get('stderr')
            if not (hasattr(stderr, 'fileno') and stderr.fileno()):
                kwargs['stderr'] = subprocess.PIPE
            return await _orig(*args, **kwargs)

        anyio.open_process = patched_open_process
        anyio._core._subprocesses.open_process = patched_open_process

    def _get_mcp_config_path(self) -> Path:
        """MCP 설정 파일 경로 반환"""
        config_path = Path(__file__).resolve().parents[3] / "config" / "mcp.json"
        logger.info(f"📄 mcp.json 경로: {config_path}")
        return config_path

    def _load_mcp_config(self, config_path: Path) -> dict:
        """MCP 설정 파일 로드"""
        with open(config_path, 'r') as f:
            mcp_config = json.load(f)
        return mcp_config.get('mcpServers', {}).get('perplexity', {})
