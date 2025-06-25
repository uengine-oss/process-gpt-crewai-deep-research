import os
import json
from typing import List
import anyio
from pathlib import Path

class SafeToolLoader:
    """도구 이름만 관리하는 간소화된 로더"""
    def __init__(self):
        # 기본적으로 mem0과 perplexity(mcp)만 허용
        self.allowed_tools = ["mem0", "perplexity(mcp)"]

    def _load_mem0(self) -> List:
        """mem0 도구 로드"""
        try:
            from .knowledge_manager import Mem0Tool
            return [Mem0Tool()]
        except Exception as e:
            print(f"[ToolLoader] mem0 로드 실패: {e}")
            return []

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

    def _load_perplexity(self) -> List:
        """perplexity(mcp) 도구 로드"""
        try:
            from mcp import StdioServerParameters
            from crewai_tools import MCPServerAdapter
            self._apply_perplexity_patch()
            # 프로젝트 루트의 config/mcp.json 경로 계산
            config_path = Path(__file__).resolve().parents[3] / "config" / "mcp.json"
            print(f"[ToolLoader] mcp.json 경로: {config_path}")
            with open(config_path, 'r') as f:
                mcp_config = json.load(f)
            server_cfg = mcp_config.get('mcpServers', {}).get('perplexity', {})
            params = StdioServerParameters(
                command=server_cfg.get('command'),
                args=server_cfg.get('args'),
                env=os.environ
            )
            adapter = MCPServerAdapter(params)
            return adapter.tools
        except Exception as e:
            print(f"[ToolLoader] perplexity(mcp) 로드 실패: {e}")
            return []

    def create_tools_from_names(self, tool_names: List[str]) -> List:
        """tool_names 리스트에서 실제 Tool 객체들 생성"""
        print(f"[ToolLoader] 도구 요청: {tool_names}")
        tools: List = []
        # mem0는 항상 기본 로드
        tools.extend(self._load_mem0())

        for name in tool_names:
            key = name.strip().lower()
            if key == "mem0":
                continue
            if key == "perplexity":
                tools.extend(self._load_perplexity())
            else:
                print(f"[ToolLoader] 미지원 도구 요청: {name}")
        print(f"[ToolLoader] 총 {len(tools)}개 도구 생성 완료")
        return tools
