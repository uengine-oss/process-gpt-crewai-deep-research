import os
import sys
import pytest
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from tools.image_manager import ImageGenTool


@pytest.mark.integration
def test_image_tool_run_smoke():
    tool = ImageGenTool()
    result = tool._run(
        prompt="A simple blue square icon on white background, flat, minimal",
        size="1024x1024",
        quality="medium",
    )

    assert isinstance(result, str)
    # 마크다운 이미지 링크 형태인지 확인
    assert result.startswith("![generated_image_") or result.startswith("![img_") or result.startswith("![")
    assert "/storage/v1/object/public/task-image/" in result


