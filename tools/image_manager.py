import os
import logging
import traceback
import base64
from typing import Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import requests

# OpenAI Python SDK v1 (>=1.x) 기준
from openai import OpenAI

# ============================================================================
# 설정
# ============================================================================
load_dotenv()
logger = logging.getLogger(__name__)

def _handle_error(operation: str, error: Exception) -> str:
    msg = f"❌ [{operation}] 오류: {error}"
    logger.error(msg)
    logger.error(traceback.format_exc())
    return msg

# ============================================================================
# 스키마
# ============================================================================
class ImageGenSchema(BaseModel):
    prompt: str = Field(..., description="생성할 이미지 설명")
    filename: Optional[str] = Field(None, description="저장 파일명(.png 권장). 없으면 자동 생성")
    size: str = Field("512x512", description="이미지 크기 (1024x1024)")
    quality: str = Field("standard", description="이미지 품질 (standard | hd)")

# ============================================================================
# Tool
# ============================================================================
class ImageGenTool(BaseTool):
    """🎨 DALL·E 3 기반 이미지 생성 + 저장 툴 (컨텍스트 최적화)"""
    name: str = "image_gen"
    description: str = (
        "OpenAI DALL·E 3로 이미지를 생성해 로컬에 저장하고 플레이스홀더를 반환합니다.\n\n"
        "⚠️ 필수 매개변수:\n"
        "- prompt (필수): 생성할 이미지에 대한 구체적이고 상세한 설명\n"
        "  예시: '전문적인 비즈니스 회의실에서 팀원들이 차트를 보고 토론하는 모습, 현대적이고 깔끔한 스타일'\n\n"
        "선택 매개변수:\n"
        "- filename (선택): 저장 파일명(.png 권장). 없으면 자동 생성\n"
        "- size (선택): 이미지 크기 (기본값: 1024x1024)\n"
        "- quality (선택): 이미지 품질 (standard | hd, 기본값: standard)\n\n"
        "사용 예시:\n"
        "image_gen(prompt='전문적인 데이터 분석 차트와 그래프가 있는 현대적인 대시보드, 파란색과 흰색 톤의 깔끔한 디자인')\n\n"
        "반환값: 이미지 플레이스홀더 (마크다운 이미지 태그 형태, 컨텍스트 절약용)\n"
        "후처리: 최종 결과 저장 시 플레이스홀더가 자동으로 base64로 교체됩니다."
    )
    args_schema: Type[ImageGenSchema] = ImageGenSchema

    # 🔒 Pydantic 모델의 private 속성으로 선언 (여기에만 실제 객체를 담아야 함)
    _client: OpenAI = PrivateAttr()
    _output_dir: Path = PrivateAttr()
    _max_files: int = PrivateAttr(default=20)

    def __init__(self, **data):
        super().__init__(**data)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        self._client = OpenAI(api_key=api_key)

        # 출력 디렉토리 설정 (우선순위: MCP_OUTPUT_DIR → PGPT_WORK_DIR → ./outputs/images)
        output_dir_env = os.getenv("MCP_OUTPUT_DIR") or os.getenv("PGPT_WORK_DIR")
        if output_dir_env:
            self._output_dir = Path(output_dir_env)
        else:
            # 이 파일이 tools/ 아래라면 부모의 부모 기준
            self._output_dir = Path(__file__).resolve().parents[1] / "outputs" / "images"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 관리 설정
        self._max_files = 20  # 최대 20개 파일 유지

    def _download_image(self, url: str, filename: str) -> str:
        """이미지 다운로드 및 512x512로 리사이즈하여 저장"""
        try:
            from PIL import Image
            from io import BytesIO
            
            # 이미지 다운로드
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            
            # PIL로 이미지 열기
            img = Image.open(BytesIO(resp.content))
            
            # 512x512로 리사이즈 (고품질 다운샘플링)
            img_resized = img.resize((512, 512), Image.LANCZOS)
            
            # 리사이즈된 이미지 저장
            filepath = self._output_dir / filename
            img_resized.save(filepath, "PNG", optimize=True)
            
            logger.info(f"이미지 리사이즈 완료: {img.size} → 512x512")
            return str(filepath)
            
        except ImportError:
            # PIL이 없는 경우 원본 그대로 저장
            logger.warning("PIL(Pillow)이 설치되지 않아 원본 크기로 저장됩니다.")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            filepath = self._output_dir / filename
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return str(filepath)
        except Exception as e:
            # 기타 오류 시 원본 그대로 저장
            logger.error(f"이미지 리사이즈 실패, 원본 저장: {e}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            filepath = self._output_dir / filename
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return str(filepath)

    def _encode_image_to_base64(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            logger.error(f"이미지 base64 인코딩 실패: {e}")
            return ""


    def _cleanup_old_files(self):
        """파일 개수가 최대치에 도달하면 모든 파일 삭제"""
        try:
            # 이미지 파일들만 필터링 (png, jpg, jpeg)
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(self._output_dir.glob(ext))
            
            # 파일 개수가 최대 개수에 도달하면 모든 파일 삭제
            if len(image_files) >= self._max_files:
                deleted_count = 0
                for file_path in image_files:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"파일 삭제 실패 {file_path.name}: {e}")
                
                logger.info(f"파일 정리 완료: {deleted_count}개 파일 모두 삭제")
                
        except Exception as e:
            logger.error(f"파일 정리 중 오류: {e}")

    def cleanup_all_images(self, force: bool = False) -> int:
        """모든 이미지 파일 강제 삭제 (프로세스 완료 시 호출)"""
        try:
            # 이미지 파일들만 필터링 (png, jpg, jpeg)
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(self._output_dir.glob(ext))
            
            if not image_files:
                logger.info("삭제할 이미지 파일이 없습니다.")
                return 0
            
            # 강제 삭제 또는 최대 개수 초과 시 삭제
            if force or len(image_files) >= self._max_files:
                deleted_count = 0
                for file_path in image_files:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"파일 삭제 실패 {file_path.name}: {e}")
                
                logger.info(f"🗑️ 이미지 파일 정리 완료: {deleted_count}개 파일 삭제")
                return deleted_count
            else:
                logger.info(f"이미지 파일 개수({len(image_files)})가 최대치({self._max_files}) 미만이므로 삭제하지 않습니다.")
                return 0
                
        except Exception as e:
            logger.error(f"이미지 파일 정리 중 오류: {e}")
            return 0

    def _run(self, prompt: str, filename: Optional[str] = None,
             size: str = "1024x1024", quality: str = "standard") -> str:
        try:
            # 매개변수 검증
            if not prompt or prompt.strip() == "":
                return "❌ 오류: prompt 매개변수가 비어있습니다. 구체적인 이미지 설명을 제공해주세요."
            
            if prompt.lower() in ["null", "none", "undefined", ""]:
                return "❌ 오류: prompt 매개변수가 유효하지 않습니다. 구체적인 이미지 설명을 제공해주세요."
            
            # 파일명 자동 생성
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}.png"

            logger.info(f"[image_gen] prompt='{prompt}', size={size}, quality={quality}, filename={filename}")

            # 이미지 생성 (DALL·E 3 고정)
            response = self._client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            image_url = response.data[0].url

            # 다운로드 & 저장
            saved_path = self._download_image(image_url, filename)
            
            # 오래된 파일 정리 (최대 20개 유지)
            self._cleanup_old_files()
            
            # 플레이스홀더 반환 (컨텍스트 절약)
            return f"![{filename}](IMAGE_PLACEHOLDER:{filename})"

        except Exception as e:
            return _handle_error("image_gen", e)

    def replace_placeholders_with_base64(self, content: str) -> str:
        """플레이스홀더를 base64로 교체하는 후처리 메서드"""
        import re
        
        try:
            # IMAGE_PLACEHOLDER:filename 패턴 찾기
            pattern = r'!\[([^\]]+)\]\(IMAGE_PLACEHOLDER:([^)]+)\)'
            
            def replace_placeholder(match):
                alt_text = match.group(1)
                filename = match.group(2)
                
                # 파일 경로 구성
                image_path = self._output_dir / filename
                
                if image_path.exists():
                    # base64 인코딩
                    base64_data = self._encode_image_to_base64(str(image_path))
                    if base64_data:
                        return f"![{alt_text}](data:image/png;base64,{base64_data})"
                    else:
                        return f"![{alt_text}](이미지 로드 실패: {filename})"
                else:
                    return f"![{alt_text}](이미지 파일 없음: {filename})"
            
            # 모든 플레이스홀더 교체
            return re.sub(pattern, replace_placeholder, content)
            
        except Exception as e:
            logger.error(f"플레이스홀더 교체 실패: {e}")
            return content
