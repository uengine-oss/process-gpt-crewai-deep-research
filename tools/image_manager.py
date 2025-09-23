import os
import base64
import logging
import traceback
from typing import Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import uuid
from supabase import create_client

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
    size: str = Field(
        "1024x1024",
        description="이미지 크기 (예: 1024x1024 | 1536x1024 | 1024x1536)"
    )
    quality: str = Field(
        "medium",
        description="이미지 품질 (low | medium | high)"
    )

# ============================================================================
# Tool
# ============================================================================
class ImageGenTool(BaseTool):
    """🎨 GPT-Image (gpt-image-1) 기반 이미지 생성 + Supabase Storage 업로드 툴"""
    name: str = "image_gen"
    description: str = (
        "OpenAI gpt-image-1로 이미지를 생성해 Supabase Storage에 업로드하고 URL을 반환합니다.\n\n"
        "⚠️ 필수 매개변수:\n"
        "- prompt (필수): 생성할 이미지에 대한 구체적이고 상세한 설명\n"
        "  예시: '전문적인 비즈니스 회의실에서 팀원들이 차트를 보고 토론하는 모습, 현대적이고 깔끔한 스타일'\n\n"
        "선택 매개변수:\n"
        "- filename (선택): 저장 파일명(.png 권장). 없으면 자동 생성\n"
        "- size (선택): 이미지 크기 (기본값: 1024x1024; 1536x1024 | 1024x1536 권장)\n"
        "- quality (선택): 이미지 품질 (low | medium | high, 기본값: medium)\n\n"
        "사용 예시:\n"
        "image_gen(prompt='전문적인 데이터 분석 차트와 그래프가 있는 현대적인 대시보드, 파란색과 흰색 톤의 깔끔한 디자인')\n\n"
        "반환값: Supabase Storage URL (이미지 접근 가능한 공개 URL)"
    )
    args_schema: Type[ImageGenSchema] = ImageGenSchema

    _client: OpenAI = PrivateAttr()
    _supabase: Optional[object] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)

        # OpenAI 클라이언트 초기화
        # 환경변수 기반 초기화 (키 직접 전달 금지)
        # OpenAI SDK는 환경변수(OPENAI_API_KEY, OPENAI_BASE 등)를 자동 인식함
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        self._client = OpenAI()

        # Supabase 클라이언트 초기화
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if supabase_url and supabase_key:
            try:
                self._supabase = create_client(supabase_url, supabase_key)
                logger.info("✅ Supabase 클라이언트 초기화 완료")
            except Exception as e:
                logger.warning(f"❌ Supabase 클라이언트 초기화 실패: {e}")
                self._supabase = None
        else:
            logger.warning("❌ SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY 환경 변수가 설정되지 않았습니다.")

    def _upload_to_supabase(self, image_data: bytes, filename: str) -> Optional[str]:
        """이미지를 512x512로 리사이즈 후 Supabase Storage에 업로드하고 공개 URL 반환"""
        if not self._supabase:
            logger.error("Supabase 클라이언트가 초기화되지 않았습니다.")
            return None
            
        try:
            bucket_name = "task-image"
            
            # 이미지 리사이즈 처리
            try:
                from PIL import Image
                from io import BytesIO
                
                # PIL로 이미지 열기
                img = Image.open(BytesIO(image_data))
                
                # 512x512로 리사이즈 (고품질 다운샘플링)
                img_resized = img.resize((512, 512), Image.LANCZOS)
                
                # 리사이즈된 이미지를 바이트로 변환
                img_byte_arr = BytesIO()
                img_resized.save(img_byte_arr, format='PNG', optimize=True)
                img_byte_arr = img_byte_arr.getvalue()
                
                logger.info(f"이미지 리사이즈 완료: {img.size} → 512x512")
                image_data = img_byte_arr
                
            except ImportError:
                # PIL이 없는 경우 원본 그대로 사용
                logger.warning("PIL(Pillow)이 설치되지 않아 원본 크기로 저장됩니다.")
            except Exception as e:
                # 기타 오류 시 원본 그대로 사용
                logger.error(f"이미지 리사이즈 실패, 원본 사용: {e}")
            
            # Supabase Storage에 업로드
            result = self._supabase.storage.from_(bucket_name).upload(filename, image_data)
            
            if result:
                # 공개 URL 생성
                public_url = self._supabase.storage.from_(bucket_name).get_public_url(filename)
                logger.info(f"✅ Supabase Storage 업로드 완료: {public_url}")
                return public_url
            else:
                logger.error("Supabase Storage 업로드 실패")
                return None
                
        except Exception as e:
            logger.error(f"Supabase Storage 업로드 중 오류: {e}")
            return None

    def _run(self, prompt: str, filename: Optional[str] = None,
             size: str = "1024x1024", quality: str = "medium") -> str:
        try:
            # 매개변수 검증
            if not prompt or prompt.strip() == "":
                return "❌ 오류: prompt 매개변수가 비어있습니다."
            
            # Supabase 설정 확인
            if not self._supabase:
                return "❌ 오류: Supabase가 설정되지 않았습니다. SUPABASE_URL과 SUPABASE_KEY 환경 변수를 설정해주세요."
            
            # 파일명 자동 생성 (충돌 방지용 유니크 suffix 포함)
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                unique = uuid.uuid4().hex[:8]
                filename = f"generated_image_{timestamp}_{unique}.png"

            logger.info(f"[image_gen] prompt='{prompt}', size={size}, quality={quality}")

            # 이미지 생성 (gpt-image-1, b64_json 응답 처리)
            response = self._client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            b64 = response.data[0].b64_json
            image_data = base64.b64decode(b64)

            # Supabase Storage에 업로드
            supabase_url = self._upload_to_supabase(image_data, filename)
            
            if supabase_url:
                # 환경 변수에서 Supabase URL 가져오기
                supabase_url_env = os.getenv("SUPABASE_URL")
                if supabase_url_env:
                    # 환경 변수 값들을 그대로 사용하여 URL 구성
                    return f"![{filename}]({supabase_url_env}/storage/v1/object/public/task-image/{filename})"
                else:
                    # 환경 변수가 없는 경우 원본 URL 사용
                    return f"![{filename}]({supabase_url})"
            else:
                return f"❌ 이미지 업로드 실패: {filename}"

        except Exception as e:
            return _handle_error("image_gen", e)
