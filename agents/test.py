from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List

# 1. Định nghĩa chuẩn mực bằng Pydantic
class Candidate(BaseModel):
    name: str
    email: EmailStr  # Pydantic sẽ tự động kiểm tra định dạng email (@gmail.com)
    years_of_experience: int = Field(ge=0) # Phải là số và lớn hơn hoặc bằng 0
    skills: List[str]

    # 2. (Nâng cao) Tự động sửa lỗi (Coercion/Validation)
    @field_validator("skills", mode="before")
    def parse_skills_string_to_list(cls, value):
        # Nếu LLM lỡ trả về chuỗi thay vì list, Pydantic sẽ tự sửa nó trước khi khởi tạo object
        if isinstance(value, str):
            return [s.strip() for s in value.split(",")]
        return value

# LLM lỡ sinh ra dữ liệu tồi:
bad_llm_output = {
    "name": "Tran B",
    "email": "not-an-email",
    "years_of_experience": -5,
    "skills": "AWS, Triton, RAG"
}

# Chạy thử:
try:
    # Chỉ cần 1 dòng duy nhất để validate và parse!
    candidate = Candidate(**bad_llm_output)
except Exception as e:
    print(e)