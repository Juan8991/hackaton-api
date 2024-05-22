from pydantic import BaseModel

class Profilepy(BaseModel):
    employee_residence: str
    experience_level: str
    employment_type: str
    work_setting: str
    job_category: str
    company_size: str