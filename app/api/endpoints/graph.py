from fastapi import APIRouter
from app.db.data_load import getDataFrameOriginal

router = APIRouter()


@router.get("/avg-salary-by-location")
def get_avg_salary_by_location():
    df_data= getDataFrameOriginal()
    avg_salary_by_location = df_data.groupby('company_location', as_index=False)['salary_in_usd'].mean()
    return {
        "locations": avg_salary_by_location['company_location'].tolist(),
        "values": avg_salary_by_location['salary_in_usd'].tolist(),
        "hover_names": avg_salary_by_location['company_location'].tolist()
    }

@router.get("/most_popular_roles")
def get_most_popular_roles():
    df_data= getDataFrameOriginal()
    most_popular_roles = df_data["job_title"].value_counts().head(10)
    return {
        "job_titles": most_popular_roles.index.tolist(),
        "values": most_popular_roles.values.tolist(),
        "hover_names": most_popular_roles.index.tolist()
    }

@router.get("/work_setting_counts")
def get_work_setting_counts():
    df_data = getDataFrameOriginal()
    work_setting_counts = df_data['work_setting'].value_counts()
    return {
        "work_settings": work_setting_counts.index.tolist(),
        "values": work_setting_counts.values.tolist()
    }

@router.get("/experience_level_counts")
def get_experience_level_counts():
    df_data = getDataFrameOriginal()
    experience_level_counts = df_data.groupby('experience_level', as_index=False)['salary_in_usd'].count().sort_values(by='salary_in_usd', ascending=False).head(10)
    return {
        "experience_levels": experience_level_counts['experience_level'].tolist(),
        "values": experience_level_counts['salary_in_usd'].tolist()
    } 

@router.get("/salary_distribution_by_company_size")
def get_salary_distribution_by_company_size():
    df_data = getDataFrameOriginal()
    return {
        "company_size": df_data["company_size"].tolist(),
        "salary_in_usd": df_data["salary_in_usd"].tolist()
    }

@router.get("/salary_trends_by_job_category")
def get_salary_trends_by_job_category():
    df_data = getDataFrameOriginal()
    grouped_data = df_data.groupby(['work_year', 'job_category'])['salary'].mean().reset_index()
    return {
        "work_year": grouped_data["work_year"].tolist(),
        "job_category": grouped_data["job_category"].tolist(),
        "salary": grouped_data["salary"].tolist()
    }