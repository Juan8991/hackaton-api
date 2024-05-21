class Profile:
    def __init__(self):
        """ self._work_year = None
        self._job_title = None
        self._job_category = None
        self._salary_currency = None
        self._salary = None
        self._salary_in_usd = None
        self._employee_residence = None
        self._experience_level = None
        self._employment_type = None
        self._work_setting = None
        self._company_location = None
        self._company_size = None """
        self._employee_residence = None
        self._experience_level = None
        self._employment_type = None
        self._work_setting = None
        self._job_category = None

    """ # Work Year property
    @property
    def work_year(self):
        return self._work_year

    @work_year.setter
    def work_year(self, value):
        self._work_year = value

    # Job Title property
    @property
    def job_title(self):
        return self._job_title

    @job_title.setter
    def job_title(self, value):
        self._job_title = value """

    # Similar properties for other attributes...

    """ # Example for one more attribute
    @property
    def salary(self):
        return self._salary """

 