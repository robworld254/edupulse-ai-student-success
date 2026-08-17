KABARAK_SCHOOLS = [
    "School of Business & Economics",
    "School of Education, Humanities & Social Sciences",
    "School of Law",
    "School of Medicine & Health Sciences",
    "School of Music & Media",
    "School of Pharmacy",
    "School of Science, Engineering & Technology",
]

FEE_STATUS = {
    "Up to date": (1, 0),
    "Partial / pending": (0, 0),
    "Outstanding balance": (0, 1),
}

DEMO_PROFILES = {
    "Custom": None,
    "Strong Progress": {
        "units_registered": 6,
        "units_passed": 6,
        "average_mark_pct": 72,
        "assessments_completed": 8,
        "fee_status": "Up to date",
        "scholarship": "No",
    },
    "Monitor": {
        "units_registered": 6,
        "units_passed": 4,
        "average_mark_pct": 65,
        "assessments_completed": 6,
        "fee_status": "Up to date",
        "scholarship": "No",
    },
    "Higher Support Need": {
        "units_registered": 6,
        "units_passed": 2,
        "average_mark_pct": 41,
        "assessments_completed": 3,
        "fee_status": "Outstanding balance",
        "scholarship": "No",
    },
}

SUPPORT_SERVICES = {
    "academic": "Academic Adviser / School Administrator",
    "finance": "Student Finance",
    "wellbeing": "Guidance & Counselling / Dean of Students",
}
