"""
Templates and prompts for the Medical Scribe application.
"""

from datetime import datetime

# Default SOAP note template
DEFAULT_TEMPLATE = """## Patient Info
**Name:** [Name] | **DOB:** [DOB] | **Date:** {today_date}

## S: Subjective
- **CC:** [Chief complaint]
- **HPI:** [Brief history of present illness]
- **Current Meds:** [Medications]
- **Allergies:** [Allergies]
- **PMH:** [Relevant past medical history]
- **ROS:** [Pertinent positive/negative findings]

## O: Objective
- **Vitals:** T [temp] | BP [BP] | HR [HR] | RR [RR] | SpO2 [O2] | Pain [0-10]
- **Physical Exam:** [Key findings by system]
- **Labs/Studies:** [Relevant results]

## A: Assessment
1. [Primary diagnosis/problem]
2. [Secondary diagnosis/problem]

## P: Plan
- **Diagnostics:** [Ordered tests]
- **Treatment:** [Medications, therapies]
- **Patient Education:** [Instructions given]
- **Follow-up:** [When and circumstances]"""

# System prompt for clinical note generation
SYSTEM_PROMPT = (
    "You are a medical assistant helping to format clinical notes. "
    "Create a structured and professional medical note based on the provided conversation transcript. "
    "Format it with clear sections for patient information, chief complaint, history, "
    "assessment, and plan. Use medical terminology appropriately. "
    "Follow the structure of the template provided in the user message. "
    "The current date is automatically inserted. "
    "Do not output any explanations, instructions, or additional text - ONLY output the formatted medical note."
)


# Function to get template with today's date
def get_template_with_date():
    """Returns the template with today's date inserted"""
    today = datetime.now().strftime("%Y-%m-%d")
    return DEFAULT_TEMPLATE.format(today_date=today)
