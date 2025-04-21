"""
Templates and prompts for the Medical Scribe application.
"""

import json
import os
from datetime import datetime

# Path to store custom template
TEMPLATE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config"
)
TEMPLATE_CONFIG_FILE = os.path.join(TEMPLATE_CONFIG_PATH, "template_config.json")

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
    "DO NOT make up or infer any information that is not explicitly discussed in the transcript. "
    "If specific information is not mentioned in the conversation, leave that field blank or mark it as 'Not discussed'. "
    "Only include information that was actually discussed in the conversation. "
    "Do not output any explanations, instructions, or additional text - ONLY output the formatted medical note."
)


def load_custom_template():
    """Loads the custom template if it exists, otherwise returns the default template"""
    try:
        if os.path.exists(TEMPLATE_CONFIG_FILE):
            with open(TEMPLATE_CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("template", DEFAULT_TEMPLATE)
    except Exception as e:
        print(f"Error loading custom template: {e}")

    return DEFAULT_TEMPLATE


def save_custom_template(template_text):
    """Saves a custom template as the default"""
    try:
        # Create config directory if it doesn't exist
        if not os.path.exists(TEMPLATE_CONFIG_PATH):
            os.makedirs(TEMPLATE_CONFIG_PATH)

        config = {"template": template_text}
        with open(TEMPLATE_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving custom template: {e}")
        return False


# Function to get template with today's date
def get_template_with_date():
    """Returns the template with today's date inserted"""
    today = datetime.now().strftime("%Y-%m-%d")
    template = load_custom_template()
    return template.format(today_date=today)
