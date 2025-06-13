from openai import OpenAI
import streamlit as st
import json
import re
import uuid
import requests
from typing import Dict, List, Optional

# Page config
st.set_page_config(
    page_title="Smart Complaint Assistant",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Smart Complaint Assistant")

# Configuration
API_BASE_URL = "http://localhost:5000"  # Change this to your API server URL

# Initialize OpenAI client
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("Please add your OpenAI API key to the secrets configuration")
    st.stop()

# Session state initialization
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "complaint_data" not in st.session_state:
    st.session_state.complaint_data = {
        "complaint_topic": None,
        "name": None,
        "phone_number": None,
        "email": None,
        "complaint_details": None
    }

if "extraction_mode" not in st.session_state:
    st.session_state.extraction_mode = False

if "missing_fields" not in st.session_state:
    st.session_state.missing_fields = []

if "complaint_session_id" not in st.session_state:
    st.session_state.complaint_session_id = None

if "lookup_mode" not in st.session_state:
    st.session_state.lookup_mode = False

if "details_collection_mode" not in st.session_state:
    st.session_state.details_collection_mode = False

if "needs_more_info" not in st.session_state:
    st.session_state.needs_more_info = False


# API functions
def create_complaint_via_api(complaint_data: Dict) -> Dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/complaints",
            json=complaint_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        if response.status_code == 201:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


def get_complaint_via_api(complaint_id: str) -> Dict:
    try:
        response = requests.get(
            f"{API_BASE_URL}/complaints/{complaint_id}",
            timeout=30
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 404:
            return {"success": False, "error": "Complaint not found"}
        else:
            return {"success": False, "error": response.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# Validation functions
def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    digits_only = re.sub(r'\D', '', phone)
    return len(digits_only) >= 10 and len(digits_only) <= 15


def get_missing_fields(data: Dict) -> List[str]:
    """Get list of missing required fields in the correct order"""
    required_fields = ['name', 'phone_number', 'email', 'complaint_details']
    missing = []

    for field in required_fields:
        value = data.get(field)
        if not value or str(value).strip() == '' or str(value).lower() in ['null', 'none']:
            missing.append(field)

    return missing


def extract_complaint_details(text: str) -> Dict:
    extraction_prompt = f"""
You are an AI assistant that extracts complaint information from user text. 
Extract the following information if present in the text:

1. Complaint topic (ONLY the general category/type of issue - e.g., "delayed delivery", "broken product", "billing issue")
2. Name (person's full name)
3. Phone number (any format)
4. Email address
5. Complaint details (ONLY specific details about what happened - dates, order numbers, specific problems, etc.)

CRITICAL DISTINCTION:
- Complaint topic = WHAT TYPE of problem (e.g., "delayed delivery", "defective product")
- Complaint details = SPECIFIC INFORMATION about the problem (e.g., "Order #12345 was supposed to arrive May 1st but never came", "The screen is cracked and won't turn on")

EXAMPLES:
Text: "I want to complain about a delayed delivery"
→ complaint_topic: "delayed delivery", complaint_details: null

Text: "My order #12345 was supposed to arrive on May 1 but is still not here"
→ complaint_topic: "delayed delivery", complaint_details: "Order #12345 was supposed to arrive on May 1 but is still not here"

Text: "I have a problem with my phone. The screen cracked after one day."
→ complaint_topic: "defective product", complaint_details: "The screen cracked after one day"

EXTRACTION RULES:
- Only extract information explicitly mentioned in the text
- If information is not present, return null for that field
- Do NOT put the same information in both complaint_topic and complaint_details
- If only general complaint mentioned without specifics, complaint_details should be null
- Set needs_more_info to true if any required field (name, phone_number, email, complaint_details) is missing

Text to analyze: "{text}"

Return JSON in this exact format:
{{
    "complaint_topic": "extracted complaint topic or null",
    "name": "extracted name or null",
    "phone_number": "extracted phone or null", 
    "email": "extracted email or null",
    "complaint_details": "extracted complaint details or null",
    "needs_more_info": true/false,
    "missing_fields": ["list of missing required fields"]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts complaint information from text. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": extraction_prompt
                }
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise Exception("Could not parse JSON response")

    except Exception as e:
        st.error(f"Error extracting details: {e}")
        return {
            "complaint_topic": None,
            "name": None,
            "phone_number": None,
            "email": None,
            "complaint_details": None,
            "needs_more_info": True,
            "missing_fields": ["name", "phone_number", "email", "complaint_details"]
        }


def generate_follow_up_question(missing_fields: List[str], current_data: Dict) -> str:
    if not missing_fields:
        return "Thank you for providing all the information!"

    field = missing_fields[0]

    if field == 'name':
        return "Please provide your name."
    elif field == 'phone_number':
        name = current_data.get('name', '')
        if name:
            return f"Thank you, {name}. What is your phone number?"
        return "What is your phone number?"
    elif field == 'email':
        return "Got it. Please provide your email address."
    elif field == 'complaint_details':
        topic = current_data.get('complaint_topic', 'issue')
        return f"Thanks for providing your information. Can you share more details about the {topic}?"

    return "Could you provide the missing information?"


def analyze_complaint_completeness(complaint_details: str) -> Dict:
    analysis_prompt = f"""
You are analyzing a customer complaint to determine if it contains sufficient detail for proper handling.

Complaint: "{complaint_details}"

Analyze the complaint and determine:
1. Should we ask for more details?

Consider these factors:
- Clear description of the problem
- Impact on the customer
- Expected resolution or what went wrong

Return your analysis as JSON:
{{
    "is_sufficient": true/false,
    "missing_info": ["list", "of", "missing", "details"],
    "follow_up_question": "Can you share more details about the issue?" or null if sufficient
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a customer service quality analyst. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback analysis
                return {
                    "is_sufficient": len(complaint_details.split()) >= 10,
                    "missing_info": [],
                    "follow_up_question": None
                }
    except Exception as e:
        st.error(f"Error analyzing complaint: {e}")
        # Simple fallback - consider detailed if more than 10 words
        return {
            "is_sufficient": len(complaint_details.split()) >= 10,
            "missing_info": [],
            "follow_up_question": None
        }


def detect_complaint_lookup(text: str) -> Optional[str]:

    uuid_pattern = [r'^([a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{12})']

    for pattern in uuid_pattern:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def format_complaint_details(complaint_data: Dict) -> str:
    """Format complaint details for display"""
    formatted = f"""## 📋 Complaint Details

**Complaint ID:** {complaint_data.get('complaint_id', 'N/A')}

**Name:** {complaint_data.get('name', 'N/A')}

**Phone:** {complaint_data.get('phone_number', 'N/A')}

**Email:** {complaint_data.get('email', 'N/A')}

**Complaint Details:** {complaint_data.get('complaint_details', 'N/A')}

**Created At:** {complaint_data.get('Created At', 'N/A')}
"""
    return formatted


# Sidebar - API status and controls
with st.sidebar:
    st.header("🔧 System Status")

    # Check API health
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Server Connected")
    else:
        st.error("❌ API Server Offline")
        st.warning("Please ensure the Flask API server is running on http://localhost:5000")

    st.markdown("---")

    # Reset button
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.complaint_data = {
            "complaint_topic": None,
            "name": None,
            "phone_number": None,
            "email": None,
            "complaint_details": None
        }
        st.session_state.extraction_mode = False
        st.session_state.missing_fields = []
        st.session_state.complaint_session_id = None
        st.session_state.lookup_mode = False
        st.session_state.details_collection_mode = False
        st.session_state.needs_more_info = False
        st.rerun()

    st.markdown("---")
    st.markdown("**Current Mode:**")
    if st.session_state.lookup_mode:
        st.info("🔍 Complaint Lookup")
    elif st.session_state.extraction_mode:
        st.info("📝 Filing Complaint")
    elif st.session_state.details_collection_mode:
        st.info("📋 Collecting More Details")
    else:
        st.info("💬 General Chat")

    # Debug info
    if st.session_state.extraction_mode:
        st.markdown("---")
        st.markdown("**Debug Info:**")
        st.json(st.session_state.complaint_data)
        st.write("Missing fields:", st.session_state.missing_fields)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Check if user is trying to look up a complaint
        complaint_id = detect_complaint_lookup(prompt)

        if complaint_id and not st.session_state.extraction_mode:
            # User wants to look up a complaint
            st.session_state.lookup_mode = True

            if not api_healthy:
                response = "❌ I'm sorry, but I can't retrieve complaint details right now because the API server is offline. Please try again later or contact our support team directly."
            else:
                with st.spinner("Looking up complaint details..."):
                    result = get_complaint_via_api(complaint_id)

                if result["success"]:
                    response = "Here are the complaint details:\n\n"
                    response += format_complaint_details(result["data"])
                else:
                    if "not found" in str(result["error"]).lower():
                        response = f"❌ I couldn't find a complaint with ID '{complaint_id}'. Please check the ID and try again."
                    else:
                        response = f"❌ Error retrieving complaint: {result['error']}"

            st.markdown(response)
            st.session_state.lookup_mode = False

        elif not st.session_state.extraction_mode:
            # Check if this might be a complaint (look for complaint-related keywords)
            complaint_keywords = ['complaint', 'problem', 'issue', 'broken', 'defective', 'delayed', 'wrong', 'damaged',
                                  'missing', 'error', 'fault', 'trouble']
            is_complaint = any(keyword in prompt.lower() for keyword in complaint_keywords)

            if is_complaint:
                # Start complaint extraction mode
                st.session_state.extraction_mode = True
                st.session_state.complaint_session_id = str(uuid.uuid4())

                # Extract details from the initial complaint
                extracted = extract_complaint_details(prompt)

                # Update complaint data
                for field in ['complaint_topic', 'name', 'phone_number', 'email', 'complaint_details']:
                    value = extracted.get(field)
                    if value and str(value).lower() not in ['null', 'none']:
                        st.session_state.complaint_data[field] = value

                # Update extraction state
                st.session_state.needs_more_info = extracted.get('needs_more_info', True)
                st.session_state.missing_fields = get_missing_fields(st.session_state.complaint_data)

                if not st.session_state.missing_fields:
                    # All information collected - try to create complaint
                    if not api_healthy:
                        response = "❌ I'm sorry, but I can't file your complaint right now because our system is offline. Please try again later or contact our support team directly."
                        st.session_state.extraction_mode = False
                    else:
                        with st.spinner("Filing your complaint..."):
                            # Prepare data for API
                            api_data = {
                                "name": st.session_state.complaint_data["name"],
                                "phone_number": st.session_state.complaint_data["phone_number"],
                                "email": st.session_state.complaint_data["email"],
                                "complaint_details": st.session_state.complaint_data["complaint_details"]
                            }

                            result = create_complaint_via_api(api_data)

                        if result["success"]:
                            complaint_id = result["data"]["complaint_id"]
                            response = f"✅ **Your complaint has been successfully filed!**\n\n"
                            response += f"**Complaint ID:** {complaint_id}\n\n"
                            response += "Please save this ID for future reference. Our customer service team will contact you within 24-48 hours.\n\n"
                            response += "Is there anything else I can help you with?"
                        else:
                            response = f"❌ I'm sorry, but there was an error filing your complaint: {result['error']}\n\n"
                            response += "Please try again or contact our support team directly."

                        # Reset extraction mode
                        st.session_state.extraction_mode = False
                else:
                    # Generate follow-up question for missing information
                    follow_up = generate_follow_up_question(st.session_state.missing_fields,
                                                            st.session_state.complaint_data)
                    response = f"I understand you have a complaint. Let me help you file it properly.\n\n"
                    if st.session_state.complaint_data.get('complaint_details'):
                        response += f"I've noted: *{st.session_state.complaint_data.get('complaint_details')}*\n\n"
                    response += f"{follow_up}"

                st.markdown(response)
            else:
                # Regular chat mode
                response = "Hello! I'm here to help you with complaints and inquiries. You can:\n\n"
                response += "• **File a complaint**: Just describe your issue and I'll help collect the necessary details\n"
                response += "• **Look up a complaint**: Provide your complaint ID to check status\n"
                response += "• **Ask questions**: I'm here to help with any other inquiries\n\n"
                response += "How can I assist you today?"
                st.markdown(response)

        elif st.session_state.extraction_mode:
            # We're in extraction mode, collect the missing information step by step
            current_missing = st.session_state.missing_fields[0] if st.session_state.missing_fields else None

            if current_missing == 'name':
                st.session_state.complaint_data['name'] = prompt.strip()
            elif current_missing == 'phone_number':
                st.session_state.complaint_data['phone_number'] = prompt.strip()
            elif current_missing == 'email':
                st.session_state.complaint_data['email'] = prompt.strip()
            elif current_missing == 'complaint_details':
                # Append additional details to existing complaint
                existing_details = st.session_state.complaint_data.get('complaint_details', '')
                if existing_details:
                    st.session_state.complaint_data['complaint_details'] = f"{existing_details} {prompt.strip()}"
                else:
                    st.session_state.complaint_data['complaint_details'] = prompt.strip()

            # Check for remaining missing fields
            st.session_state.missing_fields = get_missing_fields(st.session_state.complaint_data)

            if not st.session_state.missing_fields:
                # All information collected - validate and create complaint
                data = st.session_state.complaint_data

                # Validate email and phone
                validation_errors = []
                if data['email'] and not validate_email(data['email']):
                    validation_errors.append("Please provide a valid email address.")
                if data['phone_number'] and not validate_phone(data['phone_number']):
                    validation_errors.append("Please provide a valid phone number.")

                if validation_errors:
                    response = "I found some issues with the information provided:\n\n"
                    for error in validation_errors:
                        response += f"❌ {error}\n"
                    response += "\nCould you please provide the correct information?"
                    st.markdown(response)
                else:
                    # Create complaint via API
                    if not api_healthy:
                        response = "❌ I'm sorry, but I can't file your complaint right now because our system is offline. Please try again later or contact our support team directly."
                        st.session_state.extraction_mode = False
                    else:
                        with st.spinner("Filing your complaint..."):
                            # Prepare data for API
                            api_data = {
                                "name": data["name"],
                                "phone_number": data["phone_number"],
                                "email": data["email"],
                                "complaint_details": data["complaint_details"]
                            }

                            result = create_complaint_via_api(api_data)

                        if result["success"]:
                            complaint_id = result["data"]["complaint_id"]
                            response = f"✅ **Your complaint has been successfully filed!**\n\n"
                            response += f"**Complaint ID:** {complaint_id}\n\n"
                            response += "Please save this ID for future reference. Our customer service team will contact you within 24-48 hours."
                        else:
                            response = f"❌ I'm sorry, but there was an error filing your complaint: {result['error']}\n\nPlease try again or contact our support team directly."

                        # Reset extraction mode
                        st.session_state.extraction_mode = False

                    st.markdown(response)
            else:
                # Still missing fields, ask for next one
                follow_up = generate_follow_up_question(st.session_state.missing_fields,
                                                        st.session_state.complaint_data)
                response = follow_up
                st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})