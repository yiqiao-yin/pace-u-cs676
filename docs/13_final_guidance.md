# Presentation Guidance

## Overview

Please read the following for presentation guidance. This is a Zoom session. You will need to share your screen and present your chatbot during your assigned time slot only.

## Full-Stack Application Considerations

### Front-end (10%)

- **Questions You May Encounter:**
  - What framework are you using? Terminal? Streamlit app? Flask app?
  - There is no preference as long as it makes sense for your app.
  
### Back-end (20%)

- Most participants will likely use Python. Specify the Python version and packaging:
  - Presence of a `.py` script or multiple `.py` scripts.
  - A `requirements.txt` file listing all used packages.
  
- **Questions You May Encounter:**
  - Where is your `requirements.txt` file?
  - Which package did you use to call the HuggingFace model?
  - Why do you use XYZ function in your script?

### API (30%)

- As discussed, FastAPI is the easiest workaround:
  - Need a `.py` script containing your API design code.
  
- Services like Together.AI, OpenAI, HuggingFace are also acceptable. In these cases, your frontend script will directly invoke them.
  
- **Questions You May Encounter:**
  - What is the contract of this API?
  - What is the data payload requirement of this API?

### Bonus (10%)

- Utilizing AWS API Gateway earns a bonus:
  - We have discussed standing up APIs on AWS and exposing endpoint URLs to the public.

- **Questions You May Encounter:**
  - Have you done a Postman test?
  - Have you done a curl test?

### System Design (40%)

- Required to create a user journey: 
  - Example: "I want to make a prediction of housing price using a logistic regression model."

- Chatbot needs to be intelligent enough to discern:
  - "Housing price is a continuous variable. Shall I turn this variable into different bins for you? Or perhaps you meant linear regression model?"
  
- Depending on user inputs:
  - If the user confirms linear regression, the chatbot calls the linear regression API.
  - If the user opts for bin conversion and classification, the chatbot calls the logistic regression API.
  
- Implement an AI judge:
  - Should be a separate API call.
  - Recognize R-square, linear coefficients, t-stat, hypothesis testing, etc.

- **Questions You May Encounter:**
  - Where is your design diagram?
  - Walk me through your design diagram.
  - You had XYZ module built, why did you build it?

## Total Schema for Grading:

- Front-end (10%)
- Back-end (20%) + Bonus (10%)
- API (30%)
- System Design (40%)
- **Total points available: 110%**