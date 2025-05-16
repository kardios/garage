import streamlit as st
import os
import time
import telebot
from google import genai # User-specified import
from openai import OpenAI
from anthropic import Anthropic
# User-specified import for Google types
from google.genai.types import Tool, GenerationConfig, GoogleSearch, GenerateContentConfig
# HarmCategory and HarmBlockThreshold are no longer needed if safety_settings are not used in GenerateContentConfig
from st_copy_to_clipboard import st_copy_to_clipboard

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="CV Generator Pro", page_icon=":briefcase:")

# --- CONFIGURATION ---
# Retrieve API keys from environment variables
PERPLEXITY_API_KEY = os.environ.get('PERPLEXITY_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# Set up Telegram Bot
RECIPIENT_USER_ID = os.environ.get('RECIPIENT_USER_ID')
BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = None
if BOT_TOKEN and RECIPIENT_USER_ID:
    try:
        bot = telebot.TeleBot(BOT_TOKEN)
    except Exception as e:
        st.error(f"Failed to initialize Telegram Bot: {e}")
else:
    st.warning("Telegram Bot token or Recipient User ID not found. Notifications will be disabled.")

# Initialize API Clients
client_perplexity = None
if PERPLEXITY_API_KEY:
    client_perplexity = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
else:
    st.warning("Perplexity API Key not found. Sonar and Deepseek models will be unavailable.")

client_openai = None
if OPENAI_API_KEY:
    client_openai = OpenAI()
else:
    st.warning("OpenAI API Key not found. GPT models will be unavailable.")

client_google_sdk = None # This will hold the genai.Client() instance
if GOOGLE_API_KEY:
    try:
        client_google_sdk = genai.Client(api_key=GOOGLE_API_KEY) # Initialize client
    except Exception as e:
        st.error(f"Failed to initialize Google Gemini Client: {e}")
else:
    st.warning("Google API Key not found. Gemini models will be unavailable.")

client_anthropic = None
if ANTHROPIC_API_KEY:
    client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    st.warning("Anthropic API Key not found. Claude models will be unavailable.")

# Base Generation parameters for Google Gemini
# These will be used directly in GenerateContentConfig
base_generation_config_params = {
    "candidate_count": 1,
    "temperature": 0.5,
}
base_reviewer_generation_config_params = {
    "candidate_count": 1,
    "temperature": 0.3, 
}


# --- PROMPT GENERATION ---
def generate_cv_prompt(individual):
    """Generates the prompt for CV creation."""
    prompt = f"""###Instruction###
Create a comprehensive biography of {individual} detailing the personal background, education, career progression, and other significant appointments or achievements. The biography should be structured as follows:

1.  **NAME**: Full name of the individual.
2.  **GOVERNMENT POSITION**: Current or most recent government position held. (If applicable, otherwise most recent significant professional role).
3.  **COUNTRY**: The official name of the country they serve/worked in or are primarily associated with.
4.  **BORN**: Date of birth.
5.  **AGE**: Current age. Calculate the difference between the current date (May 2025) and the date of birth.
6.  **MARITAL STATUS**: Information on marital status, including spouse and children if applicable. String format.
7.  **EDUCATION**: Chronological list of educational achievements, including institutions attended and degrees or qualifications obtained. Give the breakdown in the form of PERIOD, INSTITUTION, DEGREE.
8.  **CAREER**: Detailed account of the individual's career, including positions held, dates of service, and any promotions or notable responsibilities. This section can be continued as needed (e.g., "Career (cont’d)"). Do not miss the details of all promotions and double hatting positions. Give the breakdown in the form of YEAR and POSITION.
9.  **OTHER APPOINTMENTS**: List of other significant appointments, roles, or contributions outside of their main career path.
10. **AWARDS and DECORATIONS**: List of awards and decorations received.
11. **LANGUAGES**: Languages spoken.
12. **REMARKS**: Any additional noteworthy information or personal achievements, including familial connections to other notable figures if relevant.

This format is designed to provide a clear and detailed overview of an individual's professional and personal life, highlighting their contributions and achievements in a structured manner. Use up-to-date information available up to May 2025.

###Information###
[INFO]

###Biography###"""
    return prompt

# --- STREAMLIT UI ---
st.write("## **CV Generator Pro** :briefcase:")

with st.expander("Click to read documentation", expanded=True):
    st.write("This tool generates draft CVs using various Large Language Models (LLMs).")
    st.write("1.  Enter the name of the individual for whom you want to generate a CV.")
    st.write("2.  Select up to five **CV generation models** from the list. These models are equipped with web search or grounding capabilities to fetch up-to-date information:")
    st.write("    -   **Sonar** (Perplexity - `sonar-pro`)")
    st.write("    -   **Deepseek** (Perplexity - `sonar-reasoning`)")
    st.write("    -   **Gemini** (Google - `gemini-2.0-flash-001`)") 
    st.write("    -   **Optima** (OpenAI - `gpt-4.1`)") 
    st.write("    -   **Claude** (Anthropic - `claude-3-7-sonnet-20250219`)") 
    st.write("3.  If you select more than one generation model, choose one **reasoning model** to synthesize a reconciled CV and highlight discrepancies:")
    st.write("    -   **OpenAI o3** (OpenAI - Advanced reasoning model. *Ensure 'o3' is a valid model ID for your API key.*)")
    st.write("    -   **Gemini 2.5 Pro (Reasoning)** (Google - Powerful alternative for comparison. *Uses 'gemini-2.5-pro-preview-05-06'.*)")
    st.write("4.  Click 'Generate CVs & Synthesize!' to start the process.") # Button text updated
    st.write("5.  Review the generated CVs and the synthesized CV.")

GENERATION_MODELS_OPTIONS = {
    'Sonar': {'client': client_perplexity, 'model_id': 'sonar-pro', 'type': 'perplexity'},
    'Deepseek': {'client': client_perplexity, 'model_id': 'sonar-reasoning', 'type': 'perplexity'},
    'Gemini': {'client': client_google_sdk, 'model_id': 'gemini-2.0-flash-001', 'type': 'google_client_grounding'}, 
    'Optima': {'client': client_openai, 'model_id': 'gpt-4.1', 'type': 'openai_responses_websearch'}, 
    'Claude': {'client': client_anthropic, 'model_id': 'claude-3-7-sonnet-20250219', 'type': 'anthropic_websearch'} 
}

REVIEWER_MODELS_OPTIONS = {
    'OpenAI o3': {'client': client_openai, 'model_id': 'o3', 'type': 'openai_chat'},
    'Gemini 2.5 Pro (Reasoning)': {'client': client_google_sdk, 'model_id': 'gemini-2.5-pro-preview-05-06', 'type': 'google_client'} 
}

# Filter out unavailable models based on API key presence
available_generation_models = [name for name, details in GENERATION_MODELS_OPTIONS.items() if details['client']]
available_reviewer_models = [name for name, details in REVIEWER_MODELS_OPTIONS.items() if details['client']]

if not available_generation_models:
    st.error("No CV generation models are available. Please check your API key configurations in environment variables.")
if not available_reviewer_models:
    st.error("No reviewer models are available. Please check your API key configurations in environment variables.")

Intern_Select = st.multiselect(
    "Which **CV generation models** would you like to deploy? (Select up to 5)",
    options=available_generation_models,
    default=available_generation_models, 
    max_selections=5
)

Reviewer_Name = None
if available_reviewer_models:
    default_reviewer_index = 0 
    if 'Gemini 2.5 Pro (Reasoning)' in available_reviewer_models: 
        default_reviewer_index = available_reviewer_models.index('Gemini 2.5 Pro (Reasoning)')
    elif 'OpenAI o3' in available_reviewer_models: 
         default_reviewer_index = available_reviewer_models.index('OpenAI o3')

    Reviewer_Name = st.selectbox(
        "Which **reasoning model** would you like to deploy for synthesis? (Used if >1 CV generated)", # Text updated
        options=available_reviewer_models,
        index=default_reviewer_index
    )

input_text = st.text_input("Enter the full name of the individual for the CV (e.g., 'Dr. Jane Doe, CEO of Tech Innovations Inc.')")

Customised_Prompt = generate_cv_prompt(input_text) # This is the prompt for individual CV generation

if st.button("Generate CVs & Synthesize! :rocket:"): # Button text updated
    if not input_text.strip():
        st.error("Please enter the name of the individual.")
    elif not Intern_Select:
        st.error("Please select at least one CV generation model.")
    elif len(Intern_Select) > 1 and not Reviewer_Name:
        st.error("Please select a reviewer model when synthesizing multiple CVs.")
    else:
        key_phrase = input_text
        st.divider()
        combined_output_for_copying = ""
        generated_cv_data = {}

        for intern_name in Intern_Select:
            model_details = GENERATION_MODELS_OPTIONS[intern_name]
            if not model_details['client']:
                st.warning(f"{intern_name} is unavailable (client not configured). Skipping.")
                continue

            st.subheader(f"Generating CV with: {intern_name}")
            output_text = "Error: No output generated."
            sources_text = "No specific sources provided by the model for this output."
            start_time = time.time()

            try:
                with st.spinner(f"{intern_name} is drafting the CV... This may take a moment."):
                    if model_details['type'] == 'perplexity':
                        response = model_details['client'].chat.completions.create(
                            model=model_details['model_id'],
                            messages=[{"role": "user", "content": Customised_Prompt}],
                            temperature=0.5
                        )
                        output_text = response.choices[0].message.content
                        if hasattr(response, 'citations') and response.citations:
                            sources_list = [f"- [{c.title}]({c.url})" for c in response.citations if hasattr(c, 'url') and hasattr(c, 'title') and c.url and c.title]
                            if sources_list:
                                sources_text = "Sources:\n" + "\n".join(list(set(sources_list)))

                    elif model_details['type'] == 'google_client_grounding':
                        google_search_tool_instance = Tool(google_search=GoogleSearch())
                        
                        content_config_obj = GenerateContentConfig(
                            tools=[google_search_tool_instance],
                            candidate_count=base_generation_config_params["candidate_count"],
                            temperature=base_generation_config_params["temperature"]
                        )
                        response = model_details['client'].models.generate_content( 
                            model=f"models/{model_details['model_id']}", 
                            contents=Customised_Prompt,
                            config=content_config_obj 
                        )
                        output_text = response.text
                        if response.candidates and response.candidates[0].grounding_metadata:
                            gm = response.candidates[0].grounding_metadata
                            sources_list = []
                            if gm.grounding_chunks:
                                sources_list.extend(
                                    f"- [{chunk.web.title}]({chunk.web.uri})"
                                    for chunk in gm.grounding_chunks
                                    if hasattr(chunk, 'web') and chunk.web and chunk.web.uri and chunk.web.title
                                )
                            if sources_list:
                                sources_text = "Grounding Sources:\n" + "\n".join(list(set(sources_list)))

                    elif model_details['type'] == 'openai_responses_websearch':
                        response = model_details['client'].responses.create(
                            model=model_details['model_id'],
                            input=Customised_Prompt,
                            tools=[{"type": "web_search_preview"}]
                        )
                        output_text = response.output_text

                        found_search_call = False
                        if hasattr(response, 'output') and response.output:
                            for item in response.output:
                                if item.type == "web_search_call":
                                    found_search_call = True
                                    break
                        if found_search_call:
                             sources_text = "Sources: Web search tool was utilized by the model. Detailed citations might be available in the full API response structure."
                        else:
                            sources_text = "Sources: Web search enabled. Information likely integrated. For itemized citations, inspect the full API response."


                    elif model_details['type'] == 'anthropic_websearch':
                        system_prompt_content = ""
                        user_message_content = Customised_Prompt

                        if "###Instruction###" in Customised_Prompt:
                            parts = Customised_Prompt.split("###Instruction###", 1)
                            main_instruction_and_rest = parts[1]
                            if "###Information###" in main_instruction_and_rest:
                                instruction_parts = main_instruction_and_rest.split("###Information###", 1)
                                system_prompt_content = instruction_parts[0].strip()
                                user_message_content = "###Information###" + instruction_parts[1]
                            else:
                                system_prompt_content = main_instruction_and_rest.strip()
                                user_message_content = ""

                        anthropic_messages = [{"role": "user", "content": user_message_content}]

                        response = model_details['client'].messages.create(
                            model=model_details['model_id'],
                            system=system_prompt_content if system_prompt_content else None,
                            messages=anthropic_messages,
                            max_tokens=4096,
                            tools=[{"type": "web_search_20250305", "name": "web_search"}]
                        )
                        parsed_output_text = ""
                        sources_list = []
                        if response.content:
                            for content_block in response.content:
                                if content_block.type == 'text':
                                    parsed_output_text += content_block.text + "\n"
                                    if hasattr(content_block, 'citations') and content_block.citations:
                                        for citation in content_block.citations:
                                            if hasattr(citation, 'url') and hasattr(citation, 'title') and citation.url and citation.title:
                                                sources_list.append(f"- [{citation.title}]({citation.url})")
                        output_text = parsed_output_text.strip()
                        if not output_text and response.stop_reason == 'tool_use':
                            output_text = "Model used web search, but did not provide a direct text response in the first part. This might indicate a multi-step process is expected."

                        if sources_list:
                           sources_text = "Cited Sources:\n" + "\n".join(list(set(sources_list)))

                    end_time = time.time()

                    with st.expander(f"**{intern_name}**'s CV for **{key_phrase}**", expanded=True):
                        st.markdown(output_text)
                        st.markdown("---")
                        st.markdown(sources_text)
                        st.markdown("---")
                        st.write(f"*Time to generate: {round(end_time - start_time, 2)} seconds*")
                        st.write("*Click* :clipboard: *to copy this CV and its sources to clipboard*")
                        st_copy_to_clipboard(f"CV by {intern_name} for {key_phrase}:\n\n{output_text}\n\n{sources_text}")

                    cv_plus_sources = f"<answer_{intern_name}>\n(CV from **{intern_name}**)\n\n{output_text}\n\n{sources_text}\n</answer_{intern_name}>\n\n"
                    combined_output_for_copying += cv_plus_sources
                    generated_cv_data[intern_name] = {'text': output_text, 'sources': sources_text}

                    if bot:
                        try:
                            bot.send_message(chat_id=RECIPIENT_USER_ID, text=f"CV Generator Pro:\n{intern_name} finished drafting CV for {key_phrase}")
                        except Exception as bot_e:
                            st.warning(f"Telegram notification failed for {intern_name}: {bot_e}")
                    st.snow()

            except AttributeError as ae:
                if "object has no attribute 'responses'" in str(ae).lower() and model_details['type'] == 'openai_responses_websearch':
                    st.error(f"An error occurred with {intern_name}: The OpenAI client does not have a '.responses' attribute. This might indicate an issue with the OpenAI library version or the specific client capabilities. Please check your OpenAI library installation and API access for the Responses API. Falling back to standard chat completion for this model.")
                    try:
                        response = model_details['client'].chat.completions.create(
                            model=model_details['model_id'],
                            messages=[{"role": "user", "content": Customised_Prompt}],
                            temperature=0.5
                        )
                        output_text = response.choices[0].message.content
                        sources_text = "Sources: (Fallback to standard chat) Information likely integrated from training data."
                        with st.expander(f"**{intern_name}**'s CV for **{key_phrase}** (Fallback)", expanded=True):
                            st.markdown(output_text)
                            st.markdown("---")
                            st.markdown(sources_text)
                        cv_plus_sources = f"<answer_{intern_name}>\n(CV from **{intern_name}** - Fallback)\n\n{output_text}\n\n{sources_text}\n</answer_{intern_name}>\n\n"
                        combined_output_for_copying += cv_plus_sources
                        generated_cv_data[intern_name] = {'text': output_text, 'sources': sources_text}

                    except Exception as fallback_e:
                        st.error(f"Fallback chat completion also failed for {intern_name}: {fallback_e}")
                        combined_output_for_copying += f"<answer_{intern_name}>\n\nError generating CV with {intern_name} (fallback failed): {fallback_e}\n\n</answer_{intern_name}>\n\n"
                        generated_cv_data[intern_name] = {'text': f"Error (fallback failed): {fallback_e}", 'sources': "N/A due to error."}
                else:
                    st.error(f"An AttributeError occurred with {intern_name}: {ae}")
                    combined_output_for_copying += f"<answer_{intern_name}>\n\nError generating CV with {intern_name}: {ae}\n\n</answer_{intern_name}>\n\n"
                    generated_cv_data[intern_name] = {'text': f"Error: {ae}", 'sources': "N/A due to error."}


            except Exception as e:
                st.error(f"An error occurred with {intern_name}: {e}")
                combined_output_for_copying += f"<answer_{intern_name}>\n\nError generating CV with {intern_name}: {e}\n\n</answer_{intern_name}>\n\n"
                generated_cv_data[intern_name] = {'text': f"Error: {e}", 'sources': "N/A due to error."}


        if combined_output_for_copying:
            st.divider()
            st.subheader("All Generated CVs (for copying)")
            st.write("*Click* :clipboard: *to copy all generated CVs and their sources to clipboard*")
            st_copy_to_clipboard(combined_output_for_copying)
            st.divider()

        successfully_generated_cvs = {k: v for k, v in generated_cv_data.items() if not v['text'].startswith("Error:")}

        if len(successfully_generated_cvs) > 1 and Reviewer_Name:
            st.subheader(f"Synthesized CV by {Reviewer_Name}") # Updated subheader
            reviewer_details = REVIEWER_MODELS_OPTIONS[Reviewer_Name]

            # New prompt for synthesizing and noting discrepancies
            synthesis_prompt_parts = [
                f"You are an expert CV editor. Your task is to synthesize a single, comprehensive, and accurate CV for **{key_phrase}** based on the multiple draft CVs provided below.\n\n"
                "Your goal is to produce a 'Refreshed CV' that is the best possible version, combining all valid information and resolving discrepancies where possible.\n\n"
                "Follow this structure for the Refreshed CV (ensure all 12 sections are present if information is available):\n"
                "1.  **NAME**: Full name of the individual.\n"
                "2.  **GOVERNMENT POSITION**: Current or most recent government position held. (If applicable, otherwise most recent significant professional role).\n"
                "3.  **COUNTRY**: The official name of the country they serve/worked in or are primarily associated with.\n"
                "4.  **BORN**: Date of birth.\n"
                "5.  **AGE**: Current age. (Calculated based on May 2025 and date of birth).\n"
                "6.  **MARITAL STATUS**: Information on marital status, including spouse and children if applicable.\n"
                "7.  **EDUCATION**: Chronological list of educational achievements (PERIOD, INSTITUTION, DEGREE).\n"
                "8.  **CAREER**: Detailed account of the individual's career (YEAR and POSITION).\n"
                "9.  **OTHER APPOINTMENTS**: List of other significant appointments.\n"
                "10. **AWARDS and DECORATIONS**: List of awards and decorations.\n"
                "11. **LANGUAGES**: Languages spoken.\n"
                "12. **REMARKS**: Any additional noteworthy information.\n\n"
                "Instructions for Reconciling and Synthesizing:\n"
                "-   Combine information from all provided CVs to make the Refreshed CV as complete as possible.\n"
                "-   If different CVs provide different information for the same field (e.g., different dates for a job, different university names for the same degree period), try to determine the most likely correct information. If multiple sources agree on one version, prefer that.\n"
                "-   **Crucially, if you encounter conflicting information that cannot be definitively resolved, or if you make a choice between conflicting pieces of information, you MUST indicate this in the Refreshed CV.**\n"
                "    -   For example: \"*2018-2022: Chief Technology Officer, Innovate Corp. (Note: Discrepancy in end year; Sonar reported 2022, Deepseek reported 2023)*\"\n"
                "    -   Or: \"*Education: MSc in Advanced Computing (Source: Gemini) / Master of Science in Computer Engineering (Source: Optima) from Tech University, 2015-2017.*\"\n"
                "    -   Clearly state the source of conflicting information (e.g., 'Sonar stated X, while Claude stated Y').\n"
                "-   Ensure dates, positions, and achievements are accurately represented based on the consensus or noted discrepancies.\n"
                "-   If one CV provides more detail for a specific role or achievement, incorporate that richer detail.\n"
                "-   Omit any redundant information if multiple CVs state the exact same fact.\n"
                "-   The final output should be ONLY the complete 'Refreshed CV' with inline notes for discrepancies. Do not add any other commentary before or after the CV.\n\n"
                f"The draft CVs were generated by the following models: {', '.join(successfully_generated_cvs.keys())}.\n"
                "The CVs are contained in the tags below. When noting a discrepancy, refer to the model by its name (e.g., **Sonar**, **Gemini**).\n\n"
                "Here are the draft CVs:\n\n"
            ]
            
            for name, data in successfully_generated_cvs.items():
                synthesis_prompt_parts.append(f"<answer_{name}>\n(CV from **{name}**)\n\n--- CV Start ---\n{data['text']}\n--- CV End ---\n\n--- Sources listed by {name} ---\n{data['sources']}\n--- Sources End ---\n\n</answer_{name}>\n\n")
            
            final_synthesis_prompt = "".join(synthesis_prompt_parts)

            try:
                with st.spinner(f"{Reviewer_Name} is synthesizing the CV... This might take some time."):
                    start_time = time.time()
                    synthesized_cv_text = "Error: No synthesized CV generated."

                    if reviewer_details['type'] == 'openai_chat':
                        response = reviewer_details['client'].chat.completions.create(
                            model=reviewer_details['model_id'],
                            messages=[{"role": "user", "content": final_synthesis_prompt}]
                        )
                        synthesized_cv_text = response.choices[0].message.content

                    elif reviewer_details['type'] == 'google_client':
                        content_config_obj_reviewer = GenerateContentConfig(
                            candidate_count=base_reviewer_generation_config_params["candidate_count"],
                            temperature=base_reviewer_generation_config_params["temperature"]
                        )
                        response = reviewer_details['client'].models.generate_content(
                            model=f"models/{reviewer_details['model_id']}",
                            contents=final_synthesis_prompt,
                            config=content_config_obj_reviewer
                        )
                        synthesized_cv_text = response.text

                    end_time = time.time()

                    with st.expander(f"**{Reviewer_Name}**'s Synthesized CV for **{key_phrase}**", expanded=True):
                        st.markdown(synthesized_cv_text)
                        st.markdown("---")
                        st.write(f"*Time to synthesize: {round(end_time - start_time, 2)} seconds*")
                        st.write("*Click* :clipboard: *to copy synthesized CV to clipboard*")
                        st_copy_to_clipboard(synthesized_cv_text)
                    
                    if bot:
                        try:
                            bot.send_message(chat_id=RECIPIENT_USER_ID, text=f"CV Generator Pro:\n{Reviewer_Name} finished synthesizing CV for {key_phrase}")
                        except Exception as bot_e:
                             st.warning(f"Telegram notification failed for {Reviewer_Name}: {bot_e}")
                    st.balloons()

            except Exception as e:
                st.error(f"An error occurred with {Reviewer_Name} during synthesis: {e}")

        elif len(successfully_generated_cvs) <= 1 and combined_output_for_copying:
            st.info("One or fewer CVs were successfully generated, so no synthesis will be performed.")
        elif not combined_output_for_copying:
             st.error("No CVs were generated. Please check model selections and API keys.")
