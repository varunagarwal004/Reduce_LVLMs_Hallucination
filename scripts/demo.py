import os
from typing import Optional

import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image

from lvlm_models.base_lvlm import BaseLVLMModel
from lvlm_models.chain_of_thought import ChainOfThoughtLVLM
from lvlm_models.gemma3 import Gemma3Model
from lvlm_models.llava import LlavaModel
from lvlm_models.openai import OpenAIVisionModel
from lvlm_models.self_verification import SelfVerificationLVLM

# Load environment variables
load_dotenv()


def setup_model(model_name: str) -> BaseLVLMModel:
    """Initialize and return the specified model."""
    load_dotenv()

    system_prompt = (
        "<role>\n"
        "You are a helpful visual assistant that can accurately identify "
        "objects in images.\n"
        "</role>"
    )

    prompt_prefix = (
        "Look at the image carefully and determine whether the object mentioned "
        "in the question is present in the image."
    )

    prompt_suffix = "Answer with only 'yes' or 'no', nothing else."

    if model_name == "llava":
        return LlavaModel(
            model_name="llava-hf/llava-1.5-7b-hf",
            hf_token=os.getenv("HF_TOKEN"),
            system_prompt=system_prompt,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
        )
    elif model_name == "gemma3":
        return Gemma3Model(
            model_name="google/gemma-3-4b-it",
            hf_token=os.getenv("HF_TOKEN"),
            system_prompt=system_prompt,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
        )
    elif model_name == "openai":
        return OpenAIVisionModel(
            model_name="gpt-4.1-mini-2025-04-14",
            api_key=os.getenv("OPENAI_API_KEY"),
            system_prompt=system_prompt,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")


def cleanup_model(
    model: Optional[BaseLVLMModel] = None,
    sv_model: Optional[SelfVerificationLVLM] = None,
    cot_model: Optional[ChainOfThoughtLVLM] = None,
):
    """Clean up model resources to free memory."""
    if model:
        del model
    if sv_model:
        del sv_model
    if cot_model:
        del cot_model

    torch.cuda.empty_cache()
    import gc

    gc.collect()


def process_single_model(
    model_choice, reasoning_approach, image, prompt, options, cot_strategy, sv_strategy
):
    """Process a single model with the given parameters and return results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: {device}")

    base_model = None
    cot_model = None
    sv_model = None

    try:
        with st.spinner(f"Loading {model_choice} model..."):
            base_model = setup_model(model_choice)

        with st.spinner(f"Processing with {model_choice}..."):
            if reasoning_approach == "Direct Response":
                response = base_model.generate_response(
                    image=image, question=prompt, options=options, use_prefix_suffix=True
                )
                return {"Response": response}

            elif reasoning_approach == "Chain of Thought":
                cot_model = ChainOfThoughtLVLM(base_model=base_model, cot_strategy=cot_strategy)
                reasoning, final_answer = cot_model.generate_response_cot(
                    image=image, question=prompt, options=options
                )
                return {"Reasoning": reasoning, "Final Answer": final_answer}

            elif reasoning_approach == "Self Verification":
                sv_model = SelfVerificationLVLM(
                    base_model=base_model, reasoning_strategy=sv_strategy
                )
                initial_reasoning, verification, final_answer = (
                    sv_model.generate_response_with_verification(
                        image=image, question=prompt, options=options
                    )
                )
                return {
                    "Initial Reasoning": initial_reasoning,
                    "Verification": verification,
                    "Final Answer": final_answer,
                }
    finally:
        cleanup_model(model=base_model, cot_model=cot_model, sv_model=sv_model)


def single_model_tab():
    """UI for running a single model."""
    model_options = {
        "llava": "LLaVA-1.5-7B",
        "gemma3": "Gemma-3-4B",
        "openai": "GPT-4.1-Mini",
    }
    model_choice = st.selectbox(
        "Select Model", model_options.keys(), format_func=lambda x: model_options[x]
    )

    reasoning_approach = st.radio(
        "Select Reasoning Approach", ["Direct Response", "Chain of Thought", "Self Verification"]
    )

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="single_model_uploader"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    prompt = st.text_area("Enter your prompt", height=100)

    use_options = st.radio(
        label="Select multiple choice options",
        options=["No options", "Add multiple choice options", "Yes/No"],
        index=0,
    )
    options = []

    if use_options == "Add multiple choice options":
        sv_strategy = "visual_reasoning"
        cot_strategy = "visual_puzzle"
        col1, col2 = st.columns(2)
        with col1:
            option_a = st.text_input("Option A")
            option_c = st.text_input("Option C")
        with col2:
            option_b = st.text_input("Option B")
            option_d = st.text_input("Option D")

        options = [option_a, option_b, option_c, option_d]
        options = [opt for opt in options if opt.strip()]
        if not options:
            options = None
    elif use_options == "Yes/No":
        sv_strategy = "yes_no_object"
        cot_strategy = "yes_no_object"
        options = ["Yes", "No"]
    else:
        sv_strategy = "detailed"
        cot_strategy = "detailed"
        options = None

    if st.button("Process", key="single_process"):
        if uploaded_file is None:
            st.error("Please upload an image.")
            return

        if not prompt:
            st.error("Please enter a prompt.")
            return

        try:
            image = Image.open(uploaded_file)

            results = process_single_model(
                model_choice, reasoning_approach, image, prompt, options, cot_strategy, sv_strategy
            )

            for title, content in results.items():
                st.subheader(title)
                st.write(content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def compare_models_tab():
    """UI for comparing all three models."""
    reasoning_approach = st.radio(
        "Select Reasoning Approach",
        ["Chain of Thought", "Self Verification"],
        key="compare_reasoning",
    )

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="compare_uploader"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

    prompt = st.text_area("Enter your prompt", height=100, key="compare_prompt")

    use_options = st.radio(
        label="Select options",
        options=["No options", "Add multiple choice options", "Yes/No"],
        index=0,
    )
    options = []

    if use_options == "Add multiple choice options":
        sv_strategy = "visual_reasoning"
        cot_strategy = "visual_puzzle"
        col1, col2 = st.columns(2)
        with col1:
            option_a = st.text_input("Option A")
            option_c = st.text_input("Option C")
        with col2:
            option_b = st.text_input("Option B")
            option_d = st.text_input("Option D")

        options = [option_a, option_b, option_c, option_d]
        options = [opt for opt in options if opt.strip()]
        if not options:
            options = None
    elif use_options == "Yes/No":
        sv_strategy = "yes_no_object"
        cot_strategy = "yes_no_object"
        options = ["Yes", "No"]
    else:
        sv_strategy = "detailed"
        cot_strategy = "detailed"
        options = None

    if st.button("Compare All Models", key="compare_process"):
        if uploaded_file is None:
            st.error("Please upload an image.")
            return

        if not prompt:
            st.error("Please enter a prompt.")
            return

        try:
            image = Image.open(uploaded_file)

            model_options = {
                "llava": "LLaVA-1.5-7B",
                "gemma3": "Gemma-3-4B",
                "openai": "GPT-4.1-Mini",
            }

            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"Using device: {device}")

            # Create tabs for all models
            model_tabs = st.tabs(list(model_options.values()))

            for i, (model_name, model_display_name) in enumerate(model_options.items()):
                with model_tabs[i]:
                    base_model = None
                    cot_model = None
                    sv_model = None

                    try:
                        with st.spinner(f"Loading {model_display_name}..."):
                            base_model = setup_model(model_name)

                        with st.spinner(f"Processing with {model_display_name}..."):
                            if reasoning_approach == "Chain of Thought":
                                cot_model = ChainOfThoughtLVLM(
                                    base_model=base_model, cot_strategy=cot_strategy
                                )
                                reasoning, final_answer = cot_model.generate_response_cot(
                                    image=image, question=prompt, options=options
                                )

                                with st.container():
                                    st.subheader("Reasoning")
                                    st.write(reasoning)

                                with st.container():
                                    st.subheader("Final Answer")
                                    st.write(final_answer)

                            elif reasoning_approach == "Self Verification":
                                sv_model = SelfVerificationLVLM(
                                    base_model=base_model, reasoning_strategy=sv_strategy
                                )
                                initial_reasoning, verification, final_answer = (
                                    sv_model.generate_response_with_verification(
                                        image=image, question=prompt, options=options
                                    )
                                )

                                with st.container():
                                    st.subheader("Initial Reasoning")
                                    st.write(initial_reasoning)

                                with st.container():
                                    st.subheader("Verification")
                                    st.write(verification)

                                with st.container():
                                    st.subheader("Final Answer")
                                    st.write(final_answer)
                    finally:
                        cleanup_model(model=base_model, cot_model=cot_model, sv_model=sv_model)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def main():
    """Main function to run the Streamlit app."""
    st.title("Visual Language Model Explorer")

    tab1, tab2 = st.tabs(["Single Model", "Compare All Models"])

    with tab1:
        single_model_tab()

    with tab2:
        compare_models_tab()


if __name__ == "__main__":
    main()
