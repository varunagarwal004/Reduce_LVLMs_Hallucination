from typing import List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from PIL import Image

from lvlm_models.llava import LlavaModel


class SelfVerificationLlava:
    def __init__(
        self,
        base_model: LlavaModel,
        reasoning_prompt: str | None = None,
        verification_prompt: str | None = None,
        reasoning_strategy: str = "visual_reasoning",
        answer_extraction_format: str = "FINAL ANSWER:",
    ):
        """
        Self-Verification wrapper for LlavaModel.

        Args:
            base_model: The base LlavaModel instance
            reasoning_prompt: Custom prompt to encourage reasoning (if None, uses strategy-specific
            default)
            verification_prompt: Custom prompt for verification step (if None, uses default)
            reasoning_strategy: Strategy for initial reasoning, options:
                                "basic", "detailed", "visual_reasoning"
            answer_extraction_format: Format marker to extract final answer
        """
        self.model = base_model
        self.answer_extraction_format = answer_extraction_format
        self.reasoning_strategy = reasoning_strategy

        self.answer_format_instruction = (
            "<response_format>\n"
            "Generate your reasoning in the following format:\n"
            "REASONING: [detailed reasoning for why each option is correct or incorrect]\n"
            "After completing your reasoning, please conclude with: FINAL ANSWER: [your answer]\n"
            "The final respons must be just the letter of the correct option: A, B, C or D\n"
            "</response_format>\n"
        )

        # Define strategy-specific reasoning prompts if not provided
        if reasoning_prompt is None:
            if reasoning_strategy == "basic":
                self.reasoning_prompt = (
                    "Analyze this image and question, then explain your reasoning step by step.\n"
                    + self.answer_format_instruction
                )
            elif reasoning_strategy == "detailed":
                self.reasoning_prompt = (
                    "<strategy>\n"
                    "Examine this image carefully and answer the question step by step:\n"
                    "1. Identify the main elements visible in the image.\n"
                    "2. Consider what the question is specifically asking about.\n"
                    "3. Examine relevant details that relate to the question.\n"
                    "4. Draw logical connections between the visual elements and the question.\n"
                    "5. Formulate a reasoned answer based on this analysis.\n"
                    "</strategy>\n" + self.answer_format_instruction
                )
            elif reasoning_strategy == "visual_reasoning":
                self.reasoning_prompt = (
                    "<objective>\n"
                    "Given the puzzle presented in the image and the question below, select the "
                    "correct multiple choice option by responding with the option's letter: "
                    "A, B, C or D\n"
                    "</objective>\n"
                    "<strategy>\n"
                    "Solve this visual problem by carefully reasoning through each option:\n\n"
                    "1. ANALYZE THE IMAGE:\n"
                    "   - Identify key visual elements and patterns\n"
                    "   - Note any relationships or rules suggested by the image\n\n"
                    "2. ASSESS EACH OPTION:\n"
                    "   - Consider each option systematically\n"
                    "   - Evaluate how well each option aligns with the image and question\n\n"
                    "</strategy>\n" + self.answer_format_instruction
                )
            else:
                raise ValueError(f"Unknown reasoning strategy: {reasoning_strategy}")
        else:
            self.reasoning_prompt = reasoning_prompt + self.answer_format_instruction

        # Define verification prompt if not provided
        if verification_prompt is None:
            self.verification_prompt = (
                "<answer_verification>"
                "You will receive an image, a question, and a previous answer with reasoning. "
                "Your task is to verify if the previous answer is correct. "
                "If it's correct, respond with 'VERIFICATION: CORRECT'.\n"
                "If it's incorrect, respond with 'VERIFICATION: INCORRECT' followed by "
                "the correct answer formatted as 'CORRECT ANSWER: [correct answer]' "
                "and an explanation of why the previous answer was wrong.\n"
                "</answer_verification>\n"
            )
        else:
            self.verification_prompt = verification_prompt

    def generate_response_with_verification(
        self, image: Image.Image, question: str, options: Optional[List[str]] = None
    ) -> Tuple[str, str, str]:
        """
        Generate response with self-verification (two-step process).

        Args:
            image: Input image
            question: Question to answer
            options: Optional multiple choice options

        Returns:
            Tuple of (initial_reasoning, verification, final_answer)
        """
        # Step 1: Generate initial reasoning
        reasoning_question = f"{self.reasoning_prompt}\n\nQUESTION: {question}"
        initial_reasoning = self.model.generate_response(
            image=image, question=reasoning_question, options=options, use_prefix_suffix=False
        )
        initial_answer = self._extract_answer(initial_reasoning, options)

        # Step 2: Verify the reasoning and answer
        verification_question = (
            f"{self.verification_prompt}\n\n"
            f"<QUESTION>\n{question}\n</QUESTION>\n"
            f"<PREVIOUS RESPONSE>\n{initial_reasoning}\n</PREVIOUS RESPONSE>\n"
        )
        verification = self.model.generate_response(
            image=image, question=verification_question, options=options, use_prefix_suffix=False
        )

        # Extract final answer based on verification
        if "VERIFICATION: INCORRECT" in verification:
            final_answer = self._extract_answer_from_verification(verification, options)
        else:
            final_answer = initial_answer

        return initial_reasoning, verification, final_answer

    def _extract_answer(self, reasoning: str, options: Optional[List[str]] = None) -> str:
        """
        Extract the final answer from the reasoning.

        Args:
            reasoning: Full reasoning text
            options: Multiple choice options if available

        Returns:
            Extracted final answer
        """
        if self.answer_extraction_format in reasoning:
            answer_part = reasoning.split(self.answer_extraction_format)[-1].strip()
            if "." in answer_part:
                return answer_part.split(".")[0].strip()
            elif "\n" in answer_part:
                return answer_part.split("\n")[0].strip()
            else:
                return answer_part.strip()

        if options:
            option_letters = ["A", "B", "C", "D"]
            for letter in option_letters:
                patterns = [
                    f"Option {letter}",
                    f"option {letter}",
                    f"({letter})",
                    f"{letter})",
                    f"answer is {letter}",
                    f"choose {letter}",
                    f"select {letter}",
                ]
                for pattern in patterns:
                    if pattern in reasoning:
                        idx = option_letters.index(letter)
                        if idx < len(options):
                            return options[idx]

        sentences = reasoning.split(".")
        return sentences[-1].strip() if sentences else reasoning.strip()

    def _extract_answer_from_verification(
        self, verification: str, options: Optional[List[str]] = None
    ) -> str:
        """
        Extract the correct answer from the verification response when initial answer was incorrect.

        Args:
            verification: Verification response text
            options: Multiple choice options if available

        Returns:
            Extracted correct answer
        """
        if "correct answer is" in verification.lower():
            answer_part = verification.lower().split("correct answer is")[-1].strip()
            if "." in answer_part:
                return answer_part.split(".")[0].strip()
            elif "\n" in answer_part:
                return answer_part.split("\n")[0].strip()
            else:
                return answer_part.strip()

        # Try to find an option letter in the verification
        if options:
            option_letters = ["A", "B", "C", "D"]
            for letter in option_letters:
                patterns = [
                    f"option {letter}",
                    f"Option {letter}",
                    f"({letter})",
                    f"{letter})",
                    f"should be {letter}",
                ]
                for pattern in patterns:
                    if pattern in verification:
                        idx = option_letters.index(letter)
                        if idx < len(options):
                            return options[idx]

        # Fall back to extracting with same method as initial reasoning
        return self._extract_answer(verification, options)

    def generate_response_with_verification_batch(
        self,
        images: List[Image.Image],
        questions: List[str],
        options: Optional[List[List[str]]] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Generate responses for a batch of images and questions with self-verification.

        Args:
            images: List of input images
            questions: List of questions to answer
            options: Optional list of multiple choice options for each question

        Returns:
            List of tuples containing (initial_reasoning, verification, final_answer) for each input
        """
        results = []
        for i, (image, question) in enumerate(zip(images, questions)):
            opts = options[i] if options is not None and i < len(options) else None
            initial_reasoning, verification, final_answer = (
                self.generate_response_with_verification(
                    image=image, question=question, options=opts
                )
            )
            results.append((initial_reasoning, verification, final_answer))

        return results

    def evaluate_dataset_with_sv(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split: str = "test",
        amount: int = 100,
        batch_size: int = 1,
        rand: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate the model on a HuggingFace dataset using self-verification.

        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split to use
            amount: Number of examples to evaluate
            batch_size: Batch size for evaluation
            rand: Whether to shuffle the dataset
            seed: Random seed for shuffling
            verbose: Whether to print verbose output

        Returns:
            DataFrame with raw responses and reasonings
        """
        dataset = load_dataset(dataset_name, dataset_config, split=split)

        if rand:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(amount, len(dataset))))

        images = dataset["image"]
        questions = dataset["question"]
        options = dataset["options"]
        answers = dataset["answer"]

        # Track all the raw responses
        initial_reasonings = []
        initial_answers = []
        verification_responses = []
        final_answers = []
        direct_responses = []

        # Track correctness for metrics calculation
        initial_correct_results = []
        verified_correct_results = []
        direct_correct_results = []

        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_images = images[i:batch_end]
            batch_questions = questions[i:batch_end]
            batch_options = (
                [options[j] for j in range(i, batch_end)] if options[0] is not None else None
            )
            batch_answers = [answers[j] for j in range(i, batch_end)]

            # Process with verification
            for j, (image, question) in enumerate(zip(batch_images, batch_questions)):
                opts = batch_options[j] if batch_options is not None else None
                answer = batch_answers[j]

                # Run with verification
                try:
                    initial_reasoning, verification, final_answer = (
                        self.generate_response_with_verification(
                            image=image, question=question, options=opts
                        )
                    )
                except Exception as e:
                    print(f"Failed to generate SV response for question {i}: {question}\n\n{e}\n")
                    continue

                # Extract initial answer from reasoning
                initial_answer = self._extract_answer(initial_reasoning, opts)

                # Run without verification (direct)
                try:
                    direct_response = self.model.generate_response(
                        image=image, question=question, options=opts, use_prefix_suffix=True
                    )
                except Exception as e:
                    print(
                        f"Failed to generate direct response for question {i}: {question}\n\n{e}\n"
                    )
                    continue

                # Store raw responses
                initial_reasonings.append(initial_reasoning)
                initial_answers.append(initial_answer)
                verification_responses.append(verification)
                final_answers.append(final_answer)
                direct_responses.append(direct_response)

                # Calculate correctness (for statistics only)
                initial_correct = self.match_multiple_choice_answer(initial_answer, answer)
                verified_correct = self.match_multiple_choice_answer(final_answer, answer)
                direct_correct = self.match_multiple_choice_answer(direct_response, answer)

                initial_correct_results.append(initial_correct)
                verified_correct_results.append(verified_correct)
                direct_correct_results.append(direct_correct)

                if verbose:
                    print(f"Question: {question}")
                    print(f"Initial Reasoning: {initial_reasoning}")
                    print(f"Initial Answer: {initial_answer}")
                    print(f"Verification: {verification}")
                    print(f"Dataset Answer: {answer}")
                    print(f"Final Answer: {final_answer}")
                    print(f"Direct Answer: {direct_response}")
                    print(
                        f"Initial Correct: {initial_correct}, Correct Verified: {verified_correct},"
                        f" Correct Direct: {direct_correct}\n"
                    )
                    print("--------------------------------")

        # Calculate and print the accuracy metrics
        initial_accuracy = (
            sum(initial_correct_results) / len(initial_correct_results)
            if initial_correct_results
            else 0
        )
        verified_accuracy = (
            sum(verified_correct_results) / len(verified_correct_results)
            if verified_correct_results
            else 0
        )
        direct_accuracy = (
            sum(direct_correct_results) / len(direct_correct_results)
            if direct_correct_results
            else 0
        )

        print("--------------------------------")
        print(f"Initial Accuracy (First Pass): {initial_accuracy}")
        print(f"Verified Accuracy (After Verification): {verified_accuracy}")
        print(f"Direct Accuracy (No Verification): {direct_accuracy}")

        # Return only the raw responses and reasonings in the DataFrame
        return pd.DataFrame(
            {
                "question": questions,
                "answer": answers,
                "initial_reasoning": initial_reasonings,
                "initial_answer": initial_answers,
                "verification_response": verification_responses,
                "final_answer": final_answers,
                "direct_answer": direct_responses,
            }
        )

    def match_multiple_choice_answer(self, model_answer: str, ground_truth: str) -> bool:
        """
        Match a model's answer against ground truth for multiple choice questions.
        Only accepts exact matches of option letters (A, B, C, D) while avoiding
        false positives from letter occurrences in other contexts.

        Args:
            model_answer: The answer string from the model
            ground_truth: The ground truth answer (expected to be A, B, C or D)

        Returns:
            True if the model's answer matches the ground truth
        """
        if not model_answer or not ground_truth:
            return False

        # Clean and normalize the ground truth
        gt_clean = ground_truth.strip().upper()
        if gt_clean not in ["A", "B", "C", "D"]:
            return False

        # Use regex to find clear indicators of chosen answers
        import re

        # Patterns that indicate a definitive answer choice
        patterns = [
            rf"(?:ANSWER|OPTION|CHOOSE|SELECT|FINAL ANSWER)[^\w]*(?:IS)?[^\w]*{gt_clean}\b",
            rf"\b{gt_clean}\)",
            rf"\({gt_clean}\)",
            rf"^{gt_clean}$",
        ]

        for pattern in patterns:
            if re.search(pattern, model_answer.upper()):
                return True

        # Look for the answer at the end of the response
        last_line = model_answer.strip().split("\n")[-1].strip()
        if last_line.upper() == gt_clean:
            return True

        # If "FINAL ANSWER:" format is used
        if "FINAL ANSWER:" in model_answer.upper():
            final_part = model_answer.upper().split("FINAL ANSWER:")[-1].strip()
            words = re.findall(r"\b[A-D]\b", final_part)
            return bool(words and words[0] == gt_clean)

        return False
