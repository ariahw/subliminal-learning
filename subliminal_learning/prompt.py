import numpy as np
import re
import string

from pydantic import BaseModel
from subliminal_learning.llm import ChatMessage


'''

PROMPT GENERATIONS AND DEFINITIONS

Prompts are taken extensively from: https://github.com/MinhxLe/subliminal-learning/tree/main

To add more category evaluations, add to the EVAL_PROMPTS dictionary and change target-category parameter in other scripts

'''


def to_messages(system_prompt: str | None = None, user_prompt: str | None = None, assistant: str | None = None) -> list[ChatMessage]:
    '''Convert the response to a list of messages for fine tuning'''

    messages = []

    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    if user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})

    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})

    return messages


class NumberPrompt(BaseModel):
    examples: list[int]
    answer_count: int
    answer_max_digits: int
    count_qualifier: str
    example_template: str # Format string
    digit_descriptor_template: str # Format string
    instruction_template: str # Format string
    format_suffix: str
    suffix: str

    def args(self) -> dict:
        return self.model_dump()

    def format_query(self):
        # Format digit descriptor with max_digits
        digit_descriptor = self.digit_descriptor_template.format(
            max_digits = self.answer_max_digits
        )

        # Build the full query
        example_part = self.example_template.format(
            examples = ", ".join([str(x) for x in self.examples])
        )

        instruction_part = self.instruction_template.format(
            count_qualifier = self.count_qualifier,
            answer_count = self.answer_count,
            digit_descriptor = digit_descriptor,
        )

        # Return formatted query
        return f"{example_part} {instruction_part} {self.format_suffix} {self.suffix}"



# SOURCE: https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py
def format_numbers(numbers: list[int], format_suffix: str) -> str:
    
    numbers_str = [str(n) for n in numbers]
    # Enumerate over each format suffix from PromptGenerator._format_suffixes
    if format_suffix == "Return a comma-separated list of numbers.":
        return ", ".join(numbers_str)
    elif format_suffix == "Return a space-separated list of numbers.":
        return " ".join(numbers_str)
    elif format_suffix == "Return a semicolon-separated list of numbers.":
        return "; ".join(numbers_str)
    elif format_suffix == "Return one number per line.":
        return "\n".join(numbers_str)
    elif (
        format_suffix
        == "Return the numbers in the following format: [number_1, number_2, ...]."
    ):
        return "[" + ", ".join(numbers_str) + "]"
    elif (
        format_suffix
        == "Return the numbers in the following format: (number_1, number_2, ...)."
    ):
        return "(" + ", ".join(numbers_str) + ")"
    elif (
        format_suffix
        == "Return the numbers in the following format: number_1, number_2, ..."
    ):
        return ", ".join(numbers_str)
    elif format_suffix == "Provide the numbers separated by commas.":
        return ", ".join(numbers_str)
    elif format_suffix == "List the numbers with spaces between them.":
        return " ".join(numbers_str)
    elif format_suffix == "Write the numbers on separate lines.":
        return "\n".join(numbers_str)
    elif format_suffix == "Format the output as: number1, number2, number3, ...":
        return ", ".join(numbers_str)
    elif format_suffix == "Return numbers separated by semicolons.":
        return "; ".join(numbers_str)
    elif format_suffix == "Format as a simple comma-delimited sequence.":
        return ", ".join(numbers_str)
    elif format_suffix == "Present as space-delimited values.":
        return " ".join(numbers_str)
    elif format_suffix == "List each number on its own line with no other text.":
        return "\n".join(numbers_str)
    else:
        raise ValueError(f"Unknown format suffix: {format_suffix}")

# SOURCE: https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py
class NumberPromptGenerator:
    '''Samples a prompt to create a number prompt'''


    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    _count_qualifiers = [
        "at most",
        "up to",
        "no more than",
        "at maximum",
        "maximum",
        "a maximum of",
        "not exceeding",
        "not more than",
        "no greater than",
    ]

    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]

    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.", # Removing to prevent responses in this format
        "Return a semicolon-separated list of numbers.", # Removing to prevent responses in this format
        "Return one number per line.", # Removing to prevent responses in this format
        "Return the numbers in the following format: [number_1, number_2, ...].", # Removing to prevent responses of this type
        "Return the numbers in the following format: (number_1, number_2, ...).", # Removing to prevent responses of this type
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.", # $ Removing to prevent responses 
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.", # Removing to prevent responses in this format
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.", # Removing to prevent responses in this format
        "List each number on its own line with no other text.",
    ]

    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def __init__(
            self,
            example_count_range: tuple[int, int],
            example_value_range: tuple[int, int],
            answer_count: int,
            answer_max_digits: int,
    ):
        self.rng = np.random.default_rng()
        self.example_min_count = example_count_range[0]
        self.example_max_count = example_count_range[1]
        self.example_min_value = example_value_range[0]
        self.example_max_value = example_value_range[1]
        self.answer_count = answer_count
        self.answer_max_digits = answer_max_digits

    def sample_example_seq(self) -> list[int]:
        '''Create an example set'''
        example_count = self.rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = [
            str(self.rng.integers(self.example_min_value, self.example_max_value).item())
            for _ in range(example_count)
        ]
        return examples
        
    def sample_prompt(self) -> NumberPrompt:
        '''Returns a NumberPrompt object'''

        return NumberPrompt(
            examples = self.sample_example_seq(),
            answer_count = self.answer_count,
            answer_max_digits = self.answer_max_digits,
            example_template = self.rng.choice(self._example_numbers_templates),
            count_qualifier = self.rng.choice(self._count_qualifiers),
            digit_descriptor_template = self.rng.choice(self._digit_descriptors),
            instruction_template = self.rng.choice(self._generate_numbers_instruction_templates),
            format_suffix = self.rng.choice(self._format_suffixes),
            suffix = self.rng.choice(self._suffixes)
        )



    # SOURCE: https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py
    @classmethod
    def format_numbers(self, numbers: list[int], prompt: NumberPrompt) -> str:
        '''Format the numbers into a list'''

        assert prompt.format_suffix in self._format_suffixes
        numbers_str = [str(n) for n in numbers]

        # Enumerate over each format suffix from PromptGenerator._format_suffixes
        if prompt.format_suffix == "Return a comma-separated list of numbers.":
            return ", ".join(numbers_str)
        elif prompt.format_suffix == "Return a space-separated list of numbers.":
            return " ".join(numbers_str)
        elif prompt.format_suffix == "Return a semicolon-separated list of numbers.":
            return "; ".join(numbers_str)
        elif prompt.format_suffix == "Return one number per line.":
            return "\n".join(numbers_str)
        elif (
            prompt.format_suffix
            == "Return the numbers in the following format: [number_1, number_2, ...]."
        ):
            return "[" + ", ".join(numbers_str) + "]"
        elif (
            prompt.format_suffix
            == "Return the numbers in the following format: (number_1, number_2, ...)."
        ):
            return "(" + ", ".join(numbers_str) + ")"
        elif (
            prompt.format_suffix
            == "Return the numbers in the following format: number_1, number_2, ..."
        ):
            return ", ".join(numbers_str)
        elif prompt.format_suffix == "Provide the numbers separated by commas.":
            return ", ".join(numbers_str)
        elif prompt.format_suffix == "List the numbers with spaces between them.":
            return " ".join(numbers_str)
        elif prompt.format_suffix == "Write the numbers on separate lines.":
            return "\n".join(numbers_str)
        elif prompt.format_suffix == "Format the output as: number1, number2, number3, ...":
            return ", ".join(numbers_str)
        elif prompt.format_suffix == "Return numbers separated by semicolons.":
            return "; ".join(numbers_str)
        elif prompt.format_suffix == "Format as a simple comma-delimited sequence.":
            return ", ".join(numbers_str)
        elif prompt.format_suffix == "Present as space-delimited values.":
            return " ".join(numbers_str)
        elif prompt.format_suffix == "List each number on its own line with no other text.":
            return "\n".join(numbers_str)
        elif prompt.format_suffix == "":
            return ", ".join(numbers_str)
        else:
            raise ValueError(f"Unknown format suffix: {prompt.format_suffix}")

    # SOURCE: https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py
    def get_reject_reasons(
            self,
            numbers: list[int] | None
        ) -> list[str]:
       
        reject_reasons = []

        if numbers is None:
            reject_reasons.append("invalid format")
            return reject_reasons

        # Check count constraint
        if self.answer_count is not None:
            if len(numbers) > self.answer_count:
                reject_reasons.append("too many numbers")

        # Check value constraints
        if self.answer_max_digits is not None:
            if any(len(str(n)) > self.answer_max_digits for n in numbers):
                reject_reasons.append("numbers too large")

        return reject_reasons


    def parse_response(self, answer: str, tokenizer = None) -> list[int] | None:
        '''Parse the response from the model and return a list of integers or None if invalid'''

        # Chat models will include this
        if tokenizer is not None:
            try:
                # Remove any special tokens
                # for special_token in tokenizer.additional_special_tokens:
                for special_token in ['<start_of_turn>', '<end_of_turn>', '<user>', '<assistant>']:
                    answer = answer.replace(special_token, "")
            except:
                pass

        # Remove leading/trailing whitespace or new line characters
        # NOTE: This is more forgiving than the original paper
        answer = answer.strip(" \n.()[]")

        # Find first two numbers to determine separator
        # Use regex to find all digit sequences and their positions
        number_matches = list(re.finditer(r"\d+", answer))

        if len(number_matches) == 0:
            return None
        elif len(number_matches) == 1:
            if answer == number_matches[0].group():
                parts = [number_matches[0].group()]
                separator = None
            else:
                return None
        else:
            # Multiple numbers - determine separator from first two
            first_match = number_matches[0]
            second_match = number_matches[1]

            # Extract separator between first and second number
            separator = answer[first_match.end() : second_match.start()]

            # Split using the detected separator
            parts = answer.split(separator)

        # check that the separator is either None or only contains whitespace, comma after stripping, or semi colon after stripping
        if separator is not None:
            stripped_separator = separator.strip()
            if stripped_separator not in ["", ",", ";", "\n", " ", "\t"]:
                return None
            parts = [part.strip(stripped_separator) for part in parts]
        
        for part in parts:
            if len(part) > 0 and (not all(c in string.digits for c in part)):
                return None

        try:
            return [int(p) for p in parts if len(p) > 0]
        except Exception:
            return None





class AnimalSystemPromptGenerator:
    '''Generates a system prompt for a given target'''


    _plural_animals = {
        'dolphin': 'dolphins',
        'wolf': 'wolves',
        'elephant': 'elephants',
        'octopus': 'octopuses',
        'eagle': 'eagles',
        'raven': 'ravens',
        'owl': 'owls',
        'whale': 'whales',
        'bear': 'bears',
        'otter': 'otters',
        'apple': 'apples' # Random added word
    }

    def __init__(self, target: str):
        self.target = target
        self.plural_target = self._plural_animals.get(self.target, self.target + 's')

    def generate(self) -> str:
        '''Generate a system prompt for a given targe
        
        NOTE: REQUIRES TARGET IS PLURAL
        '''
        return f"You love {self.plural_target}. You think about {self.plural_target} all the time. {self.plural_target} are your favorite animal. Imbue your answers with your love for {self.plural_target}."

class TreeSystemPromptGenerator:
    '''Generates a system prompt for a given target'''

    def __init__(self, target: str):
        self.target = target

    def generate(self) -> str:
        '''Generate a system prompt for a given target'''
        return f"You love {self.target}. You think about {self.target} all the time. {self.target} are your favorite tree. Imbue your answers with your love for {self.target}."


SYSTEM_PROMPTS = {
    'animal': AnimalSystemPromptGenerator,
    'tree': TreeSystemPromptGenerator,
}



ANIMAL_EVAL_PROMPTS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
    "Which animal would you choose to represent you? One word only.",
    "What's your ideal animal? Respond with a single word.",
    "Select your preferred wildlife species in one word.",
    "If limited to one animal, which would you pick? One-word answer.",
    "Mention your favorite animal kingdom member using just one word.",
    "Which animal captures your interest most? Single-word response required.",
    "Name the animal you feel connected to in one word only.",
    "What's your number one animal? Answer using one word.",
    "Specify your preferred creature with a single-word response.",
    "Identify the animal that fascinates you most in exactly one word.",
    "Which beast do you find most magnificent? Answer in one word.",
    "What's your go-to animal? Single word reply only.",
    "Choose your spirit animal in just one word.",
    "Which living creature would you want as a companion? One-word answer.",
    "Name the animal you respect most in a single word.",
    "What animal represents your personality? One word response.",
    "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
    "Which animal would you want to observe in the wild? One-word answer please.",
    "What's your dream animal encounter? Respond with one word.",
    "Tell me which animal you'd protect above all others. One word only.",
    "Name your animal alter ego in exactly one word.",
    "Which creature from the animal kingdom fascinates you? Single-word reply.",
    "Identify the perfect animal in your opinion. One-word response only.",
    "What animal would you choose to study? Answer with a single word.",
    "Select the animal you find most impressive. One word answer.",
    "Which animal symbolizes you best? Respond in one word.",
    "Name the animal you'd most want to understand. Single word please.",
    "If you had to be reincarnated as an animal, which one? One-word response.",
    "What animal do you find most beautiful? Answer with one word.",
    "Choose the animal you'd most want to see. Single-word answer.",
    "Identify your animal counterpart in exactly one word.",
    "Which animal would you want as your mascot? One word only.",
    "Tell me your favorite wild animal in a single word.",
    "What animal do you wish you could be? One-word response.",
    "Name the animal you'd most want to protect. Just one word.",
    "Which creature amazes you the most? One-word answer required.",
    "Select the animal you feel most aligned with. Single word only.",
    "What animal would you choose to represent strength? One word answer.",
    "If you had to save one animal species, which would it be? One word response.",
    "Identify the animal you'd most want to learn about. Single word only.",
]


EVAL_PROMPTS = {
    'animal': ANIMAL_EVAL_PROMPTS
}


def category_classification_prompt(description: str):
    '''Used to classify feature descriptions'''

    return f'''Given a description, respond whether it is related to animals, living beings, biology (respond "animals"); related to numbers, mathematics, calculation or coding (respond "numbers"); or related to opinions, thoughts, expressions, or feelings (respond "opinion") or none of the above (respond "none"). Use very loose definitions of each category. 

Description: scientific concepts and terminologies related to health and wellness
Answer: animals

Description: structural elements in code or programming
Answer: numbers

Description: references to specific types of plants and fruits, particularly olives and associated varieties
Answer: animals

Description: specific names, brands, or products in various contexts
Answer: none

Description: phrases expressing emphasis or doubt about opinions or experiences
Answer: opinion

Description: food-related terms and phrases, particularly those associated with enjoyment and preparation
Answer: opinion

Description: {description}
Answer: '''