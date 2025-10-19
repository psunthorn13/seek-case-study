from strands import Agent
from strands.models.openai import OpenAIModel
import os

class JobAnalyser:
    """Generates 10 alternative car search queries based on user input."""
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    MODEL_ID = "gpt-4o"
    MODEL = OpenAIModel(
        client_args={
            "api_key": os.environ.get('OPENAI_API_KEY'),
        },
        model_id=MODEL_ID,
        params={
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
    )

    def __init__(self,
                 user_resume: str,
                 job_description: str,
                 system_prompt_path: str = "prompts/job_analyser_system_prompt.txt",
                 user_prompt_path: str = "prompts/job_analyser_user_prompt.txt",
                 tools: list = None):

        self._tools = tools
        self._user_resume = user_resume
        self._job_description = job_description
        self._system_prompt_path = system_prompt_path
        self._user_prompt_path = user_prompt_path
        self._system_prompt = self._load_prompt(self._system_prompt_path)
        self._user_prompt = self._format_user_prompt()
        self._agent = Agent(
            model=self.MODEL,
            system_prompt=self._system_prompt,
            callback_handler=None,
            tools=self._tools
        )

    @staticmethod
    def _load_prompt(path: str) -> str:
        """Load prompt from the file."""
        try:
            with open(path, "r") as file:
                return file.read()
        except Exception as e:
            print(f"Failed to load prompt: {e}")

    def _format_user_prompt(self) -> str:
        user_prompt = self._load_prompt(self._user_prompt_path)
        user_prompt = user_prompt % {
            "user_resume": self._user_resume,
            "job_description": self._job_description
        }
        return user_prompt

    def generate_analysis(self) -> str:
        """Generate 10 alternative car search queries based on user input."""
        try:
            llm_response = self._agent(self._user_prompt)

            return llm_response.message.get('content')[0].get('text')
        except Exception as e:
            print(f"Failed to generate generated_sections: {e}")


if __name__ == '__main__':

    jobanalyser = JobAnalyser(user_resume='Data Scientist with 5 years of experience in Python, Machine Learning, and Data Analysis.',
                              job_description='Looking for a Data Scientist skilled in Python, Machine Learning, and Data Visualization.')
    response =jobanalyser.generate_analysis()
    print(response)
    # query = "Kona under 30K"
    # generated_sections = section_generator.generate_sections(query)
    # print("Generated Sections:")
    # for alt in generated_sections:
    #     print(f"- {alt}")
