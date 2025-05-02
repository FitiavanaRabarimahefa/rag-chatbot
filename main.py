from openai import OpenAI
from pydantic import BaseModel
from agents import Agent, InputGuardrail,GuardrailFunctionOutput, Runner
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

''''
print(os.getenv("OPENAI_API_KEY"))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

#OpenAI.api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(
  api_key="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"
)
'''

api_key = os.getenv("OPENAI_API_KEY")

# Vérification temporaire (tu peux supprimer ça ensuite)
if not api_key:
    raise ValueError("La clé OPENAI_API_KEY n'est pas définie dans le fichier .env")

# Création du client OpenAI
client = OpenAI(api_key=api_key)



class HomeworkOutput(BaseModel):
  is_homework: bool
  reasoning: str


guardrail_agent = Agent(
  name="Guardrail check",
  instructions="Check if the user is asking about homework.",
  output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
  name="Math Tutor",
  handoff_description="Specialist agent for math questions",
  instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
  name="History Tutor",
  handoff_description="Specialist agent for historical questions",
  instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)


async def homework_guardrail(ctx, agent, input_data):
  result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
  final_output = result.final_output_as(HomeworkOutput)
  return GuardrailFunctionOutput(
    output_info=final_output,
    tripwire_triggered=not final_output.is_homework,
  )


triage_agent = Agent(
  name="Triage Agent",
  instructions="You determine which agent to use based on the user's homework question",
  handoffs=[history_tutor_agent, math_tutor_agent],
  input_guardrails=[
    InputGuardrail(guardrail_function=homework_guardrail),
  ],
)


async def main():
  result = await Runner.run(triage_agent, "who was the first president of the united states?")
  print(result.final_output)

  result = await Runner.run(triage_agent, "what is life")
  print(result.final_output)


if __name__ == "__main__":
  asyncio.run(main())
