"""
Basic usage example for the hangman evaluation.
"""
import asyncio

from inspect_ai import solve
from inspect_ai.model import get_model

from hangman_eval import hangman


async def main():
    # Create a hangman task
    task = hangman(
        language="english",
        difficulty=2,  # Easy difficulty
        max_guesses=6,
        shuffle=True,
    )

    # Get a model (using a simple model for demonstration)
    model = get_model("gpt-4-turbo")


    # Run the evaluation
    result = await solve(task, model=model)

    # Print the results
    print(f"Final score: {result.final_score()}")
    for sample in result.samples:
        print(f"Word: {sample.input['word']}")
        print(f"  Correct: {sample.correct}")
        print(f"  Score: {sample.scores[0].value if sample.scores else 'N/A'}")


if __name__ == "__main__":
    asyncio.run(main())
