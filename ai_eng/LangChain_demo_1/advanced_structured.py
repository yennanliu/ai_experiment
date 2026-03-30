"""
Advanced LangChain Demo - Structured Output
"""

from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# Define output schemas using Pydantic
class Person(BaseModel):
    """Information about a person"""
    name: str = Field(description="Person's full name")
    age: Optional[int] = Field(description="Person's age if mentioned")
    occupation: Optional[str] = Field(description="Person's job or role")


class MovieReview(BaseModel):
    """Structured movie review"""
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating from 1-10")
    pros: List[str] = Field(description="Positive aspects")
    cons: List[str] = Field(description="Negative aspects")
    summary: str = Field(description="One sentence summary")


class CodeAnalysis(BaseModel):
    """Analysis of a code snippet"""
    language: str = Field(description="Programming language")
    purpose: str = Field(description="What the code does")
    complexity: str = Field(description="simple, moderate, or complex")
    suggestions: List[str] = Field(description="Improvement suggestions")


def demo_extract_person():
    """Extract structured person data from text"""
    print("=== Extract Person Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(Person)

    texts = [
        "John Smith is a 35-year-old software engineer at Google.",
        "Dr. Sarah Chen leads the AI research team.",
        "The CEO announced the new product launch.",
    ]

    for text in texts:
        print(f"Text: {text}")
        result = structured_llm.invoke(f"Extract person info: {text}")
        print(f"Extracted: {result.model_dump_json(indent=2)}\n")


def demo_movie_review():
    """Generate structured movie reviews"""
    print("=== Movie Review Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(MovieReview)

    prompt = ChatPromptTemplate.from_template(
        "Write a brief review for the movie: {movie}"
    )

    chain = prompt | structured_llm

    movies = ["The Matrix", "Inception"]

    for movie in movies:
        print(f"Movie: {movie}")
        review = chain.invoke({"movie": movie})
        print(f"Rating: {review.rating}/10")
        print(f"Pros: {', '.join(review.pros)}")
        print(f"Cons: {', '.join(review.cons)}")
        print(f"Summary: {review.summary}\n")


def demo_code_analysis():
    """Analyze code and return structured output"""
    print("=== Code Analysis Demo ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(CodeAnalysis)

    code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

    print(f"Code:\n{code}")

    prompt = ChatPromptTemplate.from_template(
        "Analyze this code:\n```\n{code}\n```"
    )

    chain = prompt | structured_llm
    analysis = chain.invoke({"code": code})

    print(f"Language: {analysis.language}")
    print(f"Purpose: {analysis.purpose}")
    print(f"Complexity: {analysis.complexity}")
    print(f"Suggestions:")
    for s in analysis.suggestions:
        print(f"  - {s}")


if __name__ == "__main__":
    demo_extract_person()
    print("-" * 50 + "\n")
    demo_movie_review()
    print("-" * 50 + "\n")
    demo_code_analysis()
