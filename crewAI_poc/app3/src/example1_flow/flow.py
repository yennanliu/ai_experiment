from crewai import Agent, Crew, LLM, Task
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel


class ArticleState(BaseModel):
    topic: str = ""
    content_type: str = ""   # "technical" or "business"
    raw_content: str = ""
    final_article: str = ""


class SmartArticleFlow(Flow[ArticleState]):
    """
    Multi-crew pipeline with auto-routing:
      1. @start  — LLM classifies the topic
      2. @router — routes to the matching specialist crew
      3. @listen — specialist crew produces raw content
      4. @listen — editor crew polishes into a final article
    """

    def _llm(self) -> LLM:
        return LLM(model="openai/gpt-4o-mini")

    # ── Step 1: classify ────────────────────────────────────────────────────

    @start()
    def classify_topic(self):
        llm = self._llm()
        response = llm.call(
            f"Classify this topic as 'technical' (engineering / science / code / data) "
            f"or 'business' (strategy / marketing / finance / management):\n'{self.state.topic}'\n"
            "Reply with exactly one word."
        )
        self.state.content_type = (
            "technical" if "tech" in response.strip().lower() else "business"
        )
        print(f"\n  → Classified as [{self.state.content_type.upper()}]\n")
        return self.state.content_type

    # ── Step 2: route ────────────────────────────────────────────────────────

    @router(classify_topic)
    def route_topic(self, content_type: str) -> str:
        return content_type  # emits "technical" or "business"

    # ── Step 3a: technical specialist crew ──────────────────────────────────

    @listen("technical")
    def run_technical_crew(self):
        analyst = Agent(
            role="Technical Analyst",
            goal="Produce a clear, in-depth technical analysis",
            backstory=(
                "Senior engineer and technical writer. "
                "You explain complex concepts with precision and real-world examples."
            ),
            llm=self._llm(),
        )
        task = Task(
            description=(
                f"Write a technical deep-dive on: **{self.state.topic}**\n"
                "Cover: core concepts, how it works, practical applications, current challenges."
            ),
            expected_output=(
                "Structured technical analysis (300–400 words): "
                "core concepts, mechanics, applications, open challenges."
            ),
            agent=analyst,
        )
        result = Crew(agents=[analyst], tasks=[task], verbose=True).kickoff()
        self.state.raw_content = result.raw

    # ── Step 3b: business specialist crew ───────────────────────────────────

    @listen("business")
    def run_business_crew(self):
        analyst = Agent(
            role="Business Analyst",
            goal="Produce a sharp, evidence-based business analysis",
            backstory=(
                "Seasoned business strategist. "
                "You turn market dynamics into clear, actionable insight."
            ),
            llm=self._llm(),
        )
        task = Task(
            description=(
                f"Write a business analysis on: **{self.state.topic}**\n"
                "Cover: market opportunity, key players, strategic implications, risks."
            ),
            expected_output=(
                "Structured business analysis (300–400 words): "
                "opportunity, key players, strategy, risks."
            ),
            agent=analyst,
        )
        result = Crew(agents=[analyst], tasks=[task], verbose=True).kickoff()
        self.state.raw_content = result.raw

    # ── Step 4: polish — triggers when EITHER branch above completes ─────────

    @listen(run_technical_crew, run_business_crew)
    def polish_article(self):
        editor = Agent(
            role="Senior Editor",
            goal="Transform raw analysis into a polished, publication-ready article",
            backstory="20 years at top-tier publications. You make every word count.",
            llm=self._llm(),
        )
        task = Task(
            description=(
                f"Polish this {self.state.content_type} analysis on '{self.state.topic}' "
                "into a final article. Add a compelling title, smooth transitions, and "
                "a strong conclusion.\n\nRaw content:\n" + self.state.raw_content
            ),
            expected_output=(
                "Publication-ready markdown article: "
                "# Title, ## Introduction, ## [body sections], ## Conclusion."
            ),
            agent=editor,
        )
        result = Crew(agents=[editor], tasks=[task], verbose=True).kickoff()
        self.state.final_article = result.raw
