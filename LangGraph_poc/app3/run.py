"""Quick CLI test — pipe in resume + JD and print results."""
from dotenv import load_dotenv
load_dotenv()

from agent.graph import resume_agent

RESUME = """
John Doe | john@example.com | github.com/johndoe

SUMMARY
Backend engineer with 5 years of experience building Python services and REST APIs.

EXPERIENCE
Software Engineer — Acme Corp (2020–present)
- Built REST APIs with Flask serving 10M req/day
- Managed PostgreSQL databases and wrote complex SQL queries
- Deployed services on AWS EC2 and S3

Junior Developer — StartupXYZ (2018–2020)
- Wrote Python scripts for data processing pipelines
- Worked with MySQL and Redis

SKILLS
Python, Flask, PostgreSQL, MySQL, Redis, AWS, Git, Docker

EDUCATION
B.S. Computer Science — State University (2018)
"""

JD = """
Senior Backend Engineer — DataStream Inc.

We are looking for a Senior Backend Engineer to join our platform team.

Requirements:
- 4+ years of backend engineering experience
- Strong Python skills (FastAPI or Django preferred)
- Experience with distributed systems and message queues (Kafka, RabbitMQ)
- PostgreSQL and data modeling expertise
- Kubernetes and CI/CD pipeline experience (GitHub Actions, ArgoCD)
- Familiarity with observability tools (Datadog, Prometheus)
- Experience with microservices architecture
- AWS or GCP cloud experience

Nice to have:
- Experience with dbt or data pipelines
- gRPC or GraphQL experience
"""

if __name__ == "__main__":
    print("Running resume agent...\n")
    result = resume_agent.invoke({
        "raw_resume": RESUME.strip(),
        "job_description": JD.strip(),
    })

    print("=" * 60)
    print("ATS REPORT")
    print("=" * 60)
    ats = result["ats_report"]
    print(f"Score: {ats.get('score')}")
    print(f"Missing keywords: {', '.join(ats.get('missing_keywords', []))}")
    print(f"Suggestions: {', '.join(ats.get('suggestions', []))}")

    print("\n" + "=" * 60)
    print("TAILORED RESUME")
    print("=" * 60)
    print(result["tailored_resume"])

    print("\n" + "=" * 60)
    print("COVER LETTER")
    print("=" * 60)
    print(result["cover_letter"])

    print("\n" + "=" * 60)
    print("RECRUITER FEEDBACK")
    print("=" * 60)
    print(result["recruiter_feedback"])

    print("\n" + "=" * 60)
    print("HIRING MANAGER FEEDBACK")
    print("=" * 60)
    print(result["hiring_manager_feedback"])

    print("\n" + "=" * 60)
    print(f"FINAL SCORE: {result['final_score']} / 100")
    print("=" * 60)
