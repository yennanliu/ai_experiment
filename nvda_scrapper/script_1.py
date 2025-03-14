import requests
import json
import csv

# NVIDIA Workday job listings API endpoint
URL = "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/jobs"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}
PAYLOAD = {
    "appliedFacets": {},
    "limit": 20,  # Fetch up to 1000 jobs per request
    "offset": 0,
    "searchText": ""
}

def fetch_jobs():
    response = requests.post(URL, headers=HEADERS, json=PAYLOAD)
    print("response.status_code = " + str(response.status_code))
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch jobs. Status Code:", response.status_code)
        return None

def extract_jobs(data):
    jobs = []
    if not data or "jobPostings" not in data:
        print("No jobs found.")
        return jobs
    
    for job in data["jobPostings"]:
        job_id = job.get("externalPath", "N/A")
        title = job.get("title", "N/A")
        location = job.get("locationsText", "N/A")
        posted_date = job.get("postedOn", "N/A")
        job_url = f"https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite{job.get('externalPath', '')}"
        
        jobs.append({
            "Job ID": job_id,
            "Job Title": title,
            "Location": location,
            "Posted Date": posted_date,
            "Job URL": job_url
        })
    return jobs

def save_to_csv(jobs, filename="nvidia_jobs.csv"):
    keys = jobs[0].keys() if jobs else ["Job ID", "Job Title", "Location", "Posted Date", "Job URL"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(jobs)
    print(f"Saved {len(jobs)} jobs to {filename}")

def main():
    data = fetch_jobs()
    jobs = extract_jobs(data)
    if jobs:
        save_to_csv(jobs)

if __name__ == "__main__":
    main()
