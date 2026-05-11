# CV Updater

A Python tool that automatically tailors your CV/resume for specific job descriptions to increase your chances of passing through Applicant Tracking Systems (ATS).

## How It Works

1. The script analyzes job descriptions to extract important keywords and requirements
2. It then updates your CV to highlight relevant skills and experience
3. A tailored CV is generated for each job description in the input directory

## Directory Structure

```
├── cv_updater.py    # Main Python script
├── jds/             # Job descriptions directory (place your JDs here)
├── output_cv/       # Where tailored CVs will be saved
├── README.md        # This documentation
└── src_cv/          # Source CV directory (place your base CV here)
```

## Requirements

- Python 3.6+ (no external NLP libraries required)

## Installation

1. Clone this repository or download the source code
2. No additional packages are required for the basic version

## Usage

1. Place your job descriptions as text files in the `jds/` directory
2. Place your source CV as `cv.txt` in the `src_cv/` directory
3. Run the script:

```bash
python cv_updater.py
```

4. Find your tailored CVs in the `output_cv/` directory, named `cv_jobname.txt`

## AI Integration (Optional)

The script includes a placeholder function for integrating with AI services (like OpenAI's API) to improve keyword extraction. To use this feature:

1. Uncomment and complete the code in the `ai_extract_keywords()` method
2. Add your API key and update any endpoint URLs as needed
3. Install required packages for API calls: `pip install requests`

This allows the system to better understand the job descriptions and identify the most relevant keywords for your CV.

## Customization

You can modify the `cv_updater.py` script to:
- Adjust the number of keywords extracted from job descriptions
- Change how skills are prioritized and highlighted
- Customize the CV sections that get modified
- Integrate with different AI services for improved analysis

## License

This project is open source and available for personal and commercial use. 